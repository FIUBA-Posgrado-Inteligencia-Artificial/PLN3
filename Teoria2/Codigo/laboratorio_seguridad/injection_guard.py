
"""Inspección opcional del informe con el clasificador oficial de Meta."""

import asyncio
import math
from threading import Lock

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphBubbleUp

from config import MAX_TOKENS, MODELO, UMBRAL
from contracts import EVENTO_PROTECCION, Herramienta, ResultadoInspeccion

CLASE_MALICIOSA = 1


MENSAJES_INSPECCION = {
    ResultadoInspeccion.ALERTA: "Informe comercial: bloqueado por Prompt Guard 2. "
    "El contenido no se entregó al agente. La evaluación continúa en modo solo lectura.",
    ResultadoInspeccion.SIN_ALERTA: "Informe inspeccionado sin alerta; continúa el contenido original.",
    ResultadoInspeccion.ERROR_INSPECCION: "Error de inspección; el informe no se entrega al modelo.",
    ResultadoInspeccion.ERROR_CARGA: "El detector no pudo cargarse; no se inspeccionó ningún documento.",
    ResultadoInspeccion.ERROR_CONSULTA: "La consulta MCP falló; no se inspeccionó el informe.",
    ResultadoInspeccion.INFORME_INVALIDO: "La respuesta no contiene un informe de texto válido.",
}


class ErrorProteccionInyeccion(RuntimeError):
    """Identifica la etapa que falló y conserva la excepción original como causa."""

    def __init__(self, mensaje: str, etapa: str):
        super().__init__(mensaje)
        self.etapa = etapa


def mensaje_error_carga(exc: Exception) -> str:
    """Traduce causas conocidas sin deducir el estado de acceso a partir de un 403."""
    causas = []
    actual = exc
    while actual is not None and id(actual) not in {id(e) for e in causas}:
        causas.append(actual)
        actual = actual.__cause__ or actual.__context__
    if any("torchvision::nms does not exist" in str(e) for e in causas):
        return (
            "La instalación de PyTorch y TorchVision es incompatible "
            "(torchvision::nms). Usá un entorno virtual limpio con "
            "requirements-inyeccion.txt; el detector no necesita TorchVision."
        )
    if any(
        isinstance(e, ModuleNotFoundError)
        and (e.name or "").split(".")[0] in {"torch", "transformers", "sentencepiece"}
        for e in causas
    ):
        return "Faltan dependencias del detector. Instalá requirements-inyeccion.txt."
    try:
        from huggingface_hub.utils import HfHubHTTPError
    except ImportError:
        return "No se pudo cargar Prompt Guard 2. La causa no pudo determinarse."
    for error in causas:
        if not isinstance(error, HfHubHTTPError):
            continue
        respuesta = error.response
        estado = respuesta.status_code if respuesta is not None else None
        # El mensaje del servidor puede precisar lo que el código HTTP no distingue.
        detalle = (error.server_message or "").lower()
        if not detalle and respuesta is not None:
            detalle = respuesta.headers.get("X-Error-Message", "").lower()
        if not detalle:
            detalle = str(error).lower()
        if any(texto in detalle for texto in ("awaiting a review", "pending approval", "pending review")):
            return "Hugging Face indica que la solicitud de acceso está pendiente de aprobación."
        if any(texto in detalle for texto in ("has been rejected", "was rejected", "request has been denied")):
            return "Hugging Face indica que la solicitud de acceso fue rechazada."
        if any(texto in detalle for texto in ("invalid credentials", "invalid token", "invalid username or password")):
            return "Hugging Face informa credenciales incorrectas. Revisá HF_TOKEN."
        if estado == 401:
            return "Hugging Face no pudo autenticar la descarga. Revisá HF_TOKEN y el acceso de tu cuenta."
        if estado == 403:
            return "Hugging Face denegó el acceso al modelo. Revisá los permisos; la respuesta no precisa el estado de la solicitud."
        return "Falló la descarga de Prompt Guard 2 desde Hugging Face. No se confirmó un problema de permisos."
    return "No se pudo cargar o descargar Prompt Guard 2. La causa no pudo determinarse."


class PromptGuardMiddleware(AgentMiddleware):
    """Filtra únicamente la respuesta del informe, antes de entregarla al LLM."""

    def __init__(self, emitir):
        self.emitir = emitir
        self.bloqueo = Lock()
        try:
            # Estas dependencias y los pesos se cargan solo al activar la opción.
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.torch = torch
            self.tokenizer = AutoTokenizer.from_pretrained(MODELO)
            if not self.tokenizer.is_fast:
                raise ValueError("La segmentación requiere el tokenizer rápido oficial")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                MODELO, use_safetensors=True
            ).to("cpu").eval()
            if self.model.config.num_labels != 2:
                raise ValueError("El detector debe ser un clasificador binario")
        except Exception as exc:
            raise ErrorProteccionInyeccion(
                mensaje_error_carga(exc)
                + " No se inspeccionó ningún documento y la ejecución se detuvo.",
                etapa="carga",
            ) from exc

    def inspeccionar(self, texto: str) -> dict:
        # Usa el tokenizer oficial; los desbordamientos evitan descartar el final.
        with self.bloqueo:
            entradas = self.tokenizer(
                texto,
                max_length=MAX_TOKENS,
                truncation=True,
                return_overflowing_tokens=True,
                return_special_tokens_mask=True,
                padding=True,
                return_tensors="pt",
            )
            entradas.pop("overflow_to_sample_mapping", None)
            especiales = entradas.pop("special_tokens_mask")
            originales = self.tokenizer(
                texto, add_special_tokens=False, truncation=False
            )["input_ids"]
            conservados = entradas["input_ids"][
                (especiales == 0) & (entradas["attention_mask"] == 1)
            ].tolist()
            # Comprueba la cobertura con el tokenizer cargado, incluidos los límites
            # entre segmentos. Si se perdieron tokens, no permite continuar.
            if conservados != originales:
                raise ValueError("La segmentación no conserva todos los tokens del informe")
            cantidad = entradas["input_ids"].shape[0]
            if cantidad == 0 or entradas["input_ids"].shape[1] > MAX_TOKENS:
                raise ValueError("Segmentación del documento inválida")
            maximo = 0.0
            with self.torch.no_grad():
                for inicio in range(0, cantidad, 4):
                    lote = {k: v[inicio : inicio + 4] for k, v in entradas.items()}
                    logits = self.model(**lote).logits
                    probabilidades = self.torch.softmax(logits, dim=-1)[:, CLASE_MALICIOSA]
                    if len(probabilidades) != len(lote["input_ids"]):
                        raise ValueError("El detector no devolvió un resultado por segmento")
                    for valor in probabilidades.tolist():
                        if not math.isfinite(valor) or not 0 <= valor <= 1:
                            raise ValueError("El detector devolvió un resultado inválido")
                        maximo = max(maximo, valor)
            # La receta oficial de Meta usa la clase 1 para contenido malicioso.
            # Los pesos pueden llamarla LABEL_1; no se deduce su significado del nombre.
            return {
                "alerta": maximo > UMBRAL,
                "segmentos": cantidad,
                "puntaje_maximo": maximo,
                "umbral": UMBRAL,
                "etiqueta_clase": self.model.config.id2label[CLASE_MALICIOSA],
            }

    async def awrap_tool_call(self, request, handler):
        if request.tool_call["name"] != Herramienta.CONSULTAR_INFORME:
            return await handler(request)
        try:
            resultado = await handler(request)
            if isinstance(resultado, ToolMessage) and resultado.status == "error":
                raise RuntimeError("consultar_informe devolvió un resultado MCP con estado de error")
        except GraphBubbleUp:
            # Las pausas de LangGraph deben llegar al host sin convertirse en fallos.
            raise
        except Exception as exc:
            if isinstance(exc, BaseExceptionGroup) and exc.subgroup(GraphBubbleUp):
                raise
            # Una consulta fallida no es un documento ni un resultado del clasificador.
            motivo = self.emitir(ResultadoInspeccion.ERROR_CONSULTA)
            detalle = f" Motivo del servicio: {motivo}." if isinstance(motivo, str) else ""
            raise ErrorProteccionInyeccion(
                "La consulta del informe falló." + detalle
                + " No se inspeccionó el documento y la evaluación se detuvo.",
                etapa="consulta",
            ) from exc
        try:
            if not isinstance(resultado, ToolMessage):
                raise ValueError("La herramienta no devolvió un informe de texto válido.")
            contenido = resultado.content
            if isinstance(contenido, str):
                texto = contenido
            elif isinstance(contenido, list) and all(
                isinstance(bloque, dict)
                and bloque.get("type") == "text"
                and isinstance(bloque.get("text"), str)
                for bloque in contenido
            ):
                texto = "\n".join(bloque["text"] for bloque in contenido)
            else:
                raise ValueError("La respuesta del informe tiene un formato no compatible.")
            if not texto.strip():
                raise ValueError("El informe está vacío.")
        except ValueError as exc:
            self.emitir(ResultadoInspeccion.INFORME_INVALIDO)
            raise ErrorProteccionInyeccion(
                f"{exc} La evaluación se detuvo sin habilitar actualizaciones.",
                etapa="validacion",
            ) from exc
        try:
            inspeccion = await asyncio.to_thread(self.inspeccionar, texto)
        except GraphBubbleUp:
            raise
        except Exception as exc:
            if isinstance(exc, BaseExceptionGroup) and exc.subgroup(GraphBubbleUp):
                raise
            self.emitir(ResultadoInspeccion.ERROR_INSPECCION)
            raise ErrorProteccionInyeccion(
                "No se pudo inspeccionar el informe. La ejecución se detuvo sin entregarlo al modelo.",
                etapa="inspeccion",
            ) from exc
        alerta = inspeccion["alerta"]
        self.emitir(
            ResultadoInspeccion.ALERTA if alerta else ResultadoInspeccion.SIN_ALERTA,
            modelo=MODELO,
            clase_maliciosa=CLASE_MALICIOSA,
            segmentos=inspeccion["segmentos"],
            puntaje_maximo=inspeccion["puntaje_maximo"],
            umbral=inspeccion["umbral"],
            etiqueta_clase=inspeccion["etiqueta_clase"],
        )
        if alerta:
            # El informe es una fuente no confiable. Ante una alerta, sustituimos
            # el mensaje completo para no entregar su contenido ni sus artefactos.
            return ToolMessage(
                content="Consulta bloqueada por una alerta de prompt injection. "
                "El informe no se entregó al modelo. La evaluación continúa en modo "
                "solo lectura: el host no permite modificar estados con esta fuente bloqueada.",
                tool_call_id=request.tool_call["id"],
                name=Herramienta.CONSULTAR_INFORME,
                status="error",
            )
        return resultado
