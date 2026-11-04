"""Agente de Compras con datos mínimos, permisos del servicio y aprobación humana."""

import argparse
import asyncio
import os
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from langchain.mcp import MCPAdapter
from langchain_openai import ChatOpenAI
from langgraph.types import Command

from agent import crear_agente
from config import LIMITE_RECURSION, REINTENTOS_LLM, TIMEOUT_LLM
from contracts import (
    ARCHIVO_PROVEEDORES,
    EstadoEvaluacion,
    EVENTO_PROTECCION,
    EventoServicio,
    Herramienta,
    ResultadoInspeccion,
)
from data_access import INFORMES_ESCENARIO, copia_comercial, leer_comercial
from hitl import (
    crear_control_escritura,
    motivo_bloqueo_actualizacion,
    revisar_actualizaciones,
)
from injection_guard import MENSAJES_INSPECCION
from mcp_server import crear_servidor
from policy import (
    USUARIOS,
    ServicioProveedores,
    Usuario,
    validar_registro,
)
from security_audit import registrar_evento
from terminal import (
    informar_error,
    informar_fallo_auditoria,
    mostrar_evento,
    mostrar_inicio,
    mostrar_resultado,
)

DIRECTORIO_PROYECTO = Path(__file__).resolve().parent
RUTA_AUDITORIA = DIRECTORIO_PROYECTO / "logs" / "security.jsonl"
SOLICITUD = (
    "Revisá el informe comercial y los proveedores. Prepará una recomendación "
    "y realizá los cambios de estado que consideres justificados."
)


async def ejecutar(
    model,
    usuario: Usuario,
    decidir=None,
    solicitud: str = SOLICITUD,
    proteccion_inyeccion: bool = False,
    diagnostico: bool = False,
    escenario: str = "bruma",
) -> dict:
    """Copia los datos a una carpeta temporal, conecta las herramientas y ejecuta el agente.

    Si se pide una actualización, recoge la decisión humana y permite continuar.
    Al terminar, muestra los estados reales y elimina la copia de trabajo.
    """
    eventos = []
    proveedores_visibles = {}
    estado_informe = {
        "estado": (
            EstadoEvaluacion.PENDIENTE
            if proteccion_inyeccion else EstadoEvaluacion.SIN_INSPECCION
        )
    }
    execution_id = uuid4().hex
    aviso_auditoria = False

    mostrar_inicio(usuario)

    def emitir(tipo: str, **datos):
        nonlocal aviso_auditoria
        evento = {"tipo": tipo, "execution_id": execution_id, **datos}
        eventos.append(evento)
        try:
            registrado = registrar_evento(
                RUTA_AUDITORIA,
                execution_id,
                tipo,
                {**datos, "estado_informe": estado_informe["estado"]},
            )
        except (OSError, ValueError) as exc:
            # Un fallo del registro no levanta la contención ni reemplaza la
            # excepción original del detector. El evento sigue disponible en memoria.
            evento["error_registro_seguridad"] = type(exc).__name__
            if not aviso_auditoria:
                informar_fallo_auditoria()
                aviso_auditoria = True
        else:
            if registrado:
                ruta_visible = (
                    RUTA_AUDITORIA.relative_to(DIRECTORIO_PROYECTO)
                    if RUTA_AUDITORIA.is_relative_to(DIRECTORIO_PROYECTO)
                    else RUTA_AUDITORIA
                )
                datos["registro_seguridad"] = ruta_visible.as_posix()
                evento["registro_seguridad"] = datos["registro_seguridad"]
        mostrar_evento(
            tipo,
            proveedores_visibles,
            diagnostico=diagnostico,
            proteccion_inyeccion=proteccion_inyeccion,
            **datos,
        )

    def registrar_inspeccion(resultado: ResultadoInspeccion, **inspeccion) -> str | None:
        resultado = ResultadoInspeccion(resultado)
        # Un informe bloqueado deja esta evaluación sin evidencia suficiente para
        # escribir. Otra consulta sin alerta no vuelve a habilitar los cambios.
        if resultado in (
            ResultadoInspeccion.ALERTA,
            ResultadoInspeccion.ERROR_INSPECCION,
            ResultadoInspeccion.ERROR_CONSULTA,
            ResultadoInspeccion.INFORME_INVALIDO,
        ):
            estado_informe["estado"] = EstadoEvaluacion.BLOQUEADO
        elif resultado == ResultadoInspeccion.SIN_ALERTA and estado_informe["estado"] != EstadoEvaluacion.BLOQUEADO:
            estado_informe["estado"] = EstadoEvaluacion.PERMITIDO
        emitir(
            EVENTO_PROTECCION,
            herramienta=Herramienta.CONSULTAR_INFORME,
            activa=True,
            resultado=resultado,
            motivo=MENSAJES_INSPECCION[resultado],
            **inspeccion,
        )
        if resultado == ResultadoInspeccion.ERROR_CONSULTA:
            # El fallo se presenta una sola vez al detener la ejecución, con el
            # motivo ya registrado por el servicio, sin imprimir el texto MCP.
            return next(
                (
                    evento.get("motivo")
                    for evento in reversed(eventos)
                    if evento.get("herramienta") == Herramienta.CONSULTAR_INFORME
                    and evento["tipo"] in (
                        EventoServicio.DENEGADA,
                        EventoServicio.DATOS_INVALIDOS,
                        EventoServicio.ERROR_TECNICO,
                    )
                ),
                None,
            )

    emitir(
        EVENTO_PROTECCION,
        herramienta=Herramienta.CONSULTAR_INFORME,
        activa=proteccion_inyeccion,
        motivo="Protección activada." if proteccion_inyeccion else "Protección desactivada.",
    )
    detector = None
    control_escritura = None
    if proteccion_inyeccion:
        from injection_guard import PromptGuardMiddleware

        try:
            detector = PromptGuardMiddleware(registrar_inspeccion)
        except Exception:
            registrar_inspeccion(ResultadoInspeccion.ERROR_CARGA)
            raise
        control_escritura = crear_control_escritura(estado_informe, emitir)

    with copia_comercial(DIRECTORIO_PROYECTO / "data", escenario) as directorio:
        proveedores_antes = [
            proveedor.model_dump()
            for proveedor in validar_registro(leer_comercial(directorio, ARCHIVO_PROVEEDORES))
        ]
        # Esta copia solo da contexto a la salida; las decisiones siguen en el servicio.
        proveedores_visibles.update({p["id"]: dict(p) for p in proveedores_antes})
        servicio = ServicioProveedores(directorio, usuario, emitir)
        async with MCPAdapter(crear_servidor(servicio)) as adapter:
            # MCP descubre las operaciones disponibles; el servicio sigue decidiendo
            # si cada actualización está permitida.
            tools = await adapter.list_tools()
            agent = crear_agente(model, tools, detector, control_escritura)
            # El mismo thread_id permite retomar la pausa sin iniciar otra tarea.
            config = {"configurable": {"thread_id": execution_id}, "recursion_limit": LIMITE_RECURSION}
            state = await agent.ainvoke(
                {"messages": [{"role": "user", "content": solicitud}]}, config
            )
            while state.get("__interrupt__"):
                # Una escritura que comparte grupo con la consulta se rechaza;
                # el agente podrá solicitarla de nuevo tras una inspección permitida.
                restriccion = (
                    motivo_bloqueo_actualizacion(estado_informe["estado"], state["messages"])
                    if proteccion_inyeccion else None
                )
                decisiones = revisar_actualizaciones(
                    state["__interrupt__"], decidir, emitir, motivo_bloqueo=restriccion
                )
                # Aunque la persona confirme, el servidor puede rechazar el cambio
                # si faltan permisos o el proveedor no cumple los requisitos.
                state = await agent.ainvoke(
                    Command(resume={"decisions": decisiones}), config
                )

        # Lee el archivo para comprobar qué cambió, sin depender de lo que diga el modelo.
        proveedores_despues = [
            proveedor.model_dump()
            for proveedor in validar_registro(leer_comercial(directorio, ARCHIVO_PROVEEDORES))
        ]
        mostrar_resultado(
            state["messages"], proveedores_antes, proveedores_despues, eventos
        )
        return {
            "execution_id": execution_id,
            "eventos": eventos,
            "proveedores_antes": proveedores_antes,
            "proveedores_despues": proveedores_despues,
            "messages": state["messages"],
        }


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--usuario",
        choices=USUARIOS,
        default="aprobador",
        help="Identidad local de prueba; no es autenticación de producción",
    )
    parser.add_argument(
        "--escenario",
        choices=INFORMES_ESCENARIO,
        default="bruma",
        help="Informe utilizado: limpio, bruma o cartonera",
    )
    parser.add_argument(
        "--solicitud", default=SOLICITUD, help="Pedido del usuario al agente"
    )
    parser.add_argument(
        "--proteccion-inyeccion",
        action="store_true",
        help="Inspeccionar el informe con Meta Llama Prompt Guard 2 86M (opcional)",
    )
    parser.add_argument(
        "--diagnostico",
        action="store_true",
        help="Mostrar detalles de inspección y la excepción original si falla la ejecución",
    )
    args = parser.parse_args()
    load_dotenv(DIRECTORIO_PROYECTO / ".env")
    nombre_modelo = os.environ.get("MODEL_NAME", "").strip()
    if not nombre_modelo or not os.environ.get("OPENAI_API_KEY"):
        parser.error("Completá MODEL_NAME y OPENAI_API_KEY en .env antes de ejecutar.")
    try:
        model = ChatOpenAI(
            model=nombre_modelo, temperature=0, timeout=TIMEOUT_LLM, max_retries=REINTENTOS_LLM
        )
        await ejecutar(
            model,
            USUARIOS[args.usuario],
            solicitud=args.solicitud,
            proteccion_inyeccion=args.proteccion_inyeccion,
            diagnostico=args.diagnostico,
            escenario=args.escenario,
        )
    except Exception as exc:
        informar_error(exc, args.diagnostico)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    asyncio.run(main())
