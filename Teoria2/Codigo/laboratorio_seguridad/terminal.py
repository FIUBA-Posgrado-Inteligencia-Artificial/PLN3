
"""Mensajes de la práctica y presentación de resultados en terminal."""

import sys
import traceback

from langchain_core.messages import AIMessage
from openai import APIConnectionError, APIStatusError, APITimeoutError

from contracts import (
    EVENTO_PROTECCION,
    EventoHost,
    EventoServicio,
    Herramienta,
    ResultadoInspeccion,
)
from data_access import ErrorAccesoDatos
from injection_guard import ErrorProteccionInyeccion
from policy import ErrorValidacion

NOMBRES_OPERACION = {
    Herramienta.CONSULTAR_INFORME: "Consulta del informe comercial",
    Herramienta.CONSULTAR_PROVEEDORES: "Consulta de proveedores",
    Herramienta.ACTUALIZAR_PROVEEDOR: "Actualización del estado de un proveedor",
}


def texto_terminal(valor, limite: int = 120) -> str:
    """Acota los campos visibles y escapa caracteres de control, sin comillas externas."""
    return repr(str(valor))[1:-1][:limite]


def mostrar_evento(
    tipo: EventoServicio | EventoHost | str,
    proveedores: dict,
    *,
    diagnostico: bool = False,
    proteccion_inyeccion: bool = False,
    **datos,
) -> None:
    """Presenta solo las decisiones y resultados relevantes para la práctica."""
    if tipo == EventoServicio.INICIADA:
        return
    if (
        proteccion_inyeccion
        and tipo in (
            EventoServicio.COMPLETADA,
            EventoServicio.DENEGADA,
            EventoServicio.DATOS_INVALIDOS,
            EventoServicio.ERROR_TECNICO,
        )
        and datos.get("herramienta") == Herramienta.CONSULTAR_INFORME
    ):
        return
    if tipo == EVENTO_PROTECCION:
        if "resultado" not in datos:
            estado = "activada (se inspeccionará al consultar el informe)" if datos["activa"] else "desactivada"
            print(f"Detector de prompt injection (Prompt Guard 2): {estado}.\n")
        elif datos["resultado"] == ResultadoInspeccion.ALERTA:
            print("Informe comercial: bloqueado por Prompt Guard 2.")
            print("ALERTA DE SEGURIDAD")
            print("  Tipo: posible prompt injection indirecta")
            print("  Origen: informe.txt")
            print("  Acción: informe bloqueado y evaluación en modo solo lectura")
            if datos.get("registro_seguridad"):
                print("  Evento registrado:", texto_terminal(datos["registro_seguridad"], limite=250))
            print("La evaluación continúa sin permitir modificaciones.")
        elif datos["resultado"] == ResultadoInspeccion.SIN_ALERTA:
            print("Informe inspeccionado con Prompt Guard 2: sin alerta; contenido permitido.")
        if diagnostico and datos.get("resultado") in (
            ResultadoInspeccion.ALERTA, ResultadoInspeccion.SIN_ALERTA
        ):
            print("  Detector:", datos["modelo"])
            print("  Segmentos analizados:", datos["segmentos"])
            print(f"  Puntaje máximo: {datos['puntaje_maximo']:.6f}")
            print(f"  Umbral de bloqueo: > {datos['umbral']:.2f}")
            print(
                f"  Clase maliciosa: índice {datos['clase_maliciosa']} "
                f"({datos['etiqueta_clase']})."
            )
        return
    if tipo == EventoHost.LOTE_REVISION:
        print(f"\nEl agente propuso {datos['total']} actualizaciones.")
        print("Se requiere revisar todas antes de continuar.")
        return
    if tipo == EventoHost.LOTE_RESUELTO:
        if datos["confirmadas"]:
            print(f"\nContinuando con las operaciones autorizadas: {datos['confirmadas']}.")
        else:
            print("\nNo se autorizó ninguna actualización.")
        return
    if tipo == EventoHost.SOLICITUD_AUTORIZADA:
        if datos.get("total", 1) > 1:
            print("Solicitud autorizada. Se enviará al servidor al terminar la revisión.")
        else:
            print("Solicitud autorizada. El servidor verificará si puede ejecutarla.")
        return
    if tipo == EventoHost.SOLICITUD_RECHAZADA:
        print("Solicitud no autorizada. No se ejecutó la herramienta.")
        return
    herramienta = datos.get("herramienta")
    argumentos = datos.get("argumentos", {})
    identificador = argumentos.get("proveedor_id")
    proveedor = proveedores.get(identificador, {}) if isinstance(identificador, str) else {}
    nombre = texto_terminal(proveedor.get("nombre", "Proveedor no identificado"))
    referencia = f"{nombre} ({texto_terminal(identificador)})"
    if tipo in (EventoHost.REVISION_SOLICITADA, EventoHost.ESCRITURA_BLOQUEADA):
        numero = (
            f"[{datos['posicion']}/{datos['total']}] "
            if datos.get("total", 1) > 1 else ""
        )
        etiqueta = (
            "Actualización bloqueada por el host"
            if tipo == EventoHost.ESCRITURA_BLOQUEADA else "Actualización propuesta"
        )
        print(f"\n{numero}{etiqueta}: {referencia}")
        print(
            f"  Estado: {texto_terminal(proveedor.get('estado', 'desconocido'))}"
            f" → {texto_terminal(argumentos.get('nuevo_estado', 'no indicado'))}"
        )
        if tipo == EventoHost.ESCRITURA_BLOQUEADA:
            print("  Motivo:", texto_terminal(datos["motivo"], limite=500))
            print("No se ejecutó la herramienta de actualización.")
    elif tipo == EventoServicio.COMPLETADA:
        if herramienta == Herramienta.ACTUALIZAR_PROVEEDOR:
            estado = argumentos.get("nuevo_estado")
            cambio = datos.get("cambio")
            if cambio is False:
                print(
                    f"Servidor: {referencia} ya tenía el estado "
                    f"{texto_terminal(estado)}. Sin cambios."
                )
            elif cambio is True:
                print(
                    f"Servidor: actualización realizada para {referencia}. "
                    f"Estado: {texto_terminal(estado)}."
                )
            else:
                print(
                    f"Servidor: operación completada para {referencia}. "
                    "No se informó si hubo cambios."
                )
            # Actualiza solo la vista usada por las próximas propuestas; el estado
            # efectivo se comprueba al final leyendo el registro del servicio.
            if proveedor and isinstance(cambio, bool):
                proveedor["estado"] = estado
        elif herramienta == Herramienta.CONSULTAR_INFORME:
            print("Informe comercial consultado.")
        elif herramienta == Herramienta.CONSULTAR_PROVEEDORES:
            print("Estados y requisitos de los proveedores consultados.")
    else:
        motivos = {
            EventoServicio.DENEGADA: "operación denegada",
            EventoServicio.REGLA_INCUMPLIDA: "requisitos pendientes",
            EventoServicio.DATOS_INVALIDOS: "datos inválidos",
            EventoServicio.ERROR_TECNICO: "error técnico",
        }
        categoria = motivos.get(tipo)
        if categoria is None:
            print("Se recibió un evento no reconocido; sus detalles se conservan en memoria.")
            return
        operacion = (
            referencia if herramienta == Herramienta.ACTUALIZAR_PROVEEDOR
            else NOMBRES_OPERACION.get(herramienta, "Consulta")
        )
        print(f"Servidor: {categoria} — {operacion}.")
        print("  Motivo:", texto_terminal(datos.get("motivo", "No informado"), limite=500))


def mostrar_inicio(usuario) -> None:
    """Presenta el encabezado una sola vez por ejecución."""
    print("Laboratorio de seguridad de agentes de IA")
    print("Usuario:", usuario.nombre)


def mostrar_resultado(messages, proveedores_antes, proveedores_despues, eventos) -> None:
    """Compara el registro real sin dar por cierto lo que afirme el modelo."""
    ultimo_ai = next(
        (mensaje for mensaje in reversed(messages) if isinstance(mensaje, AIMessage)), None
    )
    # No presenta resultados de herramientas ni propuestas pendientes como si
    # fueran una respuesta final, ni recupera una conclusión anterior a esas propuestas.
    if (
        ultimo_ai is not None and ultimo_ai.content
        and not ultimo_ai.tool_calls and not ultimo_ai.invalid_tool_calls
    ):
        print("\nRecomendación del agente:\n", ultimo_ai.content)
    else:
        print("\nEl agente no produjo una recomendación final.")
    proteccion = [e for e in eventos if e["tipo"] == EVENTO_PROTECCION]
    if any(e.get("activa") for e in proteccion) and not any(
        e.get("resultado") in (
            ResultadoInspeccion.ALERTA, ResultadoInspeccion.SIN_ALERTA
        ) for e in proteccion
    ):
        print("\nEl detector no completó ninguna inspección del informe en esta ejecución.")
    if not any(
        evento["tipo"] in (
            EventoHost.REVISION_SOLICITADA, EventoHost.ESCRITURA_BLOQUEADA
        )
        for evento in eventos
    ):
        print("\nEl agente no solicitó actualizaciones; no hubo operaciones para autorizar.")
    print("\nEstados de la copia de trabajo al terminar:")
    estados_anteriores = {
        proveedor["id"]: proveedor["estado"] for proveedor in proveedores_antes
    }
    for proveedor in proveedores_despues:
        print(
            f"  {proveedor['nombre']}: {estados_anteriores[proveedor['id']]} → {proveedor['estado']}"
        )
    print("Los archivos originales no se modificaron.")


def informar_fallo_auditoria() -> None:
    """Advierte una pérdida de persistencia sin debilitar los controles de la práctica."""
    print(
        "Aviso: no se pudo guardar el evento de seguridad; queda solo en memoria. "
        "Las restricciones de seguridad siguen activas.",
        file=sys.stderr,
    )


def informar_error(exc: Exception, diagnostico: bool) -> None:
    """Un único punto presenta el fallo; el diagnóstico conserva la cadena completa."""
    if diagnostico:
        traceback.print_exception(exc, file=sys.stderr)
        return
    if isinstance(exc, ErrorProteccionInyeccion):
        mensaje = str(exc)
    elif isinstance(exc, (ErrorAccesoDatos, ErrorValidacion)):
        mensaje = str(exc)
    elif isinstance(exc, APITimeoutError):
        mensaje = "OpenAI no respondió dentro del tiempo de espera."
    elif isinstance(exc, APIConnectionError):
        mensaje = "No se pudo conectar con OpenAI. Revisá la conexión y la configuración de red."
    elif isinstance(exc, APIStatusError):
        mensajes = {
            401: "OpenAI rechazó las credenciales. Revisá OPENAI_API_KEY.",
            403: "OpenAI denegó la solicitud. Revisá los permisos de tu cuenta.",
            404: "OpenAI no encontró el recurso solicitado. Revisá MODEL_NAME y el acceso al modelo.",
            429: "OpenAI rechazó la solicitud por un límite de uso o cuota.",
        }
        mensaje = mensajes.get(exc.status_code, f"OpenAI devolvió un error al ejecutar el modelo (HTTP {exc.status_code}).")
    elif isinstance(exc, (FileNotFoundError, PermissionError, IsADirectoryError, NotADirectoryError)):
        mensaje = "No se pudo acceder a los archivos de la práctica."
    else:
        mensaje = f"La ejecución se interrumpió por un error inesperado ({type(exc).__name__})."
    print(f"Error: {mensaje} Usá --diagnostico para ver la causa original.", file=sys.stderr)
