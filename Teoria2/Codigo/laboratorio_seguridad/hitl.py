
"""Gestiona la revisión humana y el bloqueo de cambios no permitidos.

HITL (Human-in-the-Loop) permite que una persona revise las operaciones
propuestas por el agente antes de que lleguen al servidor.
"""

from langchain.agents.middleware import HumanInTheLoopMiddleware, wrap_tool_call
from langchain_core.messages import ToolMessage

from contracts import EstadoEvaluacion, EventoHost, Herramienta


def crear_middleware_hitl():
    """Solicita autorización humana antes de ejecutar cambios de estado."""

    # LangChain interrumpe las actualizaciones y espera una decisión.
    # Autorizar la solicitud no garantiza que el servidor acepte el cambio.
    return HumanInTheLoopMiddleware(
        interrupt_on={
            Herramienta.ACTUALIZAR_PROVEEDOR: {
                "allowed_decisions": ["approve", "reject"]
            },
        }
    )


def motivo_bloqueo_actualizacion(estado_informe: str, mensajes) -> str | None:
    """Comprueba si hay información suficiente para permitir cambios.

    Cuando se utiliza Prompt Guard, el informe debe haber sido
    inspeccionado sin alertas. Si fue bloqueado, la evaluación
    continúa en modo solo lectura.
    """

    if estado_informe == EstadoEvaluacion.BLOQUEADO:
        return (
            "El informe fue bloqueado por seguridad. "
            "Esta evaluación continúa en modo solo lectura."
        )

    if estado_informe != EstadoEvaluacion.PERMITIDO:
        return (
            "Primero debe consultarse e inspeccionarse el informe. "
            "Si corresponde, solicitá nuevamente el cambio después de la inspección."
        )

    # El agente puede pedir varias herramientas al mismo tiempo.
    # Si solicita leer el informe y modificar un estado en el mismo paso,
    # se impide el cambio hasta completar la inspección.
    ultimo_mensaje = next(
        (mensaje for mensaje in reversed(mensajes) if mensaje.type == "ai"),
        None,
    )

    if ultimo_mensaje is not None and any(
        llamada["name"] == Herramienta.CONSULTAR_INFORME
        for llamada in ultimo_mensaje.tool_calls
    ):
        return (
            "El informe se está consultando en este mismo paso. "
            "Esperá a que se inspeccione y, si corresponde, solicitá nuevamente el cambio."
        )

    return None


def crear_control_escritura(estado_informe: dict, emitir):
    """Bloquea las actualizaciones que no están permitidas en esta evaluación."""

    # Este control se aplica antes de ejecutar la herramienta.
    # Aunque el agente solicite un cambio, el host puede impedirlo
    # sin consultar al usuario ni enviar la operación al servidor.
    @wrap_tool_call
    async def controlar_escritura(request, handler):
        if request.tool_call["name"] == Herramienta.ACTUALIZAR_PROVEEDOR:
            motivo = motivo_bloqueo_actualizacion(
                estado_informe["estado"],
                request.state["messages"],
            )

            if motivo:
                emitir(
                    EventoHost.ESCRITURA_BLOQUEADA,
                    herramienta=request.tool_call["name"],
                    argumentos=request.tool_call["args"],
                    motivo=motivo,
                )

                # El agente recibe el motivo del rechazo para poder
                # continuar sin ejecutar la actualización.
                return ToolMessage(
                    content=motivo,
                    tool_call_id=request.tool_call["id"],
                    name=request.tool_call["name"],
                    status="error",
                )

        return await handler(request)

    return controlar_escritura


def revisar_actualizaciones(
    interrupciones, decidir, emitir, motivo_bloqueo: str | None = None
) -> list[dict]:
    """Presenta los cambios propuestos y recoge las decisiones humanas.

    Autorizar un cambio permite enviarlo al servidor, que todavía debe
    comprobar los permisos y las reglas comerciales.
    """

    # Una interrupción puede contener varias actualizaciones propuestas.
    acciones = [
        accion
        for interrupcion in interrupciones
        for accion in interrupcion.value["action_requests"]
    ]

    decisiones = []
    total = len(acciones)
    confirmadas = 0

    if total > 1 and motivo_bloqueo is None:
        emitir(EventoHost.LOTE_REVISION, total=total)

    for posicion, accion in enumerate(acciones, start=1):
        datos = {
            "herramienta": accion["name"],
            "argumentos": accion["args"],
            "posicion": posicion,
            "total": total,
        }

        # Si la operación ya está bloqueada, no tiene sentido
        # solicitar una autorización que no podrá ejecutarse.
        if motivo_bloqueo is not None:
            emitir(
                EventoHost.ESCRITURA_BLOQUEADA,
                motivo=motivo_bloqueo,
                **datos,
            )

            decisiones.append({
                "type": "reject",
                "message": motivo_bloqueo,
            })
            continue

        emitir(EventoHost.REVISION_SOLICITADA, **datos)

        # En la práctica, la persona decide desde la terminal.
        # La función decidir permite proporcionar esa respuesta desde el código.
        if decidir is not None:
            autorizada = decidir(accion)
        else:
            autorizada = (
                input("¿Autorizás intentar este cambio? [S/N]: ")
                .strip()
                .lower()
                == "s"
            )

        emitir(
            EventoHost.SOLICITUD_AUTORIZADA
            if autorizada else EventoHost.SOLICITUD_RECHAZADA,
            **datos,
        )

        # LangGraph utiliza approve y reject para continuar o cancelar
        # cada llamada a herramienta que quedó pendiente de revisión.
        decision = {"type": "approve" if autorizada else "reject"}

        if autorizada:
            confirmadas += 1
        else:
            decision["message"] = (
                "No se autorizó este cambio. Continuá sin ejecutarlo."
            )

        decisiones.append(decision)

    if total > 1 and motivo_bloqueo is None:
        emitir(
            EventoHost.LOTE_RESUELTO,
            total=total,
            confirmadas=confirmadas,
        )

    return decisiones
