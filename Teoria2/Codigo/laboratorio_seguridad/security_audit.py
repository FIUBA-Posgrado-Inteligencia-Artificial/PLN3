
"""Registro persistente y acotado de decisiones de seguridad, sin documentos."""

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import get_args

from config import MODELO
from contracts import (
    ARCHIVO_INFORME,
    ARCHIVO_PROVEEDORES,
    EstadoEvaluacion,
    EstadoProveedor,
    EVENTO_PROTECCION,
    EventoHost,
    EventoServicio,
    Herramienta,
    PATRON_ID_PROVEEDOR,
    ResultadoInspeccion,
)

HERRAMIENTAS = tuple(Herramienta)
ESTADOS = get_args(EstadoProveedor)


def _numero_finito(valor) -> bool:
    try:
        return type(valor) in (int, float) and math.isfinite(valor)
    except OverflowError:
        return False


def _proveedor(evento: dict, datos: dict) -> None:
    """Conserva solo identificadores y estados reconocibles, sin argumentos libres."""
    argumentos = datos.get("argumentos", {})
    if not isinstance(argumentos, dict):
        return
    proveedor_id = argumentos.get("proveedor_id")
    if isinstance(proveedor_id, str) and re.fullmatch(PATRON_ID_PROVEEDOR, proveedor_id):
        evento["provider_id"] = proveedor_id
    estado = argumentos.get("nuevo_estado")
    if estado in ESTADOS:
        evento["requested_state"] = estado


def _inspeccion(evento: dict, datos: dict) -> None:
    """Una inspección completada requiere sus metadatos reales, no valores inventados."""
    score = datos.get("puntaje_maximo")
    threshold = datos.get("umbral")
    segments = datos.get("segmentos")
    if (
        datos.get("modelo") != MODELO
        or not _numero_finito(score) or not 0 <= score <= 1
        or not _numero_finito(threshold) or not 0 <= threshold <= 1
        or type(segments) is not int or segments <= 0 or not _numero_finito(segments)
    ):
        raise ValueError("La inspección no contiene metadatos válidos para la auditoría")
    evento.update(detector=MODELO, score=score, threshold=threshold, segments=segments)


def registrar_evento(
    ruta: Path, execution_id: str, tipo: str, datos: dict
) -> bool:
    """Persiste decisiones y resultados; omite únicamente eventos definidos como ruido."""
    if not isinstance(execution_id, str) or not execution_id:
        raise ValueError("Falta el identificador de ejecución de la auditoría")
    if "herramienta" in datos and datos["herramienta"] not in HERRAMIENTAS:
        raise ValueError("La auditoría recibió una herramienta no reconocida")
    if tipo in (
        EventoHost.REVISION_SOLICITADA,
        EventoHost.LOTE_REVISION,
        EventoHost.LOTE_RESUELTO,
        EventoServicio.INICIADA,
    ):
        return False
    if tipo == EVENTO_PROTECCION and "resultado" not in datos:
        if type(datos.get("activa")) is bool:
            return False
        raise ValueError("La auditoría no reconoce la configuración del detector")

    evento = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "execution_id": execution_id,
    }
    if tipo == EVENTO_PROTECCION:
        resultado = datos.get("resultado")
        if resultado in (ResultadoInspeccion.ALERTA, ResultadoInspeccion.SIN_ALERTA):
            alerta = resultado == ResultadoInspeccion.ALERTA
            evento.update(
                event="prompt_injection_alert" if alerta else "prompt_injection_inspected",
                severity="warning" if alerta else "info",
                source=ARCHIVO_INFORME,
                tool=Herramienta.CONSULTAR_INFORME,
                decision="block" if alerta else "allow",
                outcome="blocked" if alerta else "no_alert",
            )
            if alerta:
                evento["action"] = "document_blocked_read_only"
            _inspeccion(evento, datos)
        elif resultado in (
            ResultadoInspeccion.ERROR_CARGA,
            ResultadoInspeccion.ERROR_INSPECCION,
        ):
            carga = resultado == ResultadoInspeccion.ERROR_CARGA
            evento.update(
                event="security_control_failed",
                severity="error",
                stage="load" if carga else "inspection",
                detector=MODELO,
                action="execution_stopped",
                outcome="failed",
            )
            if not carga:
                evento.update(source=ARCHIVO_INFORME, tool=Herramienta.CONSULTAR_INFORME)
        elif resultado in (
            ResultadoInspeccion.ERROR_CONSULTA,
            ResultadoInspeccion.INFORME_INVALIDO,
        ):
            consulta = resultado == ResultadoInspeccion.ERROR_CONSULTA
            evento.update(
                event="report_query_failed" if consulta else "report_invalid",
                severity="error",
                stage="query" if consulta else "validation",
                source=ARCHIVO_INFORME,
                tool=Herramienta.CONSULTAR_INFORME,
                outcome="failed" if consulta else "invalid",
            )
            if consulta:
                evento["action"] = "execution_stopped"
        else:
            raise ValueError("La auditoría recibió un resultado de inspección no reconocido")
    elif tipo in (EventoHost.SOLICITUD_AUTORIZADA, EventoHost.SOLICITUD_RECHAZADA, EventoHost.ESCRITURA_BLOQUEADA):
        if datos.get("herramienta", Herramienta.ACTUALIZAR_PROVEEDOR) != Herramienta.ACTUALIZAR_PROVEEDOR:
            raise ValueError("La decisión del host no corresponde a una actualización")
        evento.update(source=ARCHIVO_PROVEEDORES, tool=Herramienta.ACTUALIZAR_PROVEEDOR)
        _proveedor(evento, datos)
        if tipo == EventoHost.ESCRITURA_BLOQUEADA:
            evento.update(
                event="host_write_blocked",
                severity="warning",
                required_source=ARCHIVO_INFORME,
                action="write_not_executed",
                decision="block",
                outcome="not_executed",
            )
            estado = datos.get("estado_informe")
            if estado in (
                EstadoEvaluacion.PENDIENTE,
                EstadoEvaluacion.PERMITIDO,
                EstadoEvaluacion.BLOQUEADO,
            ):
                evento["report_state"] = estado
        else:
            autorizada = tipo == EventoHost.SOLICITUD_AUTORIZADA
            evento.update(
                event="human_review",
                severity="info",
                decision="approve" if autorizada else "reject",
                outcome="review_complete" if autorizada else "not_executed",
            )
    elif tipo in (
        EventoServicio.COMPLETADA,
        EventoServicio.DENEGADA,
        EventoServicio.REGLA_INCUMPLIDA,
        EventoServicio.DATOS_INVALIDOS,
        EventoServicio.ERROR_TECNICO,
    ):
        herramienta = datos.get("herramienta")
        if herramienta not in HERRAMIENTAS:
            raise ValueError("El resultado del servicio no identifica una herramienta válida")
        actualizacion = herramienta == Herramienta.ACTUALIZAR_PROVEEDOR
        if tipo == EventoServicio.COMPLETADA:
            if not actualizacion:
                return False
            cambio = datos.get("cambio")
            if type(cambio) is not bool:
                raise ValueError("La actualización completada no indica si cambió el registro")
            evento.update(
                event="provider_update",
                severity="info",
                decision="allow",
                outcome="updated" if cambio else "no_change",
            )
        else:
            categorias = {
                EventoServicio.DENEGADA: ("server_permission_denied", "warning", "not_executed"),
                EventoServicio.REGLA_INCUMPLIDA: ("server_business_rule_rejected", "warning", "not_executed"),
                EventoServicio.DATOS_INVALIDOS: ("server_validation_failed", "warning", "invalid"),
                EventoServicio.ERROR_TECNICO: ("server_technical_error", "error", "failed"),
            }
            nombre, severidad, outcome = categorias[tipo]
            evento.update(event=nombre, severity=severidad, outcome=outcome)
            if tipo in (EventoServicio.DENEGADA, EventoServicio.REGLA_INCUMPLIDA):
                evento["decision"] = "deny"
        evento.update(
            tool=herramienta,
            source=ARCHIVO_INFORME if herramienta == Herramienta.CONSULTAR_INFORME else ARCHIVO_PROVEEDORES,
        )
        if actualizacion:
            _proveedor(evento, datos)
    else:
        raise ValueError("La auditoría recibió un tipo de evento no reconocido")

    # Se serializa esta selección, nunca documentos, motivos, argumentos completos
    # o excepciones recibidos con el evento.
    linea = json.dumps(evento, ensure_ascii=False, allow_nan=False)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("a", encoding="utf-8", newline="\n") as archivo:
        archivo.write(linea + "\n")
    return True
