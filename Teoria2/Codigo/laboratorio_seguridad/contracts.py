"""Nombres y valores compartidos entre los módulos del laboratorio.

No contiene reglas de negocio ni controles de seguridad: solo sus nombres comunes.
"""

from enum import StrEnum
from typing import Literal


# Archivos comerciales que el laboratorio permite utilizar.
ARCHIVO_INFORME = "informe.txt"
ARCHIVO_PROVEEDORES = "proveedores.json"

PATRON_ID_PROVEEDOR = r"^PROV-[0-9]{3}$"
EstadoProveedor = Literal["PENDIENTE", "APROBADO", "RECHAZADO"]


class Herramienta(StrEnum):
    """Nombres de las herramientas publicadas mediante MCP."""

    CONSULTAR_INFORME = "consultar_informe"
    CONSULTAR_PROVEEDORES = "consultar_proveedores"
    ACTUALIZAR_PROVEEDOR = "actualizar_estado_proveedor"


class EstadoEvaluacion(StrEnum):
    """Estado del control de escrituras del host, no calidad del informe."""

    SIN_INSPECCION = "sin_inspeccion"  # Detector desactivado.
    PENDIENTE = "pendiente"           # Inspección todavía no completada.
    PERMITIDO = "permitido"           # Se permite solicitar cambios al host.
    BLOQUEADO = "bloqueado"           # No se permiten cambios en esta ejecución.


class EventoHost(StrEnum):
    """Revisión humana y bloqueos de operaciones en el host."""

    REVISION_SOLICITADA = "revision_solicitada"
    SOLICITUD_AUTORIZADA = "solicitud_autorizada"
    SOLICITUD_RECHAZADA = "solicitud_rechazada"
    ESCRITURA_BLOQUEADA = "escritura_bloqueada_host"
    LOTE_REVISION = "lote_revision"
    LOTE_RESUELTO = "lote_resuelto"


class EventoServicio(StrEnum):
    """Eventos generados al ejecutar herramientas del servicio."""

    INICIADA = "iniciada"
    COMPLETADA = "completada"
    DENEGADA = "denegada"
    REGLA_INCUMPLIDA = "regla_incumplida"
    DATOS_INVALIDOS = "datos_invalidos"
    ERROR_TECNICO = "error_tecnico"


EVENTO_PROTECCION = "proteccion_inyeccion"


class ResultadoInspeccion(StrEnum):
    """Resultados de la inspección opcional con Prompt Guard."""

    ALERTA = "alerta"
    SIN_ALERTA = "sin_alerta"
    ERROR_CARGA = "error_carga"
    ERROR_CONSULTA = "error_consulta"
    ERROR_INSPECCION = "error"
    INFORME_INVALIDO = "informe_invalido"
