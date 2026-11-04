
"""Permisos y reglas de negocio del servidor, independientes del modelo y de HITL."""

import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Annotated, Callable, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from contracts import (
    ARCHIVO_INFORME,
    ARCHIVO_PROVEEDORES,
    EstadoProveedor,
    EventoServicio,
    Herramienta,
    PATRON_ID_PROVEEDOR,
)
from data_access import ErrorAccesoDatos, guardar_proveedores, leer_comercial

ProveedorId = Annotated[str, Field(pattern=PATRON_ID_PROVEEDOR)]


class ErrorValidacion(ValueError):
    """Los argumentos o el registro de proveedores contienen datos inválidos."""

    tipo: Literal["validacion"] = "validacion"


class DenegacionSeguridad(PermissionError):
    """Se intentó una operación sin permiso o sobre archivos no permitidos."""

    tipo: Literal["seguridad"] = "seguridad"


class ReglaNegocioIncumplida(Exception):
    """El usuario tiene permiso, pero el proveedor no cumple los requisitos para aprobarlo."""

    tipo: Literal["negocio"] = "negocio"


class ErrorTecnico(RuntimeError):
    """Falló el acceso a los archivos, independientemente de los permisos del usuario."""

    tipo: Literal["tecnico"] = "tecnico"


class Proveedor(BaseModel):
    """Define los datos comerciales de un proveedor y exige los tipos correctos."""

    # Solo datos comerciales. No se incorporan contactos al esquema ni a la respuesta.
    model_config = ConfigDict(strict=True, extra="forbid")
    id: ProveedorId
    nombre: str = Field(min_length=1, max_length=100)
    estado: EstadoProveedor
    # Los requisitos deben ser booleanos; no se aceptan textos ni números equivalentes.
    seguro_vigente: bool
    continuidad_validada: bool
    homologacion_completa: bool


class SolicitudActualizacion(BaseModel):
    """Comprueba el formato del proveedor y del estado que se quiere guardar."""

    model_config = ConfigDict(strict=True, extra="forbid")
    proveedor_id: ProveedorId
    nuevo_estado: EstadoProveedor


REGISTRO = TypeAdapter(list[Proveedor])


def validar_registro(contenido: str) -> list[Proveedor]:
    """Valida el registro comercial para el servicio y la comparación de estados."""
    try:
        proveedores = REGISTRO.validate_json(contenido)
    except ValidationError as exc:
        raise ErrorValidacion("Registro de proveedores inválido") from exc
    if not proveedores or len({proveedor.id for proveedor in proveedores}) != len(
        proveedores
    ):
        raise ErrorValidacion(
            "El registro no puede estar vacío ni contener identificadores repetidos"
        )
    return proveedores


class ResultadoActualizacion(TypedDict):
    ok: bool
    cambio: bool
    tipo: Literal["exito", "validacion", "seguridad", "negocio", "tecnico"]
    mensaje: str


@dataclass(frozen=True)
class Usuario:
    """Representa al usuario de prueba y su permiso para modificar proveedores."""

    nombre: str
    puede_modificar_proveedores: bool


USUARIOS = {
    "analista": Usuario("Analista de Compras", False),
    "aprobador": Usuario("Responsable de Compras", True),
}


class ServicioProveedores:
    """El usuario autorizado puede cambiar estados; aprobar exige requisitos completos."""

    def __init__(self, directorio: Path, usuario: Usuario, emitir: Callable):
        self.directorio = directorio.resolve(strict=True)
        self.usuario = usuario
        self.emitir = emitir
        self.bloqueo = Lock()

    @contextmanager
    def _operacion(self, herramienta: str, **argumentos):
        """Registra el inicio y el resultado de cada operación, sin incluir documentos."""
        datos_evento = {
            "herramienta": herramienta,
            "usuario": self.usuario.nombre,
            "argumentos": argumentos,
        }
        resultado_operacion: dict[str, bool] = {}
        self.emitir(EventoServicio.INICIADA, **datos_evento)
        try:
            yield resultado_operacion
        except ErrorAccesoDatos as exc:
            # El almacenamiento no depende de las políticas. El servicio traduce
            # sus restricciones a los errores y eventos que expone a sus clientes.
            if exc.tipo == "seguridad":
                self.emitir(EventoServicio.DENEGADA, motivo=str(exc), **datos_evento)
                raise DenegacionSeguridad(str(exc)) from exc
            self.emitir(EventoServicio.DATOS_INVALIDOS, motivo=str(exc), **datos_evento)
            raise ErrorValidacion(str(exc)) from exc
        except ErrorValidacion as exc:
            self.emitir(EventoServicio.DATOS_INVALIDOS, motivo=str(exc), **datos_evento)
            raise
        except DenegacionSeguridad as exc:
            self.emitir(EventoServicio.DENEGADA, motivo=str(exc), **datos_evento)
            raise
        except ReglaNegocioIncumplida as exc:
            self.emitir(
                EventoServicio.REGLA_INCUMPLIDA, motivo=str(exc), **datos_evento
            )
            raise
        except (OSError, UnicodeError) as exc:
            # No poder abrir un archivo es un problema técnico; no significa que
            # el usuario carezca del permiso empresarial para aprobar proveedores.
            self.emitir(
                EventoServicio.ERROR_TECNICO,
                motivo="No se pudieron procesar los archivos del servicio",
                clase_error=type(exc).__name__,
                **datos_evento,
            )
            raise ErrorTecnico("No se pudieron procesar los archivos del servicio") from exc
        else:
            self.emitir(EventoServicio.COMPLETADA, **datos_evento, **resultado_operacion)

    def _cargar_proveedores(self) -> list[Proveedor]:
        return validar_registro(leer_comercial(self.directorio, ARCHIVO_PROVEEDORES))

    def consultar_informe(self) -> str:
        """Entrega el informe comercial sin consultar los datos personales de contacto."""
        with self._operacion(Herramienta.CONSULTAR_INFORME):
            return leer_comercial(self.directorio, ARCHIVO_INFORME)

    def consultar_proveedores(self) -> list[dict]:
        """Devuelve los proveedores después de comprobar que el registro es válido."""
        with self._operacion(Herramienta.CONSULTAR_PROVEEDORES), self.bloqueo:
            return [proveedor.model_dump() for proveedor in self._cargar_proveedores()]

    def actualizar_estado_proveedor(
        self, proveedor_id: str, nuevo_estado: EstadoProveedor
    ) -> ResultadoActualizacion:
        """Comprueba permisos y requisitos antes de guardar el nuevo estado del proveedor."""
        try:
            with (
                self._operacion(
                    Herramienta.ACTUALIZAR_PROVEEDOR,
                    proveedor_id=proveedor_id,
                    nuevo_estado=nuevo_estado,
                ) as resultado_operacion,
                self.bloqueo,
            ):
                try:
                    solicitud = SolicitudActualizacion(
                        proveedor_id=proveedor_id, nuevo_estado=nuevo_estado
                    )
                except ValidationError as exc:
                    raise ErrorValidacion("Identificador o estado inválido") from exc
                # Ni el informe ni una confirmación humana cambian el permiso del
                # usuario que la aplicación asignó al servicio.
                if not self.usuario.puede_modificar_proveedores:
                    raise DenegacionSeguridad(
                        "El usuario no tiene permiso para modificar estados de proveedores"
                    )
                proveedores = self._cargar_proveedores()
                proveedor = next(
                    (item for item in proveedores if item.id == solicitud.proveedor_id),
                    None,
                )
                if proveedor is None:
                    raise ErrorValidacion("El proveedor no existe")
                if proveedor.estado == solicitud.nuevo_estado:
                    resultado: ResultadoActualizacion = {
                        "ok": True,
                        "cambio": False,
                        "tipo": "exito",
                        "mensaje": "El proveedor ya tiene ese estado",
                    }
                    resultado_operacion["cambio"] = resultado["cambio"]
                    return resultado
                if solicitud.nuevo_estado == "APROBADO":
                    # Tener permiso no habilita aprobar un proveedor con requisitos pendientes.
                    faltantes = [
                        descripcion
                        for cumplido, descripcion in (
                            (proveedor.seguro_vigente, "el seguro"),
                            (proveedor.continuidad_validada, "el plan de continuidad"),
                            (proveedor.homologacion_completa, "la homologación"),
                        )
                        if not cumplido
                    ]
                    if faltantes:
                        detalle = (
                            ", ".join(faltantes[:-1]) + " y " + faltantes[-1]
                            if len(faltantes) > 1
                            else faltantes[0]
                        )
                        raise ReglaNegocioIncumplida(
                            f"No se puede aprobar {proveedor.nombre}. Falta validar {detalle}."
                        )
                estado_anterior = proveedor.estado
                proveedor.estado = solicitud.nuevo_estado
                contenido = (
                    json.dumps(
                        [item.model_dump() for item in proveedores],
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n"
                )
                guardar_proveedores(self.directorio, contenido)
                resultado = {
                    "ok": True,
                    "cambio": True,
                    "tipo": "exito",
                    "mensaje": f"Proveedor {solicitud.proveedor_id}: {estado_anterior} → {solicitud.nuevo_estado}",
                }
                # El evento informa el resultado real, después de guardar el registro.
                resultado_operacion["cambio"] = resultado["cambio"]
                return resultado
        except (
            ErrorValidacion,
            DenegacionSeguridad,
            ReglaNegocioIncumplida,
            ErrorTecnico,
        ) as exc:
            return {"ok": False, "cambio": False, "tipo": exc.tipo, "mensaje": str(exc)}
