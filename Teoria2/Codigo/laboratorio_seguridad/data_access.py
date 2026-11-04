"""Controla el acceso a los archivos comerciales del laboratorio.

Solo permite utilizar los documentos necesarios para la evaluación.
Cada ejecución trabaja sobre una copia temporal para conservar
intactos los archivos originales.
"""

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator, Literal

from config import LIMITE
from contracts import ARCHIVO_INFORME, ARCHIVO_PROVEEDORES


# Solo estos archivos pueden utilizarse en la evaluación.
# contactos.json queda fuera del alcance de las herramientas del agente.
ARCHIVOS_COMERCIALES = (ARCHIVO_INFORME, ARCHIVO_PROVEEDORES)

# Los informes de los tres escenarios se guardan juntos en data/.
INFORMES_ESCENARIO = {
    "limpio": "informe_limpio.txt",
    "bruma": "informe_bruma.txt",
    "cartonera": "informe_cartonera.txt",
}


class ErrorAccesoDatos(Exception):
    """Indica que un archivo no cumple las condiciones para acceder a él."""

    def __init__(self, mensaje: str, tipo: Literal["seguridad", "validacion"]):
        super().__init__(mensaje)
        self.tipo = tipo


@contextmanager
def copia_comercial(origen: Path, escenario: str = "bruma") -> Iterator[Path]:
    """Crea una copia temporal de los archivos comerciales permitidos.

    El agente trabaja sobre esta copia, por lo que los cambios realizados
    durante la práctica no afectan los documentos originales.
    """
    if origen.is_symlink():
        raise ErrorAccesoDatos(
            "El directorio de origen no puede ser un enlace", "seguridad"
        )

    # El nombre del informe se elige de una lista cerrada; las herramientas
    # siguen viendo solamente informe.txt y proveedores.json.
    if escenario not in INFORMES_ESCENARIO:
        raise ErrorAccesoDatos("Escenario no permitido", "validacion")
    nombre_informe = INFORMES_ESCENARIO[escenario]

    # Se comprueban los archivos antes de copiarlos para evitar
    # incorporar documentos no permitidos o demasiado grandes.
    contenidos = [
        (ARCHIVO_INFORME, _leer_contenido(origen, nombre_informe, (nombre_informe,))),
        (ARCHIVO_PROVEEDORES, _leer_contenido(origen, ARCHIVO_PROVEEDORES)),
    ]

    # La carpeta temporal se elimina automáticamente al finalizar.
    with TemporaryDirectory(prefix="compras_seguridad_") as carpeta:
        directorio = Path(carpeta)

        for nombre, contenido in contenidos:
            (directorio / nombre).write_bytes(contenido)

        yield directorio


def _ruta(directorio: Path, nombre: str) -> Path:
    """Comprueba que el archivo no redirija el acceso fuera del directorio."""

    directorio = directorio.resolve(strict=True)
    ruta = directorio / nombre

    # Se rechazan enlaces y rutas que puedan conducir a otros archivos.
    # Así evitamos que una ruta aparentemente válida apunte fuera
    # del directorio de trabajo.
    if (
        ruta.is_symlink()
        or ruta.resolve().parent != directorio
        or (ruta.exists() and ruta.stat().st_nlink > 1)
    ):
        raise ErrorAccesoDatos(
            "Enlace o ruta fuera del entorno controlado", "seguridad"
        )

    return ruta


def _leer_contenido(directorio: Path, nombre: str, permitidos=ARCHIVOS_COMERCIALES) -> bytes:
    """Lee un archivo permitido y comprueba su tamaño."""

    # La lista de archivos permitidos impide solicitar otros documentos,
    # aunque existan dentro del directorio.
    if nombre not in permitidos:
        raise ErrorAccesoDatos(
            "El servicio solo permite consultar información comercial",
            "seguridad",
        )

    # Se lee un byte adicional para detectar si se supera el límite.
    # Así no es necesario cargar archivos grandes completos en memoria.
    with _ruta(directorio, nombre).open("rb") as archivo:
        contenido = archivo.read(LIMITE + 1)

    if len(contenido) > LIMITE:
        raise ErrorAccesoDatos(
            "El archivo supera el límite de lectura", "validacion"
        )

    return contenido


def leer_comercial(directorio: Path, nombre: str) -> str:
    """Devuelve el contenido de un archivo comercial permitido como texto."""

    return _leer_contenido(directorio, nombre).decode("utf-8")


def guardar_proveedores(directorio: Path, contenido: str) -> None:
    """Guarda los cambios del registro de proveedores en la copia de trabajo."""

    if len(contenido.encode("utf-8")) > LIMITE:
        raise ErrorAccesoDatos(
            "El registro supera el límite de escritura", "validacion"
        )

    destino = _ruta(directorio, ARCHIVO_PROVEEDORES)
    temporal = _ruta(directorio, "proveedores.tmp")

    # Primero se escribe el registro completo en otro archivo.
    # Después se reemplaza el anterior, evitando dejarlo parcialmente escrito.
    temporal.write_text(contenido, encoding="utf-8")
    temporal.replace(destino)