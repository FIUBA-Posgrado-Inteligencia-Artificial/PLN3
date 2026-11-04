
"""Tres herramientas de negocio tipadas; ninguna expone rutas o permisos.

https://docs.langchain.com/oss/python/langchain/mcp/index
"""

from logging import DEBUG

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from contracts import EstadoProveedor, Herramienta
from policy import (
    DenegacionSeguridad,
    ErrorTecnico,
    ErrorValidacion,
    ResultadoActualizacion,
    ServicioProveedores,
)


def crear_servidor(servicio: ServicioProveedores) -> FastMCP:
    """Crea las herramientas MCP utilizando el servicio de proveedores configurado."""
    server = FastMCP("Compras: evaluación de proveedores")

    # FastMCP comprueba los argumentos. El servicio decide además si el usuario
    # puede hacer el cambio y si el proveedor cumple los requisitos.
    @server.tool(name=Herramienta.CONSULTAR_INFORME)
    def consultar_informe() -> str:
        """Consulta el informe comercial sin datos de contacto."""
        try:
            return servicio.consultar_informe()
        except (ErrorValidacion, DenegacionSeguridad, ErrorTecnico) as exc:
            # El evento del servicio ya explica el rechazo en terminal.
            # ToolError conserva el error MCP sin repetir un traceback de FastMCP.
            raise ToolError(str(exc), log_level=DEBUG) from exc

    @server.tool(name=Herramienta.CONSULTAR_PROVEEDORES)
    def consultar_proveedores() -> list[dict]:
        """Consulta los proveedores, sus estados y requisitos."""
        try:
            return servicio.consultar_proveedores()
        except (ErrorValidacion, DenegacionSeguridad, ErrorTecnico) as exc:
            raise ToolError(str(exc), log_level=DEBUG) from exc

    @server.tool(name=Herramienta.ACTUALIZAR_PROVEEDOR)
    def actualizar_estado_proveedor(
        proveedor_id: str, nuevo_estado: EstadoProveedor
    ) -> ResultadoActualizacion:
        """Cambia el estado del proveedor identificado por proveedor_id.

        nuevo_estado: PENDIENTE, APROBADO o RECHAZADO. Requiere permiso del usuario;
        aprobar exige seguro, continuidad y homologación validados.
        """
        return servicio.actualizar_estado_proveedor(proveedor_id, nuevo_estado)

    return server
