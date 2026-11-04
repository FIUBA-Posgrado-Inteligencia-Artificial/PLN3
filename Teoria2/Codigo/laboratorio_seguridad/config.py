"""Parámetros técnicos del laboratorio.

Los nombres de herramientas, estados y eventos se definen en contracts.py.
Las credenciales y el modelo del agente se configuran en .env.
"""

# Protección opcional del informe comercial.
MODELO = "meta-llama/Llama-Prompt-Guard-2-86M"
UMBRAL = 0.5
MAX_TOKENS = 512

# Tamaño máximo permitido por archivo comercial (bytes).
LIMITE = 16_384

# Límites de ejecución del agente.
TIMEOUT_LLM = 60
REINTENTOS_LLM = 1
LIMITE_RECURSION = 40
