"""Serialización de la traza ReAct observable del agente.

La "traza de razonamiento" requerida por la entrega se interpreta como traza
observable del agente (User Message -> Tool Call -> Arguments -> Observation ->
... -> Final Answer). NO se registra ni expone el chain-of-thought interno del
modelo.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, ToolMessage


def extraer_texto(mensaje: Any) -> str:
    """Normaliza el content de un mensaje (Gemini a veces devuelve una lista
    de bloques en lugar de un string)."""
    contenido = mensaje.content
    if isinstance(contenido, str):
        return contenido
    if isinstance(contenido, list):
        return "".join(
            bloque.get("text", "")
            for bloque in contenido
            if isinstance(bloque, dict)
        )
    return str(contenido)


def serializar_traza(mensajes: list) -> list[dict]:
    """Convierte los mensajes de LangGraph a un formato JSON simple, para dejar
    la traza ReAct como log observable dentro del repo (entregable pedido).

    Registra: tipo de mensaje, contenido, tool_calls (nombre + argumentos) y
    la herramienta que produjo cada ToolMessage (observación).
    NO registra razonamiento privado interno del LLM.
    """
    traza: list[dict] = []
    for m in mensajes:
        entrada: dict = {"tipo": type(m).__name__, "contenido": extraer_texto(m)}
        if isinstance(m, AIMessage) and m.tool_calls:
            entrada["tool_calls"] = [
                {"nombre": tc["name"], "argumentos": tc["args"]} for tc in m.tool_calls
            ]
        if isinstance(m, ToolMessage):
            entrada["herramienta"] = m.name
        traza.append(entrada)
    return traza
