"""Grafo ReAct del agente de Fórmula 1 (LangGraph).

Arquitectura conceptual:

    START
       |
       v
    MODELO
       |
       +---- tools_condition ----> END
       |
       v
    TOOLS
       |
       +-------------------------> MODELO

El LLM decide autónomamente si necesita una herramienta (tool_calls) o si ya
puede responder. El ciclo herramientas -> modelo se repite hasta que el modelo
tenga suficiente información.

Los clientes del LLM y la configuración viven FUERA del estado: el estado es
solo MessagesState (historial de mensajes con el reducer add_messages).
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.tools import herramientas

# Carga variables de entorno desde .env (API keys, etc.)
load_dotenv()

# recursion_limit pedido por la consigna para evitar ciclos infinitos.
RECURSION_LIMIT = 10

# Archivo SQLite usado SOLO para el Checkpointer de LangGraph.
CHECKPOINTS_DB = "checkpoints.sqlite"


def crear_llm() -> Any:
    """Crea el LLM desacoplado del grafo, leyendo la API key del entorno.

    Proveedor por defecto: Gemini (GOOGLE_API_KEY). Si se define LLM_PROVIDER
    se puede usar Groq (GROQ_API_KEY) u OpenAI (OPENAI_API_KEY). La API key
    nunca se escribe en el código.
    """
    proveedor = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if proveedor == "groq":
        return ChatGroq(
            model=os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b"),
            temperature=0,
            max_tokens=900,  # el tier gratuito de Groq limita tokens/minuto
        )
    if proveedor == "openai" or os.environ.get("USE_OPENAI", "").strip() in {
        "1", "true", "TRUE", "True"
    }:
        return ChatOpenAI(model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    return ChatGoogleGenerativeAI(
        model=os.environ.get("GOOGLE_MODEL", "gemini-flash-latest"), temperature=0
    )


# El LLM vinculado a las herramientas vive FUERA del estado del grafo.
llm_con_herramientas = None  # se inicializa en compilar_grafo()

# Prompt de sistema: orienta al LLM sobre el dominio y cuándo usar herramientas.
# No hace routing manual: el LLM sigue decidiendo autónomamente qué tool llamar.
SYSTEM_PROMPT = """Sos un asistente de Fórmula 1 con acceso a herramientas que consultan
una base de datos simulada de la temporada 2026.

Reglas:
- Respondé SIEMPRE en castellano.
- Para responder cualquier consulta sobre resultados, clasificación, pilotos o
  escuderías, usá las herramientas disponibles; nunca inventes datos.
- La base contiene una única temporada (2026). Si el usuario no indica temporada,
  no se la preguntes: usá directamente las herramientas.
- Si una herramienta devuelve un ERROR, no inventes la respuesta: leé el error,
  intentá otra estrategia (por ejemplo, corregir el nombre o buscar por otro dato)
  y, si no es posible resolverla, pedile al usuario la aclaración necesaria.
- Para preguntas sobre "cómo clasificaron" usá la sesión QUALY; para "cómo
  terminaron" usá la sesión CARRERA.
- Si varias herramientas son independientes entre sí (por ejemplo, buscar la
  escudería y buscar el gran premio), llamálas todas en el mismo turno para
  ahorrar pasos. Ejemplo: en un mismo turno podés llamar a la vez
  buscar_escuderia("Ferrari") y buscar_gran_premio("Japón")."""


async def nodo_modelo(state: MessagesState) -> dict:
    """Nodo MODELO: le pasa el historial completo al LLM y devuelve su respuesta
    (que puede incluir tool_calls o ser la respuesta final)."""
    mensajes = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in mensajes):
        mensajes = [SystemMessage(content=SYSTEM_PROMPT), *mensajes]
    respuesta = await llm_con_herramientas.ainvoke(mensajes)
    return {"messages": [respuesta]}


def compilar_grafo(checkpointer: Any) -> Any:
    """Compila el StateGraph con el checkpointer dado.

    - Estado: MessagesState (hereda el reducer add_messages).
    - Nodo 'modelo': invoca al LLM con bind_tools.
    - Nodo 'herramientas': ToolNode con manejo controlado de errores.
    - tools_condition decide si ir a herramientas o terminar.
    - El ciclo herramientas -> modelo permite múltiples ciclos ReAct.
    """
    global llm_con_herramientas

    llm = crear_llm()
    llm_con_herramientas = llm.bind_tools(herramientas)

    grafo = StateGraph(MessagesState)
    grafo.add_node("modelo", nodo_modelo)
    grafo.add_node(
        "herramientas", ToolNode(herramientas, handle_tool_errors=True)
    )

    grafo.add_edge(START, "modelo")
    grafo.add_conditional_edges(
        "modelo",
        tools_condition,  # mira si el último mensaje trae tool_calls
        {"tools": "herramientas", END: END},
    )
    grafo.add_edge("herramientas", "modelo")  # el ciclo ReAct

    return grafo.compile(checkpointer=checkpointer)
