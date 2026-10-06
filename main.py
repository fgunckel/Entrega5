"""Agente ReAct de Fórmula 1 con memoria persistente (LangGraph).

Ejecuta tres escenarios de demostración:

1. Consulta multi-paso (>= 2 tool calls autónomos del LLM).
2. Segunda consulta con el MISMO thread_id (memoria conversacional).
3. Escenario de error / información inexistente -> ciclo de retorno
   (Observación problemática -> el LLM intenta otra estrategia o pide
   aclaración).

Al final guarda la traza observable en traces/react_trace.json.

Uso:
    python main.py              # escenarios completos (crea la conversación)
    python main.py --recordar   # reabre el thread_id existente en un proceso
                                # NUEVO y hace una pregunta de seguimiento,
                                # demostrando que la memoria persiste en
                                # checkpoints.sqlite aunque el proceso se
                                # haya reiniciado.
    python main.py --chat       # consola interactiva: hacé preguntas al agente
                                # en tiempo real (mismo thread_id durante toda
                                # la sesión, memoria persistente, traza ReAct
                                # acumulada en traces/react_trace.json).
"""

from __future__ import annotations

import asyncio
import json
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from src.graph import CHECKPOINTS_DB, RECURSION_LIMIT, compilar_grafo
from src.trace import extraer_texto, serializar_traza

# Carga variables de entorno desde .env (API keys, etc.)
load_dotenv()

REACT_TRACE = "traces/react_trace.json"

# thread_id pedido por la consigna: la memoria conversacional se conserva
# entre turnos (y entre ejecuciones del proceso) mientras se reutilice.
THREAD_ID = "f1-demo-001"
THREAD_ID_ERROR = "f1-demo-error-001"

CONFIG = {
    "configurable": {"thread_id": THREAD_ID},
    "recursion_limit": RECURSION_LIMIT,
}

CONFIG_ERROR = {
    "configurable": {"thread_id": THREAD_ID_ERROR},
    "recursion_limit": RECURSION_LIMIT,
}

# thread_id propio del modo interactivo: cada sesión --chat tiene su propia
# conversación, separada de los escenarios de demostración.
THREAD_ID_CHAT = "f1-chat-001"

CONFIG_CHAT = {
    "configurable": {"thread_id": THREAD_ID_CHAT},
    "recursion_limit": RECURSION_LIMIT,
}


def _imprimir_turno(numero: str, prompt: str, respuesta: str) -> None:
    linea = "=" * 80
    print(f"\n{linea}")
    print(f"🧑 Turno {numero}: {prompt}")
    print(linea)
    print(f"🤖 {respuesta}")


async def ejecutar_consulta(app, config: dict, prompt: str) -> dict:
    """Invoca al grafo con un prompt de usuario y devuelve el resultado."""
    return await app.ainvoke(
        {"messages": [HumanMessage(content=prompt)]},
        config=config,
    )


async def escenario_multi_paso(app) -> dict:
    """Escenario 1: consulta multi-paso sobre Ferrari en el GP de Japón.

    El agente necesita varias herramientas encadenadas (escudería -> pilotos ->
    gran premio -> sesiones -> resultados) y decide el orden autónomamente.
    """
    prompt = (
        "¿Cómo clasificaron los pilotos de Ferrari en el GP de Japón "
        "y cómo terminaron la carrera?"
    )
    resultado = await ejecutar_consulta(app, CONFIG, prompt)
    _imprimir_turno("1", prompt, extraer_texto(resultado["messages"][-1]))
    return resultado


async def escenario_memoria(app) -> dict:
    """Escenario 2: seguimiento con el mismo thread_id.

    '¿Y los de Mercedes?' solo tiene sentido si el modelo recuerda que la
    consulta anterior era sobre el GP de Japón, QUALY y CARRERA. El checkpointer
    le devuelve el historial completo antes de responder.
    """
    prompt = "¿Y los de Mercedes?"
    resultado = await ejecutar_consulta(app, CONFIG, prompt)
    _imprimir_turno("2 (mismo thread_id)", prompt, extraer_texto(resultado["messages"][-1]))
    return resultado


async def probar_error(app) -> dict:
    """Escenario 3: ciclo de retorno ante error / información inexistente.

    '¿Cómo le fue a Colapinto en el GP de Brasil?' — no existe ningún GP de
    Brasil en la base, así que buscar_gran_premio devuelve un ERROR. Esa
    observación vuelve al LLM (no rompe la ejecución), que debe intentar otra
    estrategia o pedirle aclaración al usuario en vez de inventar resultados.
    """
    prompt = "¿Cómo le fue a Colapinto en el GP de Brasil?"
    resultado = await ejecutar_consulta(app, CONFIG_ERROR, prompt)
    _imprimir_turno("3 (error)", prompt, extraer_texto(resultado["messages"][-1]))
    return resultado


async def recordar(app) -> dict:
    """Modo --recordar: demuestra que la memoria sobrevive al reinicio.

    Se ejecuta en un proceso NUEVO, reabre checkpoints.sqlite con el mismo
    thread_id y hace una pregunta que solo tiene sentido con el contexto de
    los turnos anteriores ('¿Y los de McLaren?').
    """
    prompt = "¿Y los de McLaren?"
    resultado = await ejecutar_consulta(app, CONFIG, prompt)
    _imprimir_turno("recordado (proceso nuevo, mismo thread_id)", prompt, extraer_texto(resultado["messages"][-1]))
    return resultado


def _ayuda_chat() -> None:
    """Imprime los comandos disponibles en el modo interactivo."""
    print(
        "\nComandos disponibles:\n"
        "  /help    -> muestra los comandos disponibles\n"
        "  /new     -> nueva conversación (nuevo thread_id, misma memoria SQLite)\n"
        "  /thread  -> muestra el thread_id actual\n"
        "  /trace   -> muestra la última traza ReAct observable\n"
        "  /exit    -> salir (también: exit, quit, salir)\n"
    )


def _formatear_traza_observable(mensajes: list) -> str:
    """Formatea la traza observable de un turno: USER / TOOL CALL / TOOL RESULT
    / ... / FINAL RESPONSE. No incluye razonamiento privado del LLM."""
    lineas: list[str] = []
    for m in mensajes:
        tool_calls = getattr(m, "tool_calls", None)
        if isinstance(m, HumanMessage):
            lineas.append("USER")
        elif tool_calls:
            for tc in tool_calls:
                lineas.append(f"TOOL CALL: {tc['name']}({tc['args']})")
        elif isinstance(m, ToolMessage):
            lineas.append(f"TOOL RESULT: {extraer_texto(m)}")
        elif isinstance(m, AIMessage) and extraer_texto(m):
            lineas.append(f"FINAL RESPONSE: {extraer_texto(m)}")
    return "\n".join(lineas)


async def chat_interactivo() -> None:
    """Modo --chat: consola interactiva para demostrar el agente en tiempo real.

    Flujo (idéntico al de los escenarios, sin cambios de arquitectura):

        CLI -> StateGraph -> LLM -> Tool Calling -> ToolNode -> Tools
            -> Observation -> LLM -> respuesta

    - Usa un thread_id propio de la sesión, así que la memoria conversacional
      funciona entre preguntas (y persiste en checkpoints.sqlite entre
      ejecuciones si se reutiliza el mismo thread_id).
    - /new genera un nuevo thread_id y arranca una conversación limpia con el
      MISMO AsyncSqliteSaver (la conversación anterior queda en checkpoints.sqlite).
    - Cada turno se agrega a traces/react_trace.json (traza observable).
    - Comandos: /help /new /thread /trace /exit (también exit, quit, salir).
    """
    print("=" * 80)
    print("🤖 Agente ReAct de Fórmula 1 — modo interactivo")
    print(f"   thread_id: {THREAD_ID_CHAT} (memoria persistente en {CHECKPOINTS_DB})")
    print("   Escribí tu pregunta sobre la temporada 2026, /help para ver comandos.")
    print("=" * 80)

    thread_id = THREAD_ID_CHAT
    historial: list = []
    async with AsyncSqliteSaver.from_conn_string(CHECKPOINTS_DB) as checkpointer:
        app = compilar_grafo(checkpointer)
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
        while True:
            try:
                prompt = input("\n🧑 Vos: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Sesión terminada.")
                break
            if not prompt:
                continue

            comando = prompt.lower()
            if comando in {"/exit", "salir", "exit", "quit"}:
                print("👋 Sesión terminada. La memoria quedó guardada en checkpoints.sqlite.")
                break
            if comando == "/help":
                _ayuda_chat()
                continue
            if comando == "/thread":
                print(f"🔗 thread_id actual: {thread_id}")
                continue
            if comando == "/new":
                thread_id = f"{THREAD_ID_CHAT}-{uuid4().hex[:8]}"
                config = {
                    "configurable": {"thread_id": thread_id},
                    "recursion_limit": RECURSION_LIMIT,
                }
                historial = []
                print(f"🆕 Nueva conversación. thread_id: {thread_id}")
                continue
            if comando == "/trace":
                if not historial:
                    print("(todavía no hay turnos en esta conversación)")
                else:
                    # Muestra solo el último turno: desde el último USER hasta
                    # la respuesta final (eventos observables, sin CoT).
                    print("\n--- Traza ReAct observable (último turno) ---")
                    print(_formatear_traza_observable(historial[-1]))
                    print("--- fin de traza ---")
                continue

            print("🤖 ...")
            resultado = await ejecutar_consulta(app, config, prompt)
            respuesta = extraer_texto(resultado["messages"][-1])
            print(f"🤖 {respuesta}")
            # Solo los mensajes NUEVOS de este turno: el checkpointer devuelve
            # el historial completo del thread, pero /trace debe mostrar el
            # último turno (USER -> TOOL CALL/RESULT -> ... -> FINAL RESPONSE).
            mensajes = resultado["messages"]
            indice_ultimo_user = max(
                i for i, m in enumerate(mensajes) if isinstance(m, HumanMessage)
            )
            historial.append(mensajes[indice_ultimo_user:])

            # Traza observable acumulada por turno (no bloquea la conversación).
            traza_chat = {
                "escenario_chat_interactivo": serializar_traza(mensajes[indice_ultimo_user:])
            }
            try:
                with open(REACT_TRACE, encoding="utf-8") as f:
                    traza_previa = json.load(f)
                traza_chat = {**traza_previa, **traza_chat}
            except FileNotFoundError:
                pass
            with open(REACT_TRACE, "w", encoding="utf-8") as f:
                json.dump(traza_chat, f, ensure_ascii=False, indent=2)


async def main() -> None:
    async with AsyncSqliteSaver.from_conn_string(CHECKPOINTS_DB) as checkpointer:
        app = compilar_grafo(checkpointer)

        resultado_1 = await escenario_multi_paso(app)
        resultado_2 = await escenario_memoria(app)
        resultado_error = await probar_error(app)

    traza_completa = {
        "escenario_1_multi_paso": serializar_traza(resultado_1["messages"]),
        "escenario_2_memoria": serializar_traza(resultado_2["messages"]),
        "escenario_3_error_y_aclaracion": serializar_traza(resultado_error["messages"]),
    }

    with open(REACT_TRACE, "w", encoding="utf-8") as f:
        json.dump(traza_completa, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print(f"✅ Traza ReAct guardada en {REACT_TRACE}")
    print("💡 Ejecutá 'python main.py --recordar' en un proceso nuevo para")
    print("   verificar que la memoria persiste en checkpoints.sqlite.")


async def main_recordar() -> None:
    async with AsyncSqliteSaver.from_conn_string(CHECKPOINTS_DB) as checkpointer:
        app = compilar_grafo(checkpointer)
        resultado = await recordar(app)

    traza = {"escenario_recordado_proceso_nuevo": serializar_traza(resultado["messages"])}
    # Mezcla con la traza existente (si hay) para no perder los escenarios previos.
    try:
        with open(REACT_TRACE, encoding="utf-8") as f:
            traza_previa = json.load(f)
        traza = {**traza_previa, **traza}
    except FileNotFoundError:
        pass
    with open(REACT_TRACE, "w", encoding="utf-8") as f:
        json.dump(traza, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Traza ReAct guardada en {REACT_TRACE}")


if __name__ == "__main__":
    import sys

    if "--recordar" in sys.argv:
        asyncio.run(main_recordar())
    elif "--chat" in sys.argv:
        asyncio.run(chat_interactivo())
    else:
        asyncio.run(main())
