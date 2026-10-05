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
"""

from __future__ import annotations

import asyncio
import json

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
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
    else:
        asyncio.run(main())
