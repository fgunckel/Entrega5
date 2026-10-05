"""Tests del agente ReAct de Fórmula 1.

Los tests de herramientas y de datos corren siempre (no requieren API key).
Los tests de integración del agente (que invocan al LLM real) se marcan con
@pytest.mark.integration y se saltan automáticamente si no hay API key
configurada, para que el repo pueda ejecutarse en CI sin credenciales.

Ejecutar:
    pytest tests/ -v
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from src.data import (
    ESCUDERIAS_DB,
    GRANDES_PREMIOS_DB,
    PILOTOS_DB,
    PISTAS_DB,
    RESULTADOS_DB,
    SESIONES_DB,
    TIPOS_SESION_DB,
)
from src.graph import CHECKPOINTS_DB, RECURSION_LIMIT, compilar_grafo
from src.tools import (
    buscar_escuderia,
    buscar_gran_premio,
    buscar_pilotos_por_escuderia,
    buscar_resultados_sesion,
    buscar_sesion,
)
from src.trace import extraer_texto, serializar_traza

load_dotenv()

# Marca para tests que requieren el LLM real (API key).
requiere_llm = pytest.mark.skipif(
    not (os.environ.get("GROQ_API_KEY") or os.environ.get("GOOGLE_API_KEY") or os.environ.get("OPENAI_API_KEY")),
    reason="requiere una API key configurada en .env",
)


# ---------------------------------------------------------------------------
# Datos simulados
# ---------------------------------------------------------------------------


def test_datos_coherentes() -> None:
    """Verifica la integridad referencial de la base simulada."""
    assert len(ESCUDERIAS_DB) == 6
    assert len(PILOTOS_DB) == 12
    assert len(PISTAS_DB) == 4
    assert len(GRANDES_PREMIOS_DB) == 4

    escuderia_ids = {e["id"] for e in ESCUDERIAS_DB}
    assert all(p["escuderia_id"] in escuderia_ids for p in PILOTOS_DB)

    pista_ids = {p["id"] for p in PISTAS_DB}
    assert all(gp["pista_id"] in pista_ids for gp in GRANDES_PREMIOS_DB)

    tipo_ids = {t["id"] for t in TIPOS_SESION_DB}
    sesion_ids = {s["id"] for s in SESIONES_DB}
    assert all(s["tipo_sesion_id"] in tipo_ids for s in SESIONES_DB)

    piloto_ids = {p["id"] for p in PILOTOS_DB}
    assert all(r["sesion_id"] in sesion_ids for r in RESULTADOS_DB)
    assert all(r["piloto_id"] in piloto_ids for r in RESULTADOS_DB)


def test_resultados_normalizados() -> None:
    """QUALY y CARRERA comparten la única tabla RESULTADOS (sin tablas separadas)."""
    sesiones_por_tipo = {
        s["id"]: s["tipo_sesion_id"] for s in SESIONES_DB
    }
    sesiones_qualy = {s for s, t in sesiones_por_tipo.items() if t == 6}
    sesiones_carrera = {s for s, t in sesiones_por_tipo.items() if t == 7}
    assert sesiones_qualy and sesiones_carrera
    # Todos los resultados viven en la misma tabla:
    ids_sesiones = {r["sesion_id"] for r in RESULTADOS_DB}
    assert sesiones_qualy <= ids_sesiones
    assert sesiones_carrera <= ids_sesiones


# ---------------------------------------------------------------------------
# Herramientas (unitarias, sin LLM)
# ---------------------------------------------------------------------------


async def test_buscar_escuderia_encontrada() -> None:
    resultado = await buscar_escuderia.ainvoke({"nombre": "Ferrari"})
    assert "escuderia_id=1" in resultado


async def test_buscar_escuderia_error_controlado() -> None:
    resultado = await buscar_escuderia.ainvoke({"nombre": "Brabham"})
    assert resultado.startswith("ERROR:")


async def test_buscar_pilotos_por_escuderia() -> None:
    resultado = await buscar_pilotos_por_escuderia.ainvoke({"escuderia_id": 1})
    assert "Leclerc" in resultado and "Sainz" in resultado


async def test_buscar_gran_premio_por_pais() -> None:
    resultado = await buscar_gran_premio.ainvoke({"nombre": "Japón"})
    assert "gran_premio_id=1" in resultado


async def test_buscar_gran_premio_inexistente() -> None:
    resultado = await buscar_gran_premio.ainvoke({"nombre": "Brasil"})
    assert resultado.startswith("ERROR:")


async def test_buscar_sesion_qualy_y_carrera() -> None:
    qualy = await buscar_sesion.ainvoke({"gran_premio_id": 1, "tipo_sesion": "QUALY"})
    carrera = await buscar_sesion.ainvoke({"gran_premio_id": 1, "tipo_sesion": "CARRERA"})
    assert "sesion_id=1" in qualy
    assert "sesion_id=2" in carrera


async def test_buscar_sesion_tipo_invalido() -> None:
    resultado = await buscar_sesion.ainvoke({"gran_premio_id": 1, "tipo_sesion": "QUALI"})
    assert resultado.startswith("ERROR:")


async def test_buscar_resultados_sesion() -> None:
    resultado = await buscar_resultados_sesion.ainvoke({"sesion_id": 1})
    assert "QUALY" in resultado and "Leclerc" in resultado


async def test_buscar_resultados_sesion_inexistente() -> None:
    resultado = await buscar_resultados_sesion.ainvoke({"sesion_id": 999})
    assert resultado.startswith("ERROR:")


# ---------------------------------------------------------------------------
# Traza
# ---------------------------------------------------------------------------


def test_serializar_traza_estructura() -> None:
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    mensajes = [
        HumanMessage(content="pregunta"),
        AIMessage(
            content="",
            tool_calls=[{"name": "buscar_escuderia", "args": {"nombre": "Ferrari"}, "id": "call_1"}],
        ),
        ToolMessage(content="escuderia_id=1", tool_call_id="call_1", name="buscar_escuderia"),
        AIMessage(content="respuesta final"),
    ]
    traza = serializar_traza(mensajes)
    assert traza[0]["tipo"] == "HumanMessage"
    assert traza[1]["tool_calls"][0]["nombre"] == "buscar_escuderia"
    assert traza[1]["tool_calls"][0]["argumentos"] == {"nombre": "Ferrari"}
    assert traza[2]["herramienta"] == "buscar_escuderia"
    assert "razonamiento" not in json.dumps(traza)


def test_extraer_texto_con_bloques() -> None:
    from types import SimpleNamespace

    mensaje = SimpleNamespace(content=[{"text": "hola "}, {"text": "mundo"}])
    assert extraer_texto(mensaje) == "hola mundo"


# ---------------------------------------------------------------------------
# Agente completo (integración, requiere API key)
# ---------------------------------------------------------------------------


async def _crear_app(tmp_path: Path):
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "test_checkpoints.sqlite")) as checkpointer:
        yield compilar_grafo(checkpointer)


@requiere_llm
async def test_agente_multi_paso(tmp_path) -> None:
    """El agente debe hacer al menos 2 tool calls antes de responder."""
    async for app in _crear_app(tmp_path):
        config = {"configurable": {"thread_id": "test-multi"}, "recursion_limit": RECURSION_LIMIT}
        resultado = await app.ainvoke(
            {"messages": [HumanMessage(content="¿Cómo clasificaron los pilotos de Ferrari en el GP de Japón y cómo terminaron la carrera?")]},
            config=config,
        )
        cantidad_tool_calls = sum(
            len(getattr(m, "tool_calls", []) or []) for m in resultado["messages"]
        )
        assert cantidad_tool_calls >= 2
        assert extraer_texto(resultado["messages"][-1])  # hay respuesta final


@requiere_llm
async def test_memoria_entre_turnos(tmp_path) -> None:
    """Con el mismo thread_id, '¿Y los de Mercedes?' debe responder sobre Mercedes."""
    async for app in _crear_app(tmp_path):
        config = {"configurable": {"thread_id": "test-memoria"}, "recursion_limit": RECURSION_LIMIT}
        await app.ainvoke(
            {"messages": [HumanMessage(content="¿Cómo clasificaron los pilotos de Ferrari en el GP de Japón y cómo terminaron la carrera?")]},
            config=config,
        )
        resultado_2 = await app.ainvoke(
            {"messages": [HumanMessage(content="¿Y los de Mercedes?")]},
            config=config,
        )
        respuesta = extraer_texto(resultado_2["messages"][-1]).lower()
        assert "mercedes" in respuesta or "russell" in respuesta or "antonelli" in respuesta


@requiere_llm
async def test_ciclo_retorno_ante_error(tmp_path) -> None:
    """Ante un GP inexistente, el agente no debe romper ni inventar: debe
    informar el problema o pedir aclaración."""
    async for app in _crear_app(tmp_path):
        config = {"configurable": {"thread_id": "test-error"}, "recursion_limit": RECURSION_LIMIT}
        resultado = await app.ainvoke(
            {"messages": [HumanMessage(content="¿Cómo le fue a Colapinto en el GP de Brasil?")]},
            config=config,
        )
        respuesta = extraer_texto(resultado["messages"][-1])
        assert respuesta  # respondió algo controlado
        assert "brasil" not in respuesta.lower() or "no" in respuesta.lower()
