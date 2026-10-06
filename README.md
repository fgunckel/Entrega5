# Pre-Entrega 5 — Agente ReAct de Fórmula 1 con memoria persistente (LangGraph)

Agente ReAct (Reasoning + Acting) construido con **LangGraph** que responde
consultas sobre un dominio de **Fórmula 1** con datos simulados en memoria y
**memoria conversacional persistente** (Checkpointer SQLite).

> ⚠️ **Los datos son ficticios.** La base simulada no reproduce resultados
> reales de Fórmula 1: es un dataset pequeño y coherente (temporada 2026)
> creado para demostrar el razonamiento multi-paso del agente.

---

## 1. Objetivo del proyecto

Demostrar el patrón **ReAct** (Reasoning + Acting) con LangGraph: un ciclo
abierto donde el propio LLM decide, turno a turno, si necesita usar una
herramienta o si ya puede responder. A diferencia de un flujo fijo, acá el
modelo razona, actúa (tool call), observa el resultado y vuelve a razonar
hasta que tiene suficiente información.

## 2. Arquitectura

```
User
 |
 v
LLM / Model          (nodo "modelo": llm.bind_tools(tools))
 |
 +--- no tool ---> END
 |
 v
ToolNode             (nodo "herramientas": ejecuta las @tool)
 |
 v
Observation
 |
 +-------------> LLM   (ciclo ReAct: herramientas -> modelo)
```

- **Estado**: `MessagesState` (historial de mensajes con el reducer `add_messages`).
- **Nodo modelo**: invoca al LLM con las herramientas vinculadas.
- **Nodo herramientas**: `ToolNode` con `handle_tool_errors=True`.
- **`tools_condition`**: enruta a herramientas si el último mensaje trae `tool_calls`, o a `END` si el modelo ya puede responder.
- **Ciclo**: `herramientas -> modelo` se repite hasta que el modelo tenga suficiente información.
- Los clientes del LLM y la configuración viven **fuera del estado** (`src/graph.py`).

## 3. Modelo de datos F1

Base simulada **en memoria** (`src/data.py`), normalizada:

```
ESCUDERIAS
    |
    +----< PILOTOS
              |
PISTAS        |
   |          |
   +--< GRANDES_PREMIOS
              |
              +--< SESIONES >---- TIPOS_SESION
                      |
                      +--< RESULTADOS >---- PILOTOS
```

| Entidad | Campos | Cantidad |
|---|---|---|
| ESCUDERIAS | id, nombre | 6 |
| PILOTOS | id, nombre, numero, escuderia_id | 12 |
| PISTAS | id, nombre, pais | 4 |
| GRANDES_PREMIOS | id, nombre, temporada, pista_id | 4 |
| TIPOS_SESION | id, nombre (FP1..CARRERA) | 7 |
| SESIONES | id, gran_premio_id, tipo_sesion_id, fecha | 8 (QUALY + CARRERA por GP) |
| RESULTADOS | id, sesion_id, piloto_id, posicion, tiempo, puntos | 96 |

Puntos clave del diseño:

- **TIPOS_SESION es una entidad separada** (sin flags como `is_qualy`): se pueden
  agregar nuevos tipos de sesión (FP1, SPRINT, etc.) sin modificar RESULTADOS.
- **No hay tablas separadas RESULTADOS_QUALY / RESULTADOS_CARRERA**: ambos usan
  la única tabla RESULTADOS. El tipo de resultado se determina mediante
  `RESULTADOS -> SESIONES -> TIPOS_SESION` (normalización).
- SQLite **no** se usa para los datos de F1: solo como Checkpointer de LangGraph.

## 4. Herramientas disponibles

Todas son `@tool`, `async`, con type hints y docstrings descriptivos (el
docstring es lo único que el LLM lee para decidir cuándo usarlas):

| Herramienta | Responsabilidad |
|---|---|
| `buscar_escuderia(nombre)` | Devuelve el `escuderia_id` a partir del nombre. |
| `buscar_pilotos_por_escuderia(escuderia_id)` | Lista los pilotos de una escudería. |
| `buscar_gran_premio(nombre, temporada=None)` | Devuelve el `gran_premio_id`, temporada y pista (busca por nombre, país o circuito). |
| `buscar_sesion(gran_premio_id, tipo_sesion)` | Devuelve el `sesion_id` de un tipo (QUALY, CARRERA, ...). |
| `buscar_resultados_sesion(sesion_id)` | Devuelve posiciones, tiempos y puntos de una sesión. |

Cada herramienta es pequeña y con responsabilidad específica: el agente debe
encadenar varias para responder, lo que fuerza el razonamiento multi-paso.
Los errores se devuelven como mensajes `ERROR: ...` controlados (nunca como
excepciones sin capturar), de modo que la observación vuelva al LLM.

## 5. Flujo ReAct

El grafo (`src/graph.py`) usa `StateGraph(MessagesState)`:

- `START -> modelo`
- `modelo` invoca al LLM con `llm.bind_tools(tools)`.
- `tools_condition` decide: si hay `tool_calls` -> nodo `herramientas`; si no -> `END`.
- `ToolNode` ejecuta las herramientas y devuelve las observaciones.
- `herramientas -> modelo` cierra el ciclo ReAct.
- `recursion_limit = 10` evita ciclos infinitos.

El LLM decide **autónomamente** qué herramientas usar y en qué orden. No hay
ningún `if/else` que inspeccione el texto del prompt para hacer routing manual.

Ejemplo real de la traza (consulta principal):

```
Turno 1: buscar_escuderia("Ferrari")
Turno 2: buscar_pilotos_por_escuderia(1)
Turno 3: buscar_gran_premio("Japón")
Turno 4: buscar_sesion(1, "QUALY")
Turno 5: buscar_sesion(1, "CARRERA")
Turno 6: buscar_resultados_sesion([1, 2])
Turno 7: respuesta final
```

La cantidad y el orden de las llamadas pueden variar según el LLM y su
comportamiento (algunos agrupan llamadas independientes en un mismo turno,
otros las emiten una por una). Lo importante es que el agente obtiene los
`sesion_id` **obligatoriamente** mediante `buscar_sesion`: `buscar_gran_premio`
no devuelve sesiones.

## 6. Persistencia y thread_id

- **Checkpointer**: `AsyncSqliteSaver` (compatible con el entorno async del proyecto).
- Archivo: `checkpoints.sqlite` (solo para checkpoints de LangGraph, no para los datos F1).
- Cada conversación se identifica con `thread_id` (ej. `f1-demo-001`).
- Con el mismo `thread_id`, LangGraph restaura el historial completo: la memoria
  funciona entre turnos **y entre ejecuciones del proceso**.
- `recursion_limit = 10` en la config para evitar ciclos infinitos.

## 7. Instalación

Requiere Python 3.12+.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

## 8. Configuración de variables de entorno

Copiar `.env.example` a `.env` y completar la API key:

```bash
cp .env.example .env
```

| Variable | Descripción |
|---|---|
| `LLM_PROVIDER` | `gemini` (default), `groq` u `openai` |
| `GOOGLE_API_KEY` | Clave de Gemini (gratis en aistudio.google.com/apikey) |
| `GROQ_API_KEY` | Clave de Groq (https://console.groq.com/keys) |
| `GROQ_MODEL` | Opcional (default: `qwen/qwen3.8-27b`) |
| `OPENAI_API_KEY` / `USE_OPENAI` | Alternativa OpenAI |

Nunca se incluye una API key real en código, README, JSON o logs. El `.env`
está en `.gitignore`; solo se versiona `.env.example`.

## 9. Ejecución

```bash
# Escenarios completos (multi-paso + memoria + error) y traza:
python main.py

# Verificar que la memoria persiste tras reiniciar el proceso:
python main.py --recordar

# Consola interactiva: preguntale al agente en tiempo real:
python main.py --chat

# Tests (los de integración se saltan si no hay API key):
pytest tests/ -v
```

## 10. Ejemplo multi-paso

```
🧑 ¿Cómo clasificaron los pilotos de Ferrari en el GP de Japón y cómo terminaron la carrera?

El agente decide autónomamente (sin if/else manual):
  buscar_escuderia("Ferrari")        -> escuderia_id=1
  buscar_gran_premio("Japón")        -> gran_premio_id=1
  buscar_pilotos_por_escuderia(1)    -> Leclerc, Sainz
  buscar_sesion(1, "QUALY")          -> sesion_id=1
  buscar_sesion(1, "CARRERA")        -> sesion_id=2
  buscar_resultados_sesion(1)        -> resultados QUALY
  buscar_resultados_sesion(2)        -> resultados carrera
  -> respuesta final
```

La cantidad de llamadas puede variar según el LLM (algunos agrupan llamadas
independientes en un mismo turno, otros las emiten una por una); en todas las
variantes el ciclo ReAct es el mismo y supera el mínimo de 2 tool calls pedido.

## 11. Ejemplo de memoria

Primera interacción (thread_id `f1-demo-001`):

```
🧑 ¿Cómo clasificaron los pilotos de Ferrari en el GP de Japón y cómo terminaron la carrera?
🤖 ... Leclerc 2.º en QUALY, 4.º en carrera; Sainz 5.º, 5.º ...
```

Segunda interacción con **exactamente el mismo thread_id**:

```
🧑 ¿Y los de Mercedes?
🤖 Los pilotos de Mercedes en el GP de Japón 2026 fueron: Russell 4.º en QUALY
   y 3.º en carrera; Antonelli 7.º y 7.º ...
```

"¿Y los de Mercedes?" solo tiene sentido con memoria: el checkpointer le
devuelve al modelo el historial completo (mismo GP, QUALY y CARRERA) antes de
responder.

Para demostrar que la memoria sobrevive al reinicio del proceso:

```bash
python main.py             # crea la conversación en checkpoints.sqlite
python main.py --recordar  # proceso NUEVO, mismo thread_id: "¿Y los de McLaren?"
```

El agente responde sobre McLaren **en el GP de Japón** sin que se lo repitas:
la memoria vive en SQLite, no en el proceso.

## 12. Ejemplo de recuperación ante error

```
🧑 ¿Cómo le fue a Colapinto en el GP de Brasil?
🤖 No encontré un GP de Brasil en la base de datos. Los únicos Grandes
   Premios disponibles para la temporada 2026 son: Japón, Italia,
   Reino Unido, Mónaco. ¿Querés que te cuente cómo le fue a Colapinto
   en alguno de estos?
```

Flujo interno: `buscar_gran_premio("Brasil")` devuelve `ERROR: ...` → esa
observación vuelve al LLM (no rompe la ejecución, gracias a
`handle_tool_errors=True` y a que las herramientas devuelven errores como
texto) → el modelo razona de nuevo e informa el problema / pide aclaración en
vez de inventar resultados.

## 13. Modo interactivo

```bash
python main.py --chat
```

Consola interactiva para demostrar el agente en tiempo real, con el mismo
flujo de arquitectura que los escenarios (sin cambios):

```
CLI -> StateGraph -> LLM -> Tool Calling -> ToolNode -> Tools
    -> Observation -> LLM -> respuesta
```

- La sesión usa su propio `thread_id` (`f1-chat-001`): la memoria conversacional
  funciona entre preguntas y persiste en `checkpoints.sqlite` entre ejecuciones.
- Cada turno se acumula en `traces/react_trace.json` bajo
  `escenario_chat_interactivo` (traza observable de tool calls y observaciones).

Comandos disponibles:

| Comando | Acción |
|---------|--------|
| `/help` | muestra los comandos disponibles |
| `/new` | genera un nuevo `thread_id` y empieza una conversación nueva (mismo `AsyncSqliteSaver`: la conversación anterior queda guardada) |
| `/thread` | muestra el `thread_id` actual |
| `/trace` | muestra la última traza ReAct observable (`USER` / `TOOL CALL` / `TOOL RESULT` / ... / `FINAL RESPONSE`; sin chain-of-thought) |
| `/exit` | salir (también: `exit`, `quit`, `salir`) |

Ejemplo de sesión:

```
🧑 Vos: ¿Cómo clasificaron los pilotos de Ferrari en el GP de Japón?
🤖 (tabla de clasificación, tras varios tool calls encadenados)
🧑 Vos: ¿Y cómo terminaron la carrera?
🤖 (tabla de la carrera — usó la memoria del turno anterior, mismo thread_id)
🧑 Vos: /trace
   --- Traza ReAct observable (último turno) ---
   USER
   TOOL CALL: buscar_escuderia({'nombre': 'Ferrari'})
   TOOL RESULT: ...
   FINAL RESPONSE: ...
🧑 Vos: /new
   🆕 Nueva conversación. thread_id: f1-chat-001-4f47c09d
🧑 Vos: ¿Y cómo le fue a McLaren?
🤖 (pide aclaración: la conversación nueva no tiene el contexto anterior)
🧑 Vos: /exit
👋 Sesión terminada. La memoria quedó guardada en checkpoints.sqlite.
```

## 14. Ubicación de react_trace.json

`traces/react_trace.json` — generado por `python main.py`. Contiene la traza
observable de cada escenario:

- `escenario_1_multi_paso`: prompt → tool calls (nombre + argumentos) →
  observaciones → ... → respuesta final.
- `escenario_2_memoria`: seguimiento con el mismo thread_id.
- `escenario_3_error_y_aclaracion`: error controlado → aclaración.
- `escenario_recordado_proceso_nuevo`: memoria tras reiniciar el proceso.

La traza registra solo eventos observables (User Message, Tool Call,
Arguments, Observation, Final Answer). **No** se registra ni expone el
chain-of-thought interno del modelo.

## Estructura del proyecto

```
entrega5/
|
+-- src/
|   +-- __init__.py
|   +-- data.py      # base simulada de F1 (en memoria)
|   +-- tools.py     # @tool async con docstrings descriptivos
|   +-- graph.py     # StateGraph + bind_tools + checkpointer
|   +-- trace.py     # serialización de la traza ReAct observable
|
+-- traces/
|   +-- react_trace.json
|
+-- tests/
|   +-- test_agent.py
|
+-- main.py
+-- requirements.txt
+-- pytest.ini
+-- .env.example
+-- .gitignore
+-- README.md
```
