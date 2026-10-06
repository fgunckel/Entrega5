# Documentación paso a paso — Pre-Entrega 5

**Agente ReAct de Fórmula 1 con memoria persistente (LangGraph)**

Este documento explica, paso a paso y en orden de construcción, cómo funciona
cada componente del proyecto, qué problema resuelve y cómo se conecta con el
resto de la arquitectura.

---

## Índice

1. [Objetivo y requisitos de la consigna](#1-objetivo-y-requisitos-de-la-consigna)
2. [Arquitectura general](#2-arquitectura-general)
3. [Paso 1 — Configuración del entorno](#paso-1--configuración-del-entorno)
4. [Paso 2 — Base de datos simulada (`src/data.py`)](#paso-2--base-de-datos-simulada-srcdatapy)
5. [Paso 3 — Herramientas con Tool Calling (`src/tools.py`)](#paso-3--herramientas-con-tool-calling-srctoolspy)
6. [Paso 4 — Grafo ReAct con StateGraph (`src/graph.py`)](#paso-4--grafo-react-stategraph-srcgraphpy)
7. [Paso 5 — Traza observable (`src/trace.py`)](#paso-5--traza-observable-srctracepy)
8. [Paso 6 — Punto de entrada y escenarios (`main.py`)](#paso-6--punto-de-entrada-y-escenarios-mainpy)
9. [Paso 7 — Memoria persistente (SqliteSaver + thread_id)](#paso-7--memoria-persistente-sqlitesaver--thread_id)
10. [Paso 8 — Modo interactivo (`--chat`)](#paso-8--modo-interactivo---chat)
11. [Paso 9 — Tests (`tests/test_agent.py`)](#paso-9--tests-teststest_agentpy)
12. [Ejecución y verificación](#12-ejecución-y-verificación)
13. [Ejemplo real de ciclo ReAct](#13-ejemplo-real-de-ciclo-react)

---

## 1. Objetivo y requisitos de la consigna

La Pre-Entrega 5 pide construir un **agente ReAct** sobre un dominio propio
(Fórmula 1) que demuestre:

| Requisito | Dónde se cumple |
|---|---|
| Autonomía del agente | El LLM decide qué herramienta llamar en cada paso (`src/graph.py`) |
| Tool Calling | 6 herramientas `@tool` en `src/tools.py` |
| Ciclo ReAct | Ciclo `modelo → herramientas → modelo` en el StateGraph |
| StateGraph con ciclos | `StateGraph(MessagesState)` + `add_conditional_edges` |
| Recuperación ante errores | Herramientas devuelven `ERROR: ...` como texto; el LLM razona sobre el error |
| Persistencia | `AsyncSqliteSaver` en `checkpoints.sqlite` |
| Memoria por `thread_id` | Cada conversación tiene su thread; sobrevive al reinicio del proceso |
| Ejecución asíncrona | Todo el código es `async` (Python 3.12+) |
| Traza observable | `src/trace.py` → `traces/react_trace.json` |

Decisiones de diseño clave:

- **SQLite se usa SOLO como checkpointer** de LangGraph, no como base de datos
  de F1. Los datos de F1 viven en memoria (`src/data.py`), lo que mantiene el
  foco del proyecto en el agente y no en la persistencia de datos.
- **El LLM vive fuera del estado del grafo**: el estado es únicamente
  `MessagesState` (historial de mensajes con el reducer `add_messages`).
- **Los errores de herramientas nunca son excepciones sin controlar**: vuelven
  al LLM como observación para que pueda reintentar o pedir aclaración.

---

## 2. Arquitectura general

```
                        ┌─────────────────────────────────────┐
                        │            Usuario (CLI)            │
                        └──────────────┬──────────────────────┘
                                         │ HumanMessage
                        ┌────────────────▼──────────────────────┐
                        │        StateGraph (LangGraph)         │
                        │                                       │
   START ──► modelo ──► tools_condition ──► END (respuesta final)
                ▲           │ tool_calls
                │           ▼
                │        herramientas (ToolNode)
                │           │
                └───────────┘  (ciclo ReAct)
                        │
        ┌───────────────┼──────────────────────┐
        ▼               ▼                      ▼
   ChatGroq /      AsyncSqliteSaver         6 tools
   Gemini /        (checkpoints.sqlite,     (src/tools.py)
   OpenAI          memoria por thread_id)   → src/data.py
```

Flujo de un turno:

1. El usuario envía un `HumanMessage`.
2. El nodo **modelo** invoca al LLM con el historial completo + system prompt.
3. `tools_condition` mira el último mensaje:
   - si trae `tool_calls` → va al nodo `herramientas`;
   - si es texto → termina y devuelve la respuesta final.
4. El `ToolNode` ejecuta las herramientas y devuelve las observaciones.
5. Las observaciones vuelven al modelo (el ciclo se repite hasta resolver).

---

## Paso 1 — Configuración del entorno

**Archivos:** `.env`, `.env.example`, `requirements.txt`

Antes de escribir código se define la configuración externa:

- **`.env.example`** — plantilla versionada con placeholders:
  - `LLM_PROVIDER`: `gemini` (default), `groq` u `openai`.
  - `GROQ_API_KEY` / `GOOGLE_API_KEY` / `OPENAI_API_KEY`: la clave **nunca se
    escribe en el código**; se lee del entorno con `os.environ`.
  - `GROQ_MODEL` (opcional): en este proyecto se usa `openai/gpt-oss-120b`.
- **`.env`** — copia con las claves reales; está en `.gitignore`.
- **`requirements.txt`** — dependencias: `langgraph`, `langchain-core`,
  `langchain-groq`, `langchain-google-genai`, `langchain-openai`, `python-dotenv`,
  `aiosqlite`, `pytest`.

Instalación:

```bash
cd entrega5
pip install -r requirements.txt
cp .env.example .env   # y completar la API key
```

---

## Paso 2 — Base de datos simulada (`src/data.py`)

Es el **dominio** sobre el que el agente razona. Son listas de diccionarios en
memoria, con integridad referencial entre sí:

```
ESCUDERIAS ──< PILOTOS
PISTAS ──< GRANDES_PREMIOS ──< SESIONES >---- TIPOS_SESION
                                   │
                                   +--< RESULTADOS >---- PILOTOS
```

- `ESCUDERIAS_DB` (6 escuderías) → cada piloto referencia su `escuderia_id`.
- `PILOTOS_DB` (12 pilotos, 2 por escudería, incluye a Franco Colapinto).
- `PISTAS_DB` + `GRANDES_PREMIOS_DB` (4 GP de la temporada 2026).
- `TIPOS_SESION_DB` — entidad separada (QUALY, CARRERA) en lugar de flags como
  `is_qualy`: permite agregar nuevos tipos de sesión sin tocar RESULTADOS.
- `SESIONES_DB` → cada GP tiene sus sesiones con `sesion_id`.
- `RESULTADOS_DB` → posición, tiempo y puntos por piloto y sesión.

Este diseño normalizado obliga al agente a **encadenar herramientas** para
responder (no hay una sola tabla que resuelva todo), que es justo lo que la
consigna quiere demostrar.

---

## Paso 3 — Herramientas con Tool Calling (`src/tools.py`)

Cada herramienta es una función `async` decorada con `@tool` de LangChain.
El **docstring es lo único que el LLM lee** para decidir cuándo usarla, por lo
que indica explícitamente cuándo invocarla y qué devuelve.

Las 6 herramientas:

| Herramienta | Entrada | Devuelve |
|---|---|---|
| `buscar_escuderia(nombre)` | nombre | `escuderia_id` |
| `buscar_pilotos_por_escuderia(escuderia_id)` | ID numérico | lista de pilotos |
| `buscar_gran_premio(nombre, temporada?)` | nombre/país | `gran_premio_id` + pista (**sin sesiones**: obliga a usar `buscar_sesion`) |
| `buscar_sesion(gran_premio_id, tipo_sesion)` | ID + tipo | `sesion_id` de QUALY o CARRERA |
| `buscar_resultados_sesion(sesion_id)` | uno o varios IDs | posiciones, tiempos y puntos |
| `buscar_escuderias()` | — | catálogo de escuderías |

Patrones importantes:

- **Normalización** (`_normalizar`): comparación insensible a mayúsculas y
  acentos (`áéíóúüñ → aeiouun`), para que "japon" encuentre "Japón".
- **Errores como texto**: ante una entidad inexistente la herramienta devuelve
  `ERROR: ...` junto con las opciones disponibles. La observación vuelve al LLM
  (no lanza una excepción), habilitando el ciclo de retorno.
- **Desambiguación**: si una búsqueda coincide con varios GP (varias temporadas)
  devuelve un `AVISO: ...` pidiendo especificar la temporada.
- **Encadenamiento forzado**: `buscar_gran_premio` NO devuelve sesiones; el
  agente debe llamar a `buscar_sesion` para obtener los `sesion_id`. Esto
  garantiza un flujo ReAct multi-paso real.

---

## Paso 4 — Grafo ReAct con StateGraph (`src/graph.py`)

Es el corazón del proyecto. Define el **StateGraph** con el ciclo ReAct:

```
START ──► modelo ──► tools_condition ──► END
             ▲           │ tool_calls
             │           ▼
             └────── herramientas (ToolNode)
```

Componentes:

1. **`crear_llm()`** — fábrica de LLM desacoplada del grafo. Lee
   `LLM_PROVIDER` del entorno y devuelve `ChatGroq`, `ChatOpenAI` o
   `ChatGoogleGenerativeAI`. Con Groq se fija `max_tokens=900` por el límite
   del tier gratuito y `temperature=0` para respuestas determinísticas.

2. **`SYSTEM_PROMPT`** — orienta al LLM sin hacer routing manual:
   - responder siempre en castellano;
   - nunca inventar datos: usar las herramientas;
   - encadenar paso a paso (IDs → pilotos/sesiones → resultados);
   - ante un ERROR **no repetir la misma búsqueda**: informar y pedir aclaración;
   - no llamar más herramientas cuando ya tenga los datos.

3. **`nodo_modelo(state)`** — nodo del LLM. Si el historial no tiene
   `SystemMessage`, lo antepone; invoca al LLM (con `bind_tools`) y devuelve
   la respuesta al estado.

4. **`compilar_grafo(checkpointer)`** — arma el grafo:
   - `StateGraph(MessagesState)`: estado = historial de mensajes;
   - nodo `modelo` → `nodo_modelo`;
   - nodo `herramientas` → `ToolNode(herramientas, handle_tool_errors=True)`;
   - `START → modelo`;
   - `add_conditional_edges("modelo", tools_condition, ...)`: si hay
     `tool_calls` va a `herramientas`, si no, `END`;
   - `herramientas → modelo`: **el ciclo ReAct**.

5. **`RECURSION_LIMIT = 15`** — protege contra ciclos infinitos. Es 15 (y no
   10) porque el flujo mínimo con `buscar_sesion` obligatorio requiere 6 tool
   calls secuenciales (13 supersteps) con modelos que no agrupan llamadas
   paralelas.

---

## Paso 5 — Traza observable (`src/trace.py`)

La consigna pide una "traza de razonamiento". Se interpreta como **traza
observable del agente** (no se registra el chain-of-thought interno del
modelo, que es privado):

```
User Message → Tool Call → Arguments → Observation → ... → Final Answer
```

- `extraer_texto(mensaje)`: normaliza el `content` (Gemini a veces devuelve
  una lista de bloques en lugar de un string).
- `serializar_traza(mensajes)`: convierte los mensajes de LangGraph a JSON
  simple registrando tipo, contenido, `tool_calls` (nombre + argumentos) y la
  herramienta que produjo cada observación.

El resultado se guarda en `traces/react_trace.json`, con una sección por
escenario — es el entregable observable dentro del repo.

---

## Paso 6 — Punto de entrada y escenarios (`main.py`)

Orquesta todo y ejecuta los escenarios de demostración:

| Modo | Comando | Qué hace |
|---|---|---|
| Escenarios | `python main.py` | 3 escenarios + guarda la traza |
| Memoria | `python main.py --recordar` | proceso NUEVO reabre el thread y pregunta "¿Y los de McLaren?" |
| Interactivo | `python main.py --chat` | consola en tiempo real |

Los tres escenarios:

1. **Multi-paso** (`escenario_multi_paso`): "¿Cómo clasificaron los pilotos de
   Ferrari en el GP de Japón y cómo terminaron la carrera?" — requiere ≥ 5
   tool calls encadenados que el LLM decide autónomamente.
2. **Memoria** (`escenario_memoria`): "¿Y los de Mercedes?" — solo tiene
   sentido si el modelo recuerda el turno anterior (mismo `thread_id`).
3. **Error** (`probar_error`): "¿Cómo le fue a Colapinto en el GP de Brasil?"
   — no existe; `buscar_gran_premio` devuelve ERROR → el LLM **no repite** la
   búsqueda, informa el problema y pide aclaración.

Funciones compartidas:

- `ejecutar_consulta(app, config, prompt)`: invoca al grafo con un
  `HumanMessage` y la config del thread.
- `serializar_traza(...)` + guardado en `traces/react_trace.json` al final.

---

## Paso 7 — Memoria persistente (SqliteSaver + thread_id)

La memoria conversacional se apoya en dos piezas:

1. **`AsyncSqliteSaver`** — checkpointer asíncrono que guarda el estado
   completo del grafo (todo el historial de mensajes) en `checkpoints.sqlite`
   después de cada superstep.

2. **`thread_id`** — clave de la conversación dentro del checkpointer:

   ```python
   CONFIG = {
       "configurable": {"thread_id": "f1-demo-001"},
       "recursion_limit": RECURSION_LIMIT,
   }
   ```

   - Dos turnos con el mismo `thread_id` comparten historial → el agente
     "recuerda" (escenario 2: "¿Y los de Mercedes?").
   - Threads distintos (`f1-demo-error-001`, `f1-chat-001`) son conversaciones
     independientes.
   - Como el estado vive en SQLite (no en RAM), la memoria **sobrevive al
     reinicio del proceso**: `python main.py --recordar` abre un proceso
     nuevo, reutiliza el thread `f1-demo-001` y responde correctamente a
     "¿Y los de McLaren?" sin repetir contexto.

El patrón de uso es siempre el mismo:

```python
async with AsyncSqliteSaver.from_conn_string(CHECKPOINTS_DB) as checkpointer:
    app = compilar_grafo(checkpointer)
    resultado = await app.ainvoke({"messages": [HumanMessage(content=prompt)]}, config)
```

---

## Paso 8 — Modo interactivo (`--chat`)

`python main.py --chat` abre una consola en tiempo real con la misma
arquitectura (no cambia nada del grafo):

```
CLI → StateGraph → LLM → Tool Calling → ToolNode → Tools
    → Observation → LLM → respuesta
```

Características:

- **thread_id propio** (`f1-chat-001`): la memoria funciona entre preguntas y
  persiste en `checkpoints.sqlite` entre ejecuciones.
- **`/new`** genera un nuevo thread_id (`f1-chat-001-<hex>`) y arranca una
  conversación limpia con el MISMO `AsyncSqliteSaver`; la conversación previa
  queda guardada.
- Cada turno se agrega a `traces/react_trace.json` (traza observable
  acumulada, sección `escenario_chat_interactivo`).

Comandos disponibles:

| Comando | Efecto |
|---|---|
| `/help` | muestra los comandos disponibles |
| `/new` | nuevo thread_id y conversación limpia |
| `/thread` | muestra el thread_id actual |
| `/trace` | muestra la traza ReAct observable del último turno (USER → TOOL CALL/RESULT → ... → FINAL RESPONSE, sin CoT) |
| `/exit` | salir (también `exit`, `quit`, `salir`) |

Detalles de implementación:

- `input()` en un loop `while True`, con manejo de `EOFError`/`KeyboardInterrupt`
  (Ctrl+C/Ctrl+D cierran sin romper).
- `/trace` muestra solo el **último turno**: se recorta el historial desde el
  último `HumanMessage` (`indice_ultimo_user`), porque el checkpointer devuelve
  el historial completo del thread.

---

## Paso 9 — Tests (`tests/test_agent.py`)

Suite de **17 tests** (`pytest tests/ -v`, ~2-4 min porque los de integración
llaman al LLM real). Los de integración usan el decorador `@requiere_llm`: si
no hay API key en el entorno, se saltan solos.

**Datos simulados (2)** — sin red:
- `test_datos_coherentes`: integridad referencial (cada piloto apunta a una
  escudería existente, cada resultado a un piloto y sesión existentes).
- `test_resultados_normalizados`: estructura de RESULTADOS.

**Herramientas unitarias (9)** — sin red, con `await tool.ainvoke(...)`:
- `buscar_escuderia` encontrada y error controlado (con opciones disponibles).
- `buscar_pilotos_por_escuderia` (lista + error de id inválido).
- `buscar_gran_premio` por nombre/país + verificación de que **NO devuelve
  sesiones** (encadenamiento forzado vía `buscar_sesion`).
- `buscar_sesion` (QUALY y CARRERA).
- `buscar_resultados_sesion` (uno y varios `sesion_id`).

**Traza (2)** — sin red:
- `serializar_traza` registra tool_calls y observaciones correctamente.

**Integración (3)** — con LLM real:
- `test_agente_multi_paso`: el agente responde la consulta de Ferrari/Japón
  usando ≥ 2 herramientas encadenadas.
- `test_memoria_entre_turnos`: "¿Y los de Mercedes?" con el mismo thread_id
  usa el contexto del turno anterior.
- `test_ciclo_retorno_ante_error`: ante un GP inexistente hace **una sola**
  búsqueda fallida y pide aclaración (sin repetir la búsqueda ni inventar).
- `test_chat_interactivo`: el loop `--chat` con `input()` simulado
  (monkeypatch) responde dos preguntas encadenadas y sale limpio.
