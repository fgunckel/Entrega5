"""Herramientas del agente ReAct de Fórmula 1.

Cada herramienta es pequeña, async y con una responsabilidad específica.
El docstring es lo único que el LLM lee para decidir cuándo usarla, por lo
que debe indicar claramente CUÁNDO debe utilizarla y qué devuelve.

Las herramientas devuelven texto estructurado. Los errores se devuelven como
mensajes "ERROR: ..." (nunca como excepciones sin controlar), de modo que la
observación vuelva al LLM y este pueda razonar de nuevo (ciclo de retorno).
"""

from __future__ import annotations

from langchain_core.tools import tool

from src.data import (
    ESCUDERIAS_DB,
    GRANDES_PREMIOS_DB,
    PILOTOS_DB,
    PISTAS_DB,
    RESULTADOS_DB,
    SESIONES_DB,
    TIPOS_SESION_DB,
)


def _normalizar(texto: str) -> str:
    """Normaliza un nombre para comparación insensible a mayúsculas/acentos."""
    reemplazos = str.maketrans("áéíóúüñ", "aeiouun")
    return texto.strip().lower().translate(reemplazos)


@tool
async def buscar_escuderia(nombre: str) -> str:
    """Busca una escudería de Fórmula 1 por su nombre y devuelve su escuderia_id.

    Usar esta herramienta SIEMPRE que el usuario mencione una escudería por su
    nombre (ej. 'Ferrari', 'Mercedes') y necesites averiguar su escuderia_id
    antes de poder consultar sus pilotos.
    Devuelve un mensaje de ERROR si el nombre no coincide con ninguna escudería
    registrada; en ese caso, verifica la escritura o pide aclaración al usuario.
    """
    objetivo = _normalizar(nombre)
    for escuderia in ESCUDERIAS_DB:
        if _normalizar(escuderia["nombre"]) == objetivo:
            return (
                f"Escudería encontrada: '{escuderia['nombre']}' "
                f"→ escuderia_id={escuderia['id']}"
            )
    return (
        f"ERROR: no se encontró ninguna escudería con el nombre '{nombre}'. "
        "Escuderías disponibles: "
        + ", ".join(e["nombre"] for e in ESCUDERIAS_DB)
    )


@tool
async def buscar_pilotos_por_escuderia(escuderia_id: int) -> str:
    """Busca los pilotos de una escudería, dado su escuderia_id NUMÉRICO.

    Usar esta herramienta cuando ya tengas el escuderia_id NUMÉRICO (obtenido
    con buscar_escuderia) y necesites saber qué pilotos corren para esa
    escudería. Si solo tenés el nombre de la escudería, primero usá
    buscar_escuderia para obtener el ID.
    Devuelve la lista de pilotos con su id, nombre y número, o un mensaje de
    ERROR si el escuderia_id no existe.
    """
    pilotos = [p for p in PILOTOS_DB if p["escuderia_id"] == escuderia_id]
    if not pilotos:
        return (
            f"ERROR: no existe ninguna escudería con id={escuderia_id} o no tiene "
            "pilotos registrados. Usá buscar_escuderia para obtener un id válido."
        )
    lineas = [
        f"- {p['nombre']} (n°{p['numero']}, piloto_id={p['id']})" for p in pilotos
    ]
    return "Pilotos encontrados:\n" + "\n".join(lineas)


@tool
async def buscar_gran_premio(nombre: str, temporada: int | None = None) -> str:
    """Busca un Gran Premio por su nombre (o parte del nombre) y devuelve su
    gran_premio_id, la temporada y la pista donde se corre.

    Usar esta herramienta SIEMPRE que el usuario mencione un Gran Premio o un
    país/circuito (ej. 'Japón', 'GP de Japón', 'Mónaco') y necesites averiguar
    su gran_premio_id antes de consultar sesiones o resultados.
    Si hay varias temporadas, podés filtrar con el parámetro temporada.
    Para obtener el sesion_id de la QUALY o de la CARRERA, usá después
    buscar_sesion con el gran_premio_id devuelto.
    Devuelve un mensaje de ERROR si no se encuentra ningún Gran Premio.
    """
    objetivo = _normalizar(nombre)
    coincidencias = [
        gp for gp in GRANDES_PREMIOS_DB
        if objetivo in _normalizar(gp["nombre"])
        or any(
            objetivo in _normalizar(pista["nombre"])
            or objetivo in _normalizar(pista["pais"])
            for pista in PISTAS_DB
            if pista["id"] == gp["pista_id"]
        )
    ]
    if not coincidencias:
        return (
            f"ERROR: no se encontró ningún Gran Premio que coincida con '{nombre}'. "
            "Gran Premios disponibles: "
            + ", ".join(f"{gp['nombre']} {gp['temporada']}" for gp in GRANDES_PREMIOS_DB)
        )
    if len(coincidencias) > 1 and temporada is None:
        detalle = "; ".join(
            f"{gp['nombre']} {gp['temporada']} (gran_premio_id={gp['id']})"
            for gp in coincidencias
        )
        return (
            f"AVISO: se encontraron varios Gran Premios para '{nombre}': {detalle}. "
            "Especificá la temporada para desambiguar."
        )
    gp = coincidencias[0]
    pista = next(p for p in PISTAS_DB if p["id"] == gp["pista_id"])
    return (
        f"Gran Premio encontrado: {gp['nombre']} (temporada {gp['temporada']}, "
        f"pista {pista['nombre']}, {pista['pais']}) → gran_premio_id={gp['id']}"
    )


@tool
async def buscar_sesion(gran_premio_id: int, tipo_sesion: str) -> str:
    """Busca la sesión de un tipo específico (FP1, FP2, FP3, SPRINT_QUALY,
    SPRINT, QUALY o CARRERA) dentro de un Gran Premio, dado su gran_premio_id
    NUMÉRICO.

    Usar esta herramienta cuando ya conozcas el gran_premio_id (obtenido con
    buscar_gran_premio) y necesites el sesion_id de la QUALY o de la CARRERA
    para poder consultar sus resultados.
    Devuelve un mensaje de ERROR si el gran_premio_id no existe o si no hay
    ninguna sesión de ese tipo para ese Gran Premio.
    """
    if not any(gp["id"] == gran_premio_id for gp in GRANDES_PREMIOS_DB):
        return (
            f"ERROR: no existe ningún Gran Premio con id={gran_premio_id}. "
            "Usá buscar_gran_premio para obtener un gran_premio_id válido."
        )
    tipo_normalizado = tipo_sesion.strip().upper()
    tipo = next(
        (t for t in TIPOS_SESION_DB if t["nombre"] == tipo_normalizado), None
    )
    if tipo is None:
        return (
            f"ERROR: el tipo de sesión '{tipo_sesion}' no existe. Tipos válidos: "
            + ", ".join(t["nombre"] for t in TIPOS_SESION_DB)
        )
    sesion = next(
        (
            s for s in SESIONES_DB
            if s["gran_premio_id"] == gran_premio_id
            and s["tipo_sesion_id"] == tipo["id"]
        ),
        None,
    )
    if sesion is None:
        return (
            f"ERROR: el Gran Premio con id={gran_premio_id} no tiene ninguna "
            f"sesión de tipo {tipo['nombre']}. Sesiones disponibles: "
            + ", ".join(
                next(t["nombre"] for t in TIPOS_SESION_DB if t["id"] == s["tipo_sesion_id"])
                for s in SESIONES_DB
                if s["gran_premio_id"] == gran_premio_id
            )
        )
    return (
        f"Sesión encontrada: {tipo['nombre']} del gran_premio_id={gran_premio_id}, "
        f"fecha {sesion['fecha']} → sesion_id={sesion['id']}"
    )


@tool
async def buscar_resultados_sesion(sesion_id: int | list[int]) -> str:
    """Busca los resultados (posiciones, tiempos y puntos) de UNA o VARIAS
    sesiones, dado su sesion_id NUMÉRICO o una lista de sesion_id.

    Usar esta herramienta cuando ya tengas el/los sesion_id NUMÉRICOS
    (obtenidos con buscar_gran_premio o buscar_sesion). Si pasás una lista,
    devuelve los resultados de todas esas sesiones en una sola respuesta.
    Devuelve los resultados de TODOS los pilotos de cada sesión, ordenados
    por posición; los pilotos sin posición (ej. DNF) aparecen al final.
    El tipo de resultado (QUALY, CARRERA, etc.) se determina automáticamente
    vía SESIONES -> TIPOS_SESION.
    Devuelve un mensaje de ERROR si algún sesion_id no existe.
    """
    ids = sesion_id if isinstance(sesion_id, list) else [sesion_id]
    if not ids or not all(isinstance(i, int) for i in ids):
        return "ERROR: sesion_id debe ser un entero o una lista de enteros."
    bloques = []
    for id_sesion in ids:
        sesion = next((s for s in SESIONES_DB if s["id"] == id_sesion), None)
        if sesion is None:
            return (
                f"ERROR: no existe ninguna sesión con id={id_sesion}. "
                "Usá buscar_gran_premio o buscar_sesion para obtener sesion_id válidos."
            )
        tipo = next(
            t["nombre"] for t in TIPOS_SESION_DB if t["id"] == sesion["tipo_sesion_id"]
        )
        resultados = [r for r in RESULTADOS_DB if r["sesion_id"] == id_sesion]
        lineas = []
        for r in sorted(resultados, key=lambda x: (x["posicion"] is None, x["posicion"] or 999)):
            piloto = next(p for p in PILOTOS_DB if p["id"] == r["piloto_id"])
            if r["posicion"] is None:
                estado = "sin clasificar (DNF o no completó la sesión)"
            else:
                estado = f"posición {r['posicion']}, tiempo {r['tiempo']}"
                if r["puntos"] is not None:
                    estado += f", {r['puntos']:.1f} puntos"
            lineas.append(f"- {piloto['nombre']}: {estado}")
        bloques.append(
            f"Resultados de la sesión {tipo} (sesion_id={id_sesion}):\n"
            + "\n".join(lineas)
        )
    return "\n\n".join(bloques)


herramientas = [
    buscar_escuderia,
    buscar_pilotos_por_escuderia,
    buscar_gran_premio,
    buscar_sesion,
    buscar_resultados_sesion,
]
