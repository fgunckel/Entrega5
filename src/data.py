"""Base de datos simulada de Fórmula 1, en memoria.

Los datos son ficticios y coherentes entre sí (se aclara en el README).
No se usa SQLite para los datos de F1: SQLite se usa únicamente como
Checkpointer de LangGraph.

Modelo conceptual:

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
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# ESCUDERIAS
# ---------------------------------------------------------------------------

ESCUDERIAS_DB: list[dict] = [
    {"id": 1, "nombre": "Ferrari"},
    {"id": 2, "nombre": "Mercedes"},
    {"id": 3, "nombre": "McLaren"},
    {"id": 4, "nombre": "Red Bull"},
    {"id": 5, "nombre": "Williams"},
    {"id": 6, "nombre": "Aston Martin"},
]

# ---------------------------------------------------------------------------
# PILOTOS
# ---------------------------------------------------------------------------

PILOTOS_DB: list[dict] = [
    {"id": 1, "nombre": "Charles Leclerc", "numero": 16, "escuderia_id": 1},
    {"id": 2, "nombre": "Carlos Sainz", "numero": 55, "escuderia_id": 1},
    {"id": 3, "nombre": "George Russell", "numero": 63, "escuderia_id": 2},
    {"id": 4, "nombre": "Andrea Kimi Antonelli", "numero": 12, "escuderia_id": 2},
    {"id": 5, "nombre": "Lando Norris", "numero": 4, "escuderia_id": 3},
    {"id": 6, "nombre": "Oscar Piastri", "numero": 81, "escuderia_id": 3},
    {"id": 7, "nombre": "Max Verstappen", "numero": 1, "escuderia_id": 4},
    {"id": 8, "nombre": "Yuki Tsunoda", "numero": 22, "escuderia_id": 4},
    {"id": 9, "nombre": "Alex Albon", "numero": 23, "escuderia_id": 5},
    {"id": 10, "nombre": "Franco Colapinto", "numero": 43, "escuderia_id": 5},
    {"id": 11, "nombre": "Fernando Alonso", "numero": 14, "escuderia_id": 6},
    {"id": 12, "nombre": "Lance Stroll", "numero": 18, "escuderia_id": 6},
]

# ---------------------------------------------------------------------------
# PISTAS
# ---------------------------------------------------------------------------

PISTAS_DB: list[dict] = [
    {"id": 1, "nombre": "Suzuka", "pais": "Japón"},
    {"id": 2, "nombre": "Monza", "pais": "Italia"},
    {"id": 3, "nombre": "Silverstone", "pais": "Reino Unido"},
    {"id": 4, "nombre": "Mónaco", "pais": "Mónaco"},
]

# ---------------------------------------------------------------------------
# GRANDES_PREMIOS
# ---------------------------------------------------------------------------

GRANDES_PREMIOS_DB: list[dict] = [
    {"id": 1, "nombre": "GP de Japón", "temporada": 2026, "pista_id": 1},
    {"id": 2, "nombre": "GP de Italia", "temporada": 2026, "pista_id": 2},
    {"id": 3, "nombre": "GP de Reino Unido", "temporada": 2026, "pista_id": 3},
    {"id": 4, "nombre": "GP de Mónaco", "temporada": 2026, "pista_id": 4},
]

# ---------------------------------------------------------------------------
# TIPOS_SESION — entidad separada (sin flags como is_qualy).
# Permite agregar nuevos tipos de sesión sin modificar RESULTADOS.
# ---------------------------------------------------------------------------

TIPOS_SESION_DB: list[dict] = [
    {"id": 1, "nombre": "FP1"},
    {"id": 2, "nombre": "FP2"},
    {"id": 3, "nombre": "FP3"},
    {"id": 4, "nombre": "SPRINT_QUALY"},
    {"id": 5, "nombre": "SPRINT"},
    {"id": 6, "nombre": "QUALY"},
    {"id": 7, "nombre": "CARRERA"},
]

# ---------------------------------------------------------------------------
# SESIONES — QUALY y CARRERA para cada Gran Premio.
# ---------------------------------------------------------------------------

SESIONES_DB: list[dict] = [
    {"id": 1, "gran_premio_id": 1, "tipo_sesion_id": 6, "fecha": "2026-04-04"},
    {"id": 2, "gran_premio_id": 1, "tipo_sesion_id": 7, "fecha": "2026-04-05"},
    {"id": 3, "gran_premio_id": 2, "tipo_sesion_id": 6, "fecha": "2026-05-09"},
    {"id": 4, "gran_premio_id": 2, "tipo_sesion_id": 7, "fecha": "2026-05-10"},
    {"id": 5, "gran_premio_id": 3, "tipo_sesion_id": 6, "fecha": "2026-07-04"},
    {"id": 6, "gran_premio_id": 3, "tipo_sesion_id": 7, "fecha": "2026-07-05"},
    {"id": 7, "gran_premio_id": 4, "tipo_sesion_id": 6, "fecha": "2026-05-23"},
    {"id": 8, "gran_premio_id": 4, "tipo_sesion_id": 7, "fecha": "2026-05-24"},
]

# ---------------------------------------------------------------------------
# RESULTADOS — una única tabla para QUALY y CARRERA (normalización).
# El tipo de resultado se determina vía RESULTADOS -> SESIONES -> TIPOS_SESION.
# posicion/tiempo/puntos son None cuando no aplican (ej. DNF en carrera).
# ---------------------------------------------------------------------------

RESULTADOS_DB: list[dict] = [
    # --- GP de Japón 2026 — QUALY (sesion_id=1) ---
    {"id": 1, "sesion_id": 1, "piloto_id": 7, "posicion": 1, "tiempo": "1:28.772", "puntos": None},
    {"id": 2, "sesion_id": 1, "piloto_id": 1, "posicion": 2, "tiempo": "1:29.104", "puntos": None},
    {"id": 3, "sesion_id": 1, "piloto_id": 5, "posicion": 3, "tiempo": "1:29.318", "puntos": None},
    {"id": 4, "sesion_id": 1, "piloto_id": 3, "posicion": 4, "tiempo": "1:29.402", "puntos": None},
    {"id": 5, "sesion_id": 1, "piloto_id": 2, "posicion": 5, "tiempo": "1:29.510", "puntos": None},
    {"id": 6, "sesion_id": 1, "piloto_id": 6, "posicion": 6, "tiempo": "1:29.644", "puntos": None},
    {"id": 7, "sesion_id": 1, "piloto_id": 4, "posicion": 7, "tiempo": "1:29.701", "puntos": None},
    {"id": 8, "sesion_id": 1, "piloto_id": 8, "posicion": 8, "tiempo": "1:29.795", "puntos": None},
    {"id": 9, "sesion_id": 1, "piloto_id": 9, "posicion": 9, "tiempo": "1:30.015", "puntos": None},
    {"id": 10, "sesion_id": 1, "piloto_id": 10, "posicion": 10, "tiempo": "1:30.455", "puntos": None},
    {"id": 11, "sesion_id": 1, "piloto_id": 11, "posicion": 11, "tiempo": "1:30.620", "puntos": None},
    {"id": 12, "sesion_id": 1, "piloto_id": 12, "posicion": 12, "tiempo": "1:30.744", "puntos": None},
    # --- GP de Japón 2026 — CARRERA (sesion_id=2) ---
    {"id": 13, "sesion_id": 2, "piloto_id": 7, "posicion": 1, "tiempo": "1:32:45.120", "puntos": 25.0},
    {"id": 14, "sesion_id": 2, "piloto_id": 5, "posicion": 2, "tiempo": "+5.412", "puntos": 18.0},
    {"id": 15, "sesion_id": 2, "piloto_id": 3, "posicion": 3, "tiempo": "+12.880", "puntos": 15.0},
    {"id": 16, "sesion_id": 2, "piloto_id": 1, "posicion": 4, "tiempo": "+18.402", "puntos": 12.0},
    {"id": 17, "sesion_id": 2, "piloto_id": 2, "posicion": 5, "tiempo": "+21.880", "puntos": 10.0},
    {"id": 18, "sesion_id": 2, "piloto_id": 6, "posicion": 6, "tiempo": "+48.301", "puntos": 8.0},
    {"id": 19, "sesion_id": 2, "piloto_id": 4, "posicion": 7, "tiempo": "+1:01.204", "puntos": 6.0},
    {"id": 20, "sesion_id": 2, "piloto_id": 8, "posicion": 8, "tiempo": "+1:05.884", "puntos": 4.0},
    {"id": 21, "sesion_id": 2, "piloto_id": 9, "posicion": 9, "tiempo": "+1:10.015", "puntos": 2.0},
    {"id": 22, "sesion_id": 2, "piloto_id": 10, "posicion": None, "tiempo": None, "puntos": None},
    {"id": 23, "sesion_id": 2, "piloto_id": 11, "posicion": 10, "tiempo": "+1:15.880", "puntos": 0.0},
    {"id": 24, "sesion_id": 2, "piloto_id": 12, "posicion": 12, "tiempo": "+1:22.890", "puntos": 0.0},
    # --- GP de Italia 2026 — QUALY (sesion_id=3) ---
    {"id": 25, "sesion_id": 3, "piloto_id": 5, "posicion": 1, "tiempo": "1:19.532", "puntos": None},
    {"id": 26, "sesion_id": 3, "piloto_id": 3, "posicion": 2, "tiempo": "1:19.610", "puntos": None},
    {"id": 27, "sesion_id": 3, "piloto_id": 1, "posicion": 3, "tiempo": "1:19.744", "puntos": None},
    {"id": 28, "sesion_id": 3, "piloto_id": 7, "posicion": 4, "tiempo": "1:19.801", "puntos": None},
    {"id": 29, "sesion_id": 3, "piloto_id": 6, "posicion": 5, "tiempo": "1:19.955", "puntos": None},
    {"id": 30, "sesion_id": 3, "piloto_id": 2, "posicion": 6, "tiempo": "1:20.101", "puntos": None},
    {"id": 31, "sesion_id": 3, "piloto_id": 4, "posicion": 7, "tiempo": "1:20.188", "puntos": None},
    {"id": 32, "sesion_id": 3, "piloto_id": 8, "posicion": 8, "tiempo": "1:20.310", "puntos": None},
    {"id": 33, "sesion_id": 3, "piloto_id": 9, "posicion": 9, "tiempo": "1:20.640", "puntos": None},
    {"id": 34, "sesion_id": 3, "piloto_id": 10, "posicion": 10, "tiempo": "1:21.010", "puntos": None},
    {"id": 35, "sesion_id": 3, "piloto_id": 11, "posicion": 11, "tiempo": "1:21.155", "puntos": None},
    {"id": 36, "sesion_id": 3, "piloto_id": 12, "posicion": 12, "tiempo": "1:21.288", "puntos": None},
    # --- GP de Italia 2026 — CARRERA (sesion_id=4) ---
    {"id": 37, "sesion_id": 4, "piloto_id": 5, "posicion": 1, "tiempo": "1:21:10.440", "puntos": 25.0},
    {"id": 38, "sesion_id": 4, "piloto_id": 3, "posicion": 2, "tiempo": "+3.201", "puntos": 18.0},
    {"id": 39, "sesion_id": 4, "piloto_id": 1, "posicion": 3, "tiempo": "+7.940", "puntos": 15.0},
    {"id": 40, "sesion_id": 4, "piloto_id": 7, "posicion": 4, "tiempo": "+11.204", "puntos": 12.0},
    {"id": 41, "sesion_id": 4, "piloto_id": 2, "posicion": 5, "tiempo": "+15.320", "puntos": 10.0},
    {"id": 42, "sesion_id": 4, "piloto_id": 6, "posicion": 6, "tiempo": "+21.880", "puntos": 8.0},
    {"id": 43, "sesion_id": 4, "piloto_id": 4, "posicion": 7, "tiempo": "+35.401", "puntos": 6.0},
    {"id": 44, "sesion_id": 4, "piloto_id": 8, "posicion": 8, "tiempo": "+41.002", "puntos": 4.0},
    {"id": 45, "sesion_id": 4, "piloto_id": 9, "posicion": 9, "tiempo": "+55.010", "puntos": 2.0},
    {"id": 46, "sesion_id": 4, "piloto_id": 10, "posicion": 10, "tiempo": "+58.310", "puntos": 0.0},
    {"id": 47, "sesion_id": 4, "piloto_id": 11, "posicion": 11, "tiempo": "+1:02.500", "puntos": 0.0},
    {"id": 48, "sesion_id": 4, "piloto_id": 12, "posicion": 12, "tiempo": "+1:15.002", "puntos": 0.0},
    # --- GP de Reino Unido 2026 — QUALY (sesion_id=5) ---
    {"id": 49, "sesion_id": 5, "piloto_id": 1, "posicion": 1, "tiempo": "1:26.720", "puntos": None},
    {"id": 50, "sesion_id": 5, "piloto_id": 5, "posicion": 2, "tiempo": "1:26.810", "puntos": None},
    {"id": 51, "sesion_id": 5, "piloto_id": 7, "posicion": 3, "tiempo": "1:26.905", "puntos": None},
    {"id": 52, "sesion_id": 5, "piloto_id": 3, "posicion": 4, "tiempo": "1:27.012", "puntos": None},
    {"id": 53, "sesion_id": 5, "piloto_id": 2, "posicion": 5, "tiempo": "1:27.105", "puntos": None},
    {"id": 54, "sesion_id": 5, "piloto_id": 6, "posicion": 6, "tiempo": "1:27.155", "puntos": None},
    {"id": 55, "sesion_id": 5, "piloto_id": 4, "posicion": 7, "tiempo": "1:27.188", "puntos": None},
    {"id": 56, "sesion_id": 5, "piloto_id": 8, "posicion": 8, "tiempo": "1:27.310", "puntos": None},
    {"id": 57, "sesion_id": 5, "piloto_id": 9, "posicion": 9, "tiempo": "1:27.402", "puntos": None},
    {"id": 58, "sesion_id": 5, "piloto_id": 10, "posicion": 10, "tiempo": "1:28.010", "puntos": None},
    {"id": 59, "sesion_id": 5, "piloto_id": 11, "posicion": 11, "tiempo": "1:28.155", "puntos": None},
    {"id": 60, "sesion_id": 5, "piloto_id": 12, "posicion": 12, "tiempo": "1:28.588", "puntos": None},
    # --- GP de Reino Unido 2026 — CARRERA (sesion_id=6) ---
    {"id": 61, "sesion_id": 6, "piloto_id": 1, "posicion": 1, "tiempo": "1:25:30.110", "puntos": 25.0},
    {"id": 62, "sesion_id": 6, "piloto_id": 5, "posicion": 2, "tiempo": "+2.150", "puntos": 18.0},
    {"id": 63, "sesion_id": 6, "piloto_id": 7, "posicion": 3, "tiempo": "+8.440", "puntos": 15.0},
    {"id": 64, "sesion_id": 6, "piloto_id": 3, "posicion": 4, "tiempo": "+11.880", "puntos": 12.0},
    {"id": 65, "sesion_id": 6, "piloto_id": 2, "posicion": 5, "tiempo": "+15.010", "puntos": 10.0},
    {"id": 66, "sesion_id": 6, "piloto_id": 6, "posicion": 5, "tiempo": "+22.880", "puntos": 10.0},
    {"id": 67, "sesion_id": 6, "piloto_id": 4, "posicion": 7, "tiempo": "+30.402", "puntos": 6.0},
    {"id": 68, "sesion_id": 6, "piloto_id": 8, "posicion": 8, "tiempo": "+38.002", "puntos": 4.0},
    {"id": 69, "sesion_id": 6, "piloto_id": 9, "posicion": 9, "tiempo": "+55.010", "puntos": 2.0},
    {"id": 70, "sesion_id": 6, "piloto_id": 10, "posicion": None, "tiempo": None, "puntos": None},
    {"id": 71, "sesion_id": 6, "piloto_id": 11, "posicion": 11, "tiempo": "+1:05.880", "puntos": 0.0},
    {"id": 72, "sesion_id": 6, "piloto_id": 12, "posicion": 12, "tiempo": "+1:12.890", "puntos": 0.0},
    # --- GP de Mónaco 2026 — QUALY (sesion_id=7) ---
    {"id": 73, "sesion_id": 7, "piloto_id": 5, "posicion": 1, "tiempo": "1:10.220", "puntos": None},
    {"id": 74, "sesion_id": 7, "piloto_id": 1, "posicion": 2, "tiempo": "1:10.810", "puntos": None},
    {"id": 75, "sesion_id": 7, "piloto_id": 3, "posicion": 3, "tiempo": "1:11.012", "puntos": None},
    {"id": 76, "sesion_id": 7, "piloto_id": 7, "posicion": 4, "tiempo": "1:11.155", "puntos": None},
    {"id": 77, "sesion_id": 7, "piloto_id": 2, "posicion": 5, "tiempo": "1:11.402", "puntos": None},
    {"id": 78, "sesion_id": 7, "piloto_id": 6, "posicion": 6, "tiempo": "1:11.544", "puntos": None},
    {"id": 79, "sesion_id": 7, "piloto_id": 4, "posicion": 7, "tiempo": "1:12.010", "puntos": None},
    {"id": 80, "sesion_id": 7, "piloto_id": 8, "posicion": 8, "tiempo": "1:12.155", "puntos": None},
    {"id": 81, "sesion_id": 7, "piloto_id": 9, "posicion": 9, "tiempo": "1:12.402", "puntos": None},
    {"id": 82, "sesion_id": 7, "piloto_id": 10, "posicion": 10, "tiempo": "1:13.010", "puntos": None},
    {"id": 81, "sesion_id": 7, "piloto_id": 11, "posicion": 11, "tiempo": "1:13.155", "puntos": None},
    {"id": 82, "sesion_id": 7, "piloto_id": 12, "posicion": 12, "tiempo": "1:13.588", "puntos": None},
    # --- GP de Mónaco 2026 — CARRERA (sesion_id=8) ---
    {"id": 85, "sesion_id": 8, "piloto_id": 5, "posicion": 1, "tiempo": "1:42:10.550", "puntos": 25.0},
    {"id": 86, "sesion_id": 8, "piloto_id": 1, "posicion": 2, "tiempo": "+2.880", "puntos": 18.0},
    {"id": 87, "sesion_id": 8, "piloto_id": 3, "posicion": 3, "tiempo": "+7.010", "puntos": 15.0},
    {"id": 88, "sesion_id": 8, "piloto_id": 7, "posicion": 4, "tiempo": "+11.204", "puntos": 12.0},
    {"id": 87, "sesion_id": 8, "piloto_id": 2, "posicion": 5, "tiempo": "+15.880", "puntos": 10.0},
    {"id": 88, "sesion_id": 8, "piloto_id": 6, "posicion": 6, "tiempo": "+20.010", "puntos": 8.0},
    {"id": 89, "sesion_id": 8, "piloto_id": 4, "posicion": 7, "tiempo": "+25.010", "puntos": 6.0},
    {"id": 90, "sesion_id": 8, "piloto_id": 8, "posicion": 8, "tiempo": "+31.880", "puntos": 4.0},
    {"id": 91, "sesion_id": 8, "piloto_id": 9, "posicion": 9, "tiempo": "+35.010", "puntos": 2.0},
    {"id": 92, "sesion_id": 8, "piloto_id": 10, "posicion": 10, "tiempo": "+40.010", "puntos": 0.0},
    {"id": 91, "sesion_id": 8, "piloto_id": 11, "posicion": 11, "tiempo": "+45.010", "puntos": 0.0},
    {"id": 92, "sesion_id": 8, "piloto_id": 12, "posicion": 12, "tiempo": "+50.880", "puntos": 0.0},
]
