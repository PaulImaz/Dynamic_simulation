# Performance Lab (Standalone)

Aplicacion separada para la simulacion dinamica de cargas en rueda:

- `Fz` mecanica (peso estatico + transferencia longitudinal/lateral)
- `Fz` aero (front/rear split por polinomios)
- `Fy` por rueda con slip angle/slip ratio (modelo MF simplificado)
- `Fx` por rueda con slip ratio (modelo MF simplificado)

Esta version es independiente: incluye sus propios modulos locales
`body_attitude.py` y `center_map_tool_v5.py` dentro de esta carpeta.

## Requisitos

- Python 3.8+
- `flask`, `numpy`, `pandas`, `matplotlib`

## Arranque

1. Ejecuta:

```powershell
python app.py
```

2. Abre:

`http://127.0.0.1:6060`

## Endpoints

- `GET /api/health`
- `POST /api/load_upload` (multipart file o `json_data`)
- `POST /api/simulate_dynamic_aero`
- `GET /api/export_dynamic_csv`
