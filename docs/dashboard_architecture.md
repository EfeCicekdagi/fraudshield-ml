# Dashboard Architecture (Phase 9)

## Overview
The FraudShield ML Dashboard is a Streamlit-based web application that serves as the operational UI for fraud analysts.

## API-Driven Design
The dashboard strictly adheres to a decoupled, API-driven architecture. It **never**:
1. Imports `fraudshield.inference` or loads ML artifacts directly.
2. Uses or even knows about the existence of the FastAPI bundle loader.
3. Directly touches `data/cases.db` (except through FastAPI).
4. Evaluates models on the test dataset.

All model inference, explanation, and case state changes happen by communicating with the underlying FastAPI backend (`http://127.0.0.1:8000/api/v1`).

## Streamlit Multipage Structure
We leverage `st.Page` and `st.navigation` for routing:
- **`app.py`**: Entry point and router.
- **`overview.py`**: Real-time aggregated metrics from `/api/v1/dashboard/summary`.
- **`live_scoring.py`**: Web form for a single transaction. Predicts via `/predict` and optionally explains via `/explain`.
- **`batch_scoring.py`**: Chunk-based CSV processor respecting the API's `MAX_BATCH_SIZE`. Ensures safe CSV exports.
- **`cases.py`**: Queue visualization and interactive case update panel.
- **`model_information.py`**: Metadata viewer relying on `/api/v1/model-info`.

## Security & State Management
- **API Keys**: Provided via `FRAUDSHIELD_API_KEY` env var. The `APIClient` adds this to HTTP headers. Keys are never printed to the UI.
- **State**: `st.session_state` is used solely for non-persistent UI state (e.g. keeping the result of the last inference request while rendering explanations).
- **Injection Safety**: Exported CSVs from batch operations escape `=, +, -, @` to prevent spreadsheet formula injection.
