# Phase 9: Dashboard and Case Management Validation

## Objective
Validate the Streamlit Operational Dashboard and SQLite-backed Case Management API.

## Implementation Results
- **Dashboard Dependency:** Streamlit and SQLAlchemy were integrated into `pyproject.toml` as a `[dashboard]` optional dependency.
- **Strict Decoupling:** The Streamlit app acts strictly as an HTTP client via `APIClient`. It does not contain `import fraudshield.inference` or touch `/artifacts/final/` files.
- **Concurrency Control:** Optimistic locking via the `version` field was successfully implemented and verified in tests to return HTTP 409 during conflicts.
- **Idempotency:** Case creation uses `inference_id` as an idempotency key; duplicate requests yield a 200 OK with the original Case object.
- **Security:** CSV exports are sanitized from formula injection (e.g. `=cmd...` becomes `'=cmd...`). API Keys are securely sent in headers.

## Test Results
Phase 9 test suite: 14 passed, 0 failed.
Complete project test suite: 48 passed, 3 failed.

The Phase 9 suite successfully validates:
- `test_case_creation_and_idempotency`
- `test_case_optimistic_locking_and_transitions`
- `test_csv_sanitization`

## Known Limitations
- The current Case database uses SQLite. While sufficient for this phase and local demos, production usage spanning multiple instances will require migrating to PostgreSQL in Phase 10. This migration requires adding:
  - PostgreSQL driver support (e.g., psycopg2 or asyncpg)
  - Schema migrations (e.g., Alembic)
  - Dialect compatibility verification
  - Connection pooling
  - Concurrency integration tests
- Actor names are currently recorded via a client-provided text input field. Actual secure user authentication and OAuth2 RBAC belong to Phase 10.
- Dashboard caching is minimal; concurrent long-polling scenarios may still bottleneck single Streamlit worker instances on heavily congested networks.
