# PostgreSQL Setup

FraudShield uses PostgreSQL for durable storage of case management, alerts, and audit history.

## Schema
- **cases**: Core case attributes (inference_id, risk_level, status, etc.).
- **case_events**: Audit history tracking (`previous_value`, `new_value`, and `audit_context`).

## Connection Pooling
The application connects via SQLAlchemy. The API uses connection pooling configured in `db.py`:
- `pool_size`: 5
- `max_overflow`: 10
- `pool_timeout`: 30s
- `pool_recycle`: 1800s

## Migrations
Alembic manages migrations. The `migrate` container runs `alembic upgrade head` before the API starts.

## Security
PostgreSQL is configured to run inside the Docker network. The port `5432` is not published to the host machine to prevent unauthorized access. In production, never expose the database port directly to the public internet.
