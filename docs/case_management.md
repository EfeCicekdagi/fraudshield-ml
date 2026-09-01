# Case Management System (Phase 9)

## Architecture
The Case Management System relies on an SQLite backend configured to use `WAL` mode and `foreign_keys=ON`, allowing safe multi-connection read/write scenarios in a development or demo environment (can be ported to PostgreSQL).

It uses SQLAlchemy ORM mapping via `models.py` and isolates database logic in `repository.py`.

## Entities
1. **Case**: Represents an alert from the ML inference engine.
2. **CaseEvent**: Represents an immutable log of state transitions (history).

## Key Workflows
### Idempotent Creation
Cases are tied to an `inference_id`. The `CaseService.create_case` method ensures that multiple UI clicks for the same inference ID simply return the existing case, averting duplicates.

### Optimistic Concurrency Control
Updating a case requires the client to supply the current `version` integer they know about.
The UPDATE statement includes `WHERE case_id = ? AND version = ?`. If the row count modified is 0, a `409 Conflict` is returned. This prevents analysts from overwriting each other's updates simultaneously.

### Allowed Status Transitions
Hard-deletions are prevented. Status transitions are strictly enforced in `service.py`:
- `NEW` -> `UNDER_REVIEW`, `CLOSED`
- `UNDER_REVIEW` -> `CONFIRMED_FRAUD`, `FALSE_POSITIVE`, `NEW`
- `CONFIRMED_FRAUD` -> `UNDER_REVIEW`
- `FALSE_POSITIVE` -> `UNDER_REVIEW`
- `CLOSED` -> `NEW`

Every transition natively generates a `CaseEvent` within the same transaction. The `CaseEvent` audit contract strictly prefers recording `analyst-note`, priority changes, and status mutations as immutable events. It deliberately avoids storing sensitive transaction data or account identifiers inside the events.
