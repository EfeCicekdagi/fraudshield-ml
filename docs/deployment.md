# Deployment Guide

The application is a production-oriented, containerized ML inference system managed via Docker Compose.

## Services
- **postgres**: PostgreSQL database for case management.
- **migrate**: Ephemeral container to run Alembic migrations on startup.
- **api**: FastAPI inference and case management application.
- **dashboard**: Streamlit dashboard for analysts.
- **prometheus**: Time-series database for metrics.
- **grafana**: Observability dashboard.

## Security Considerations
In a true production environment, **do not** expose `/docs`, Grafana, PostgreSQL, or `/metrics` to the public internet. Ensure appropriate ingress configurations (like NGINX or an API Gateway) protect these resources and enforce authentication.

## Commands

### Development
Start the application without observability services:
```bash
docker compose up -d --build
```

### With Observability
Start the application with Prometheus and Grafana:
```bash
docker compose --profile observability up -d --build
```
*(Note: To use the profile feature, you must add `profiles: ["observability"]` to the `prometheus` and `grafana` services in `docker-compose.yml`. By default, they start together if profiles aren't used.)*

### Shutdown
Shutdown the containers without deleting the database volume:
```bash
docker compose down
```

### Destructive Volume Removal
> **WARNING**: The following command will completely delete the PostgreSQL database and Grafana persistent volumes. This action is irreversible!

```bash
docker compose down -v
```

## Scaling
To scale the API, ensure you update the `docker-compose.yml` to support multiple replicas or deploy via Kubernetes using the provided `Dockerfile`.
