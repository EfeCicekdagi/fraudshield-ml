# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-09-03
### Added
- Complete end-to-end PyTorch MLP pipeline for fraud detection.
- Fast API service for point-in-time-safe inference.
- Integrated Gradients (Captum) for explainable predictions and reason codes.
- Version-independent inference bundle logic (no Scikit-Learn `.joblib` pickling).
- PostgreSQL integration with Alembic migrations for case management.
- Streamlit dashboard for analyst operations.
- Docker Compose ecosystem including Prometheus and Grafana observability.
- Comprehensive technical documentation and demonstration assets.
- Complete CI/CD testing suite ensuring zero unhandled warnings and strong parity.
