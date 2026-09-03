# FraudShield ML - Portfolio Summary

## 1. GitHub Short Description
End-to-end explainable fraud detection with PyTorch, FastAPI, Streamlit, PostgreSQL, Docker and MLOps observability.

## 2. Portfolio Description
FraudShield is a robust, production-oriented machine learning portfolio piece demonstrating the full lifecycle of a fraud detection system. Rejecting over-fit models suffering from simulation artifacts, the final system employs a strictly point-in-time-safe PyTorch MLP. It features real-time explainability via Integrated Gradients, an asynchronous FastAPI backend, a PostgreSQL-backed case management system, a Streamlit analyst dashboard, and complete Dockerized observability with Prometheus and Grafana.

## 3. CV Bullet Points
- Engineered an end-to-end fraud detection platform leveraging PyTorch, FastAPI, and Streamlit, achieving a point-in-time-safe PR-AUC of 0.74 on synthetic data.
- Conducted rigorous data leakage and simulation-artifact audits, deliberately rejecting an overfit PR-AUC 1.0 model to prioritize scientific integrity and robust generalization.
- Containerized the entire microservice ecosystem (API, Dashboard, DB) using Docker Compose, integrating Prometheus and Grafana for real-time inference observability.

## 4. LinkedIn Project Description
🚀 Just finalized **FraudShield ML**, an end-to-end explainable fraud detection platform! 

Instead of just chasing high metrics, I focused on robust MLOps and scientific integrity. After auditing a model that hit a perfect 1.0 PR-AUC, I discovered it was memorizing dataset simulation artifacts. I explicitly rejected it in favor of a strictly point-in-time-safe PyTorch MLP. 

The system features:
🔍 Real-time explainability with Captum (Integrated Gradients)
⚡ Asynchronous FastAPI backend
📊 Streamlit Analyst Dashboard with PostgreSQL case management
🐳 Complete Docker containerization with Prometheus/Grafana observability

Check out the full repository and architecture here: [GitHub Link]

## 5. Technologies Used
- **Machine Learning**: PyTorch, Scikit-Learn, LightGBM, Captum (Integrated Gradients), Pandas, NumPy.
- **Backend & API**: FastAPI, Pydantic, Uvicorn.
- **Database & ORM**: PostgreSQL, SQLAlchemy, Alembic.
- **Frontend**: Streamlit.
- **DevOps & MLOps**: Docker, Docker Compose, Prometheus, Grafana, Pytest, GitHub Actions.

## 6. Interview Talking Points
- **End-to-End Engineering**: Emphasize the ability to take a project from raw CSV EDA all the way to a containerized microservice ecosystem.
- **Leakage and Simulation Audit**: Discuss how you didn't just accept a perfect score. You investigated feature importances and realized the simulation mechanics of PaySim (specifically balance errors) were leaking the target variable.
- **Responsible Model Selection**: Explain why you chose the PyTorch MLP (safe, portable, uncoupled from legacy Scikit-learn pickling) over the artifact-dependent LightGBM model.
- **Explainable Inference**: Highlight the implementation of Integrated Gradients and human-readable Reason Codes, shifting the focus from "black box" to "analyst actionable".
- **Testing and Containerization**: Mention the rigorous test suite (>50 tests, parity checks, zero warnings) and the strictly isolated read-only Docker inference bundles.
