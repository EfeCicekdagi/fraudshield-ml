# 5-Minute Demonstration Flow

This guide walks you through launching the entire FraudShield ecosystem locally using Docker Compose, processing a transaction, and investigating the generated alert.

## 1. Configure Environment

Ensure you are in the project root. Create or use placeholder environment values for the API key.

**PowerShell:**
```powershell
$env:FRAUDSHIELD_API_KEY="test_key"
$env:FRAUDSHIELD_DATABASE_URL="postgresql://fraudshield:password@db:5432/cases"
```

**Bash / Zsh:**
```bash
export FRAUDSHIELD_API_KEY="test_key"
export FRAUDSHIELD_DATABASE_URL="postgresql://fraudshield:password@db:5432/cases"
```

## 2. Start the Ecosystem

Start all services (FastAPI, Streamlit, PostgreSQL, Prometheus, Grafana) in the background:

```bash
docker compose up -d --build
```

## 3. Verify Readiness

Check that all containers are running successfully:

```bash
docker compose ps
```

## 4. Open the Dashboard

Navigate to [http://localhost:8501](http://localhost:8501) in your browser. You will see the FraudShield Analyst Dashboard.

## 5. Score a Synthetic Transaction

You can manually trigger a synthetic transaction using `curl` or by running the provided seed script:

```bash
python scripts/seed_demo_cases.py
```

## 6. Request an Explanation

In the dashboard, click on the newly generated alert. The UI will call the `/api/v1/explain` endpoint and render the Integrated Gradients feature attributions alongside the human-readable **Reason Codes** (e.g., `RC001: High Transfer Amount`).

## 7. Manage the Case

- Assign the case to yourself in the dashboard.
- Review the immutable event history showing the case creation and your assignment.
- Add a comment (e.g., "Confirmed synthetic demo payload").

## 8. Change the Case Status

Click **"Close as True Positive"** or **"Close as False Positive"** in the UI.

## 9. View Immutable History

Notice that every action you took (Status Change, Comment, Assignment) is cryptographically or sequentially logged in the Case Events tab.

## 10. View Observability Metrics

Navigate to [http://localhost:3000](http://localhost:3000) (Grafana). 
Log in (default: `admin`/`admin`). 
View the FastAPI observability metrics tracking your API latency and prediction request counts.

## 11. Shut Down

When finished, shut down the containers. 

```bash
docker compose down
```

> [!WARNING]
> Do NOT use `docker compose down -v` unless you explicitly want to permanently delete your PostgreSQL case database and Grafana dashboards!
