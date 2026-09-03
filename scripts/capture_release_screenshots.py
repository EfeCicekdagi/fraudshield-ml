import time
import os
from playwright.sync_api import sync_playwright

def run(playwright):
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context(viewport={"width": 1440, "height": 1000})
    page = context.new_page()

    os.makedirs("docs/assets/screenshots", exist_ok=True)

    try:
        # 1. API Documentation
        print("Capturing API Documentation...")
        page.goto("http://127.0.0.1:8000/docs")
        page.wait_for_selector(".swagger-ui")
        time.sleep(2)  # Allow fonts and styles to settle
        page.screenshot(path="docs/assets/screenshots/api-documentation.png")

        # 2. Dashboard Overview (Streamlit)
        print("Capturing Dashboard Overview...")
        page.goto("http://127.0.0.1:8501")
        # Wait for Streamlit to render its main content container
        page.wait_for_selector("div.stApp", timeout=30000)
        # Wait for the specific markdown/titles that show data is loaded
        page.wait_for_selector("text=FraudShield ML Overview", timeout=30000)
        time.sleep(5)  # Streamlit websocket data loading padding
        
        # Hide the Streamlit main menu and footer for a cleaner screenshot
        page.evaluate("""() => {
            const menu = document.querySelector('[data-testid="stHeader"]');
            if (menu) menu.style.display = 'none';
        }""")
        page.screenshot(path="docs/assets/screenshots/dashboard-overview.png")

        # 3. Live Scoring (Streamlit sidebar navigation)
        print("Capturing Live Scoring...")
        # Click on the 'Live Scoring' sidebar item
        page.click("text=Live Scoring")
        time.sleep(3)
        
        # We need to fill the form and submit. 
        # The form has a 'Score Transaction' button.
        page.wait_for_selector("text=Score Transaction")
        page.click("text=Score Transaction")
        
        # Wait for the result (e.g. "Risk Level:", "Probability:")
        time.sleep(8)
        page.screenshot(path="docs/assets/screenshots/live-scoring.png")

        # 4. Explainability (Reason Codes)
        print("Capturing Explanation...")
        try:
            page.click("text=Generate Explanation (IG)", timeout=5000)
            page.wait_for_selector("text=Feature Contributions", timeout=15000)
            time.sleep(3)
        except Exception as e:
            print(f"Explanation click failed: {e}")
        page.screenshot(path="docs/assets/screenshots/explanation-reason-codes.png")

        # 5. Case Queue (Case Management)
        print("Capturing Case Queue...")
        page.click("text=Case Queue")
        time.sleep(3)
        # Wait for cases to load
        page.wait_for_selector("text=Total Cases", timeout=15000)
        time.sleep(3)
        page.screenshot(path="docs/assets/screenshots/case-queue-history.png")

        # 6. Grafana Observability
        print("Capturing Grafana...")
        page.goto("http://127.0.0.1:3000/login")
        page.wait_for_selector("input[name='user']")
        page.fill("input[name='user']", "admin")
        page.fill("input[name='password']", "admin")
        page.click("button[type='submit']")
        
        # Wait for dashboard to load (assuming it redirects or we need to navigate)
        time.sleep(3)
        # Go to the first dashboard
        page.goto("http://127.0.0.1:3000/dashboards")
        time.sleep(3)
        # Try to click the FraudShield dashboard
        try:
            page.click("text=FraudShield")
            time.sleep(5)
        except Exception:
            pass # Just capture the current page if it fails
            
        page.screenshot(path="docs/assets/screenshots/grafana-observability.png")

        print("All screenshots captured successfully.")
        
    except Exception as e:
        print(f"Error during capture: {e}")
        raise e
    finally:
        browser.close()

with sync_playwright() as playwright:
    run(playwright)
