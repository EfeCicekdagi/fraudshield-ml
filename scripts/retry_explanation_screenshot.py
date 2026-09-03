import time
from playwright.sync_api import sync_playwright

def run(playwright):
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context(viewport={"width": 1440, "height": 1000})
    page = context.new_page()

    try:
        print("Capturing Explanation...")
        page.goto("http://127.0.0.1:8501")
        time.sleep(3)
        page.click("text=Live Scoring")
        time.sleep(3)
        
        page.wait_for_selector("text=Score Transaction")
        page.click("text=Score Transaction")
        
        # Wait for button to appear
        time.sleep(8)
        page.click("text=Generate Explanation (IG)", timeout=15000)
        
        # Wait for the IG chart
        page.wait_for_selector("text=Feature Contributions", timeout=15000)
        time.sleep(3)
        page.screenshot(path="docs/assets/screenshots/explanation-reason-codes.png")
        print("Explanation captured.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        browser.close()

with sync_playwright() as playwright:
    run(playwright)
