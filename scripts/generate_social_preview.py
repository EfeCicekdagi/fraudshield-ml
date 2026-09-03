import os
from playwright.sync_api import sync_playwright

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body {
            margin: 0;
            padding: 0;
            width: 1280px;
            height: 640px;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            color: white;
            display: flex;
            flex-direction: row;
            align-items: center;
            justify-content: space-between;
        }
        .text-content {
            padding-left: 80px;
            width: 500px;
        }
        h1 {
            font-size: 64px;
            margin: 0 0 20px 0;
            background: -webkit-linear-gradient(#60a5fa, #3b82f6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        h2 {
            font-size: 28px;
            font-weight: 400;
            color: #94a3b8;
            margin: 0;
            line-height: 1.4;
        }
        .image-container {
            width: 600px;
            height: 400px;
            margin-right: 60px;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            border: 1px solid #334155;
            background-image: url('file://{dashboard_path}');
            background-size: cover;
            background-position: top left;
        }
        .badge {
            margin-top: 40px;
            display: inline-block;
            padding: 8px 16px;
            background: rgba(59, 130, 246, 0.2);
            border: 1px solid #3b82f6;
            border-radius: 20px;
            font-size: 14px;
            color: #60a5fa;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="text-content">
        <h1>FraudShield ML</h1>
        <h2>Explainable Fraud Detection & Case Management</h2>
        <div class="badge">v1.0.0 Release</div>
    </div>
    <div class="image-container"></div>
</body>
</html>
"""

def generate():
    dashboard_img_path = os.path.abspath("docs/assets/screenshots/dashboard-overview.png")
    dashboard_img_path = dashboard_img_path.replace("\\", "/")
    
    html_content = HTML_TEMPLATE.replace("{dashboard_path}", dashboard_img_path)
    
    with open("social_preview_temp.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 640})
        page = context.new_page()
        
        # Open local HTML file
        page.goto(f"file://{os.path.abspath('social_preview_temp.html')}")
        
        # Take screenshot
        page.screenshot(path="docs/assets/social-preview.png")
        
        browser.close()
        
    os.remove("social_preview_temp.html")
    print("Social preview generated successfully.")

if __name__ == "__main__":
    generate()
