"""Screenshots of the Streamlit demo for the README.

    streamlit run app/streamlit_app.py          # in another terminal
    python scripts/make_readme_screenshots.py [--url http://localhost:8501]

Uses Playwright with the locally installed Google Chrome (no browser
download): pip install playwright. Writes docs/images/app_*.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"
HIDE_CHROME = """
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }
"""

# page path, text that only appears once the page has finished computing, and
# the clip (CSS px) that keeps the main content but drops the sidebar.
PAGES = [
    ("wavepacket", "Simulated time", dict(x=350, y=100, width=1070, height=880)),
    ("greybody", "Total power (this field", dict(x=350, y=100, width=1070, height=640)),
    ("circuit", "Exact time evolution", dict(x=350, y=100, width=1070, height=880)),
    ("amplitude", "Advantage at", dict(x=350, y=100, width=1070, height=800)),
]


def settle(page, marker: str) -> None:
    page.get_by_text(marker).first.wait_for(timeout=180_000)
    page.wait_for_function(  # Streamlit shows its status widget while a script runs
        "() => !document.querySelector('[data-testid=\"stStatusWidget\"] button')",
        timeout=180_000)
    page.wait_for_timeout(2500)  # let Plotly finish its first draw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8501")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        for path, marker, clip in PAGES:
            page = browser.new_page(viewport={"width": 1440, "height": 1300},
                                    device_scale_factor=1.25, color_scheme="light")
            page.goto(f"{args.url}/{path}")
            page.add_style_tag(content=HIDE_CHROME)
            settle(page, marker)
            if path == "wavepacket":  # catch the packet just after it splits
                page.get_by_text("▶ Play").click()
                page.wait_for_timeout(2100)
                page.get_by_text("❚❚ Pause").click()
                page.wait_for_timeout(300)
            page.screenshot(path=OUT / f"app_{path}.png", clip=clip)
            print("wrote", f"app_{path}.png")
            page.close()
        browser.close()


if __name__ == "__main__":
    main()
