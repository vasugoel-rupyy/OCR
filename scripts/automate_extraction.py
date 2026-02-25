"""
Automation script: Scrapes 'aadhar card' image URLs using Playwright Chromium,
posts each URL to the local OCR API, and logs request/response to SQLite DB.
"""
import asyncio
import json
import sqlite3
from datetime import datetime
from urllib.parse import quote

import requests
from playwright.async_api import async_playwright

# ─── Configuration ───────────────────────────────────────────────────────────
SEARCH_QUERY = "aadhar card"
BATCH_SIZE = 100
API_BASE = "http://localhost:8000"
API_ENDPOINT = f"{API_BASE}/ocr/process_url"
DB_PATH = "ocr_logs.db"
DOC_TYPE = "auto"
# ─────────────────────────────────────────────────────────────────────────────


def ensure_db():
    """Create the logs table if it doesn't already exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            api_endpoint  TEXT     NOT NULL,
            request_url   TEXT     NOT NULL,
            doc_type      TEXT     NOT NULL,
            response_json TEXT     NOT NULL,
            timestamp     TEXT     NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_log(endpoint: str, request_url: str, doc_type: str, response: dict):
    """Insert one request/response record into the DB."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO logs (api_endpoint, request_url, doc_type, response_json, timestamp) VALUES (?,?,?,?,?)",
        (endpoint, request_url, doc_type, json.dumps(response), datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


async def fetch_image_urls(query: str, count: int) -> list[str]:
    """
    Open Bing Image Search in a headless Chromium window,
    click each thumbnail to reveal the high-res URL, and return up to `count` URLs.
    """
    urls: list[str] = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-setuid-sandbox"],
        )
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
        )
        page = await context.new_page()

        search_url = f"https://www.bing.com/images/search?q={quote(query)}&form=HDRSC2"
        print(f"  → Opening: {search_url}")
        await page.goto(search_url, wait_until="domcontentloaded", timeout=60_000)

        # Wait for at least one thumbnail
        await page.wait_for_selector("a.iusc", timeout=15_000)

        thumbnails = await page.query_selector_all("a.iusc")
        print(f"  → Found {len(thumbnails)} thumbnails on Bing")

        for thumb in thumbnails:
            if len(urls) >= count:
                break

            # Each <a.iusc> carries a JSON metadata blob in the `m` attribute
            m_attr = await thumb.get_attribute("m")
            if not m_attr:
                continue

            try:
                meta = json.loads(m_attr)
                img_url = meta.get("murl") or meta.get("turl")
                if img_url and img_url.startswith("http"):
                    urls.append(img_url)
                    print(f"  [{len(urls)}/{count}] {img_url[:80]}...")
            except json.JSONDecodeError:
                continue

        await browser.close()

    return urls


def call_api(image_url: str, doc_type: str) -> tuple[dict, float]:
    """POST to the OCR API and return (parsed_json, elapsed_ms)."""
    import time
    t0 = time.time()
    try:
        resp = requests.post(
            API_ENDPOINT,
            json={"image_url": image_url, "document_type": doc_type},
            timeout=60,
        )
        elapsed = (time.time() - t0) * 1000
        if resp.ok:
            return resp.json(), elapsed
        return {"error": f"HTTP {resp.status_code}", "detail": resp.text[:500]}, elapsed
    except Exception as exc:
        elapsed = (time.time() - t0) * 1000
        return {"error": str(exc)}, elapsed


async def main():
    ensure_db()

    print(f"\n[Step 1] Searching Bing Images for: '{SEARCH_QUERY}'")
    image_urls = await fetch_image_urls(SEARCH_QUERY, BATCH_SIZE)

    if not image_urls:
        print("  ✗ No image URLs found. Exiting.")
        return

    print(f"\n[Step 2] Processing {len(image_urls)} image(s) via OCR API…\n")
    for i, url in enumerate(image_urls, 1):
        print(f"  [{i}/{len(image_urls)}] → {url[:80]}...")
        result, elapsed_ms = call_api(url, DOC_TYPE)
        save_log("/ocr/process_url", url, DOC_TYPE, result)

        decision = result.get("decision", result.get("error", "unknown"))
        score    = result.get("confidence_score", 0)
        fields   = result.get("extracted_fields", {})

        print(f"           decision={decision}  score={score:.2f}  time={elapsed_ms:.0f}ms")

        # Print extracted fields if any were returned
        if fields:
            for field_name, field_val in fields.items():
                # field_val may be a dict with 'value' key or a plain string
                val = field_val.get("value", field_val) if isinstance(field_val, dict) else field_val
                print(f"           {field_name}: {val}")
        else:
            print("           (no fields extracted)")

    print(f"\n✓ Done! All logs saved to '{DB_PATH}'")
    # Quick summary query
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT id, doc_type, timestamp FROM logs ORDER BY id DESC LIMIT 10").fetchall()
    conn.close()
    print("\n── Last 10 DB rows ──────────────────────────────")
    for row in rows:
        print(f"  id={row[0]}  doc_type={row[1]}  timestamp={row[2]}")


if __name__ == "__main__":
    asyncio.run(main())
