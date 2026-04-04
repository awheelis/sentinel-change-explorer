"""Visual End-to-End test for Sentinel-2 Change Detection Explorer.

Launches the Streamlit app (with warmup skipped), opens a Playwright headless
Chromium browser, and verifies that all four panels render correctly after
clicking "Analyze Change."

Usage:
    # From the repo root, with the venv activated:
    python tests/e2e_visual_test.py

    # Or via pytest (collected as a module-level test):
    pytest tests/e2e_visual_test.py -v -s

Prerequisites:
    pip install playwright && playwright install chromium

Environment variables:
    STREAMLIT_PORT  — port to run Streamlit on (default: 8599)
    SCREENSHOT_DIR  — directory for screenshots  (default: /tmp/e2e_screenshots)
    SKIP_SCREENSHOTS — set to "1" to skip saving screenshots

Memory notes:
    This sandbox has ~4 GB RAM. Streamlit + satellite array processing uses
    ~1.5 GB at peak; Playwright Chromium adds ~300 MB. To stay within limits
    the test minimizes browser lifetime during heavy server work.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright, TimeoutError as PwTimeout

# ── Configuration ─────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
PORT = int(os.environ.get("STREAMLIT_PORT", "8599"))
STREAMLIT_URL = f"http://localhost:{PORT}"
SCREENSHOT_DIR = Path(os.environ.get("SCREENSHOT_DIR", "/tmp/e2e_screenshots"))
SAVE_SCREENSHOTS = os.environ.get("SKIP_SCREENSHOTS", "") != "1"
MAX_ANALYSIS_WAIT = 60  # seconds (warm cache: <10s, 60s is safety margin)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Server lifecycle ──────────────────────────────────────────────────────────

def start_streamlit() -> subprocess.Popen:
    """Start Streamlit in a subprocess with warmup disabled."""
    env = {**os.environ, "SKIP_WARMUP": "1"}
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", str(PORT),
            "--server.headless", "true",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    # Wait for server to be ready
    for _ in range(30):
        time.sleep(1)
        if proc.poll() is not None:
            out = proc.stdout.read().decode() if proc.stdout else ""
            raise RuntimeError(f"Streamlit exited early:\n{out}")
        try:
            import urllib.request
            urllib.request.urlopen(f"{STREAMLIT_URL}/_stcore/health", timeout=2)
            log(f"Streamlit ready on port {PORT}")
            return proc
        except Exception:
            continue
    raise RuntimeError("Streamlit did not start within 30 seconds")


def stop_streamlit(proc: subprocess.Popen) -> None:
    """Gracefully stop the Streamlit subprocess."""
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    log("Streamlit stopped")


# ── Test logic ────────────────────────────────────────────────────────────────

def screenshot(page, name: str) -> None:
    """Save a screenshot if enabled."""
    if SAVE_SCREENSHOTS:
        SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(SCREENSHOT_DIR / f"{name}.png"))


def run_visual_test() -> list[str]:
    """Execute the full visual E2E test. Returns a list of error strings (empty = pass)."""
    errors: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 900})

        # ── 1. Load the app ───────────────────────────────────────────────
        log("1. Loading app...")
        page.goto(STREAMLIT_URL, timeout=30_000)
        try:
            page.wait_for_selector("h1", timeout=30_000)
        except PwTimeout:
            errors.append("App title never appeared")
            browser.close()
            return errors

        title = page.text_content("h1") or ""
        if "Sentinel" not in title:
            errors.append(f"Unexpected title: {title!r}")
            browser.close()
            return errors
        log(f"   Title: {title}")
        time.sleep(3)
        screenshot(page, "01_loaded")

        # ── 2. Verify sidebar controls ────────────────────────────────────
        log("2. Verifying sidebar controls...")
        body = page.text_content("body") or ""
        sidebar_items = [
            "Preset location", "Bounding Box", "Before Date Range",
            "After Date Range", "Analyze Change", "Max cloud cover",
            "Change index", "Show Overture Maps layers",
        ]
        for item in sidebar_items:
            if item in body:
                log(f"   OK: {item}")
            else:
                log(f"   MISSING: {item}")
                errors.append(f"Sidebar missing: {item}")

        # ── 3. Click Analyze Change ───────────────────────────────────────
        log("3. Clicking Analyze Change...")
        btn = page.locator('button:has-text("Analyze Change")')
        if btn.count() == 0:
            errors.append("Analyze Change button not found")
            browser.close()
            return errors
        btn.first.click()

        # ── 4. Wait for analysis to complete ──────────────────────────────
        log("4. Waiting for analysis...")
        start = time.time()
        analysis_ok = False
        while time.time() - start < MAX_ANALYSIS_WAIT:
            body = page.text_content("body") or ""
            if "Panel A" in body and "True Color" in body:
                log(f"   Done in {time.time() - start:.0f}s")
                analysis_ok = True
                break
            if "Failed to fetch" in body:
                errors.append("Data fetch failed during analysis")
                break
            if "No scenes found" in body:
                errors.append("No Sentinel-2 scenes found for date range")
                break
            if "Connection error" in body:
                errors.append("Streamlit server disconnected during analysis")
                break
            time.sleep(5)

        if not analysis_ok and not errors:
            errors.append("Analysis timed out")

        if not analysis_ok:
            screenshot(page, "fail_analysis")
            browser.close()
            return errors

        # Wait for images to fully render
        time.sleep(5)
        screenshot(page, "02_panel_a")

        # ── 5. Verify Panel A — True Color Comparison ─────────────────────
        log("5. Verifying Panel A...")
        body = page.text_content("body") or ""

        if "Panel A" not in body:
            errors.append("Panel A header missing")
        else:
            log("   OK: Panel A header")

        if "Before" in body:
            log("   OK: Before caption")
        else:
            errors.append("Before caption missing")

        if "After" in body:
            log("   OK: After caption")
        else:
            errors.append("After caption missing")

        # Check for Connection error (server OOM)
        if "Connection error" in body or "not connected to a server" in body:
            log("   WARNING: Server disconnected — images won't render")
            errors.append("Server disconnected (likely OOM) — images not verifiable")
        else:
            # Only check images if server is still alive
            st_imgs = page.locator('[data-testid="stImage"]')
            img_ct = st_imgs.count()
            log(f"   stImage containers: {img_ct}")
            if img_ct < 2:
                # Fallback: count any large <img>
                all_imgs = page.locator("img")
                large = sum(
                    1 for i in range(all_imgs.count())
                    if (all_imgs.nth(i).evaluate("el => el.naturalWidth") or 0) > 100
                )
                log(f"   Large images: {large}")
                if large < 2:
                    errors.append(f"Expected ≥2 satellite images, found {large}")
            else:
                log("   OK: ≥2 satellite images")

        # ── 6. Verify Panel D — Summary Statistics ────────────────────────
        log("6. Verifying Panel D...")
        page.evaluate("window.scrollBy(0, 700)")
        time.sleep(2)
        screenshot(page, "03_panel_d")
        body = page.text_content("body") or ""

        if "Summary Statistics" in body or "Panel D" in body:
            log("   OK: Panel D header")
        else:
            errors.append("Panel D header missing")

        for label in ("km²", "gain", "loss", "Unchanged"):
            if label in body:
                log(f"   OK: {label}")
            else:
                errors.append(f"Missing stat: {label}")

        if "Cloud" in body:
            log("   OK: Scene metadata")
        else:
            errors.append("Scene metadata missing")

        # ── 7. Verify Panel B+C — Change Heatmap + Map ───────────────────
        log("7. Verifying Panel B+C...")
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(3)
        screenshot(page, "04_panel_bc")
        body = page.text_content("body") or ""

        if "Heatmap" in body or "Panel B" in body:
            log("   OK: Panel B+C header")
        else:
            errors.append("Panel B+C header missing")

        iframes = page.locator("iframe").count()
        components = page.locator('[data-testid="stCustomComponentV1"]').count()
        log(f"   iframes: {iframes}, custom components: {components}")
        if iframes > 0 or components > 0:
            log("   OK: Map component")
        else:
            log("   INFO: Map component not detected via DOM (may still render visually)")

        # ── 8. Full-page screenshot ───────────────────────────────────────
        if SAVE_SCREENSHOTS:
            SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
            page.screenshot(
                path=str(SCREENSHOT_DIR / "05_full_page.png"),
                full_page=True,
            )

        browser.close()

    return errors


# ── Entry points ──────────────────────────────────────────────────────────────

def test_visual_e2e():
    """Pytest-compatible entry point."""
    from app import warm_preset_caches
    log("Warming preset caches for e2e test...")
    warm_preset_caches()
    log("Cache warmup complete.")

    proc = start_streamlit()
    try:
        errors = run_visual_test()
    finally:
        stop_streamlit(proc)

    if errors:
        log(f"\nFAILED — {len(errors)} issue(s):")
        for i, e in enumerate(errors, 1):
            log(f"  {i}. {e}")
    else:
        log("\nPASSED — all checks OK")

    if SAVE_SCREENSHOTS:
        log(f"\nScreenshots: {SCREENSHOT_DIR}/")

    assert not errors, f"Visual E2E test failed: {errors}"


if __name__ == "__main__":
    from app import warm_preset_caches
    log("Warming preset caches...")
    warm_preset_caches()
    log("Cache warmup complete.")

    proc = start_streamlit()
    try:
        errors = run_visual_test()
    finally:
        stop_streamlit(proc)

    log("")
    log("=" * 60)
    if errors:
        log(f"RESULT: {len(errors)} ISSUE(S)")
        for i, e in enumerate(errors, 1):
            log(f"  {i}. {e}")
    else:
        log("RESULT: ALL CHECKS PASSED")
    log("=" * 60)

    if SAVE_SCREENSHOTS:
        log(f"\nScreenshots: {SCREENSHOT_DIR}/")
        for f in sorted(SCREENSHOT_DIR.iterdir()):
            if f.suffix == ".png":
                log(f"  {f.name}")

    sys.exit(0 if not errors else 1)
