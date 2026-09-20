#!/usr/bin/env python3
"""Automated keep-alive and wake-up script for Streamlit Community Cloud.

Opens the target Streamlit application in a headless Chromium browser.
If the app is asleep, locates and clicks the 'Yes, get this app back up!' button
and waits for the application container to boot.
"""
from __future__ import annotations

import os
import sys
import time
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

DEFAULT_URL = "https://pvjjwzapucy4re6bakgfza.streamlit.app/"
TARGET_URL = os.environ.get("STREAMLIT_APP_URL", DEFAULT_URL).strip() or DEFAULT_URL


def main() -> int:
    print(f"[INFO] Target Streamlit URL: {TARGET_URL}")
    screenshot_path = "streamlit_status.png"

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
            ],
        )
        context = browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        page = context.new_page()

        print("[INFO] Navigating to page...")
        try:
            page.goto(TARGET_URL, wait_until="load", timeout=60000)
        except PlaywrightTimeoutError:
            print("[WARN] Initial page.goto timed out, checking current DOM state...")

        # Small pause to allow dynamic client JS / WebSocket initialization
        time.sleep(5)

        current_url = page.url
        print(f"[INFO] Current page URL: {current_url}")

        if "login" in current_url or "share.streamlit.io/-/auth" in current_url:
            print(
                "[NOTICE] Streamlit app redirected to login/auth. "
                "Ensure app visibility is set to 'Public' in Streamlit Cloud Settings -> Sharing "
                "if public access is intended."
            )

        page_content = page.content().lower()

        # Check for the sleeping modal or wake button
        wake_selectors = [
            'button:has-text("Yes, get this app back up!")',
            'button:has-text("get this app back up")',
            'button:has-text("Yes, wake it up")',
            'button:has-text("Wake")',
        ]

        wake_btn = None
        for sel in wake_selectors:
            try:
                candidate = page.locator(sel).first
                if candidate.is_visible(timeout=2000):
                    wake_btn = candidate
                    print(f"[INFO] Found wake button with selector: {sel}")
                    break
            except Exception:
                continue

        if wake_btn:
            print("[ACTION] App is sleeping! Clicking wake up button...")
            try:
                wake_btn.click()
                print("[ACTION] Button clicked. Waiting 25s for app container to boot...")
                time.sleep(25)
                # Check again
                page.wait_for_load_state("networkidle", timeout=30000)
            except Exception as e:
                print(f"[WARN] Error while waiting after click: {e}")
        elif "this app has gone to sleep" in page_content or "sleeping" in page_content:
            print("[WARN] Page indicates sleep but button selector didn't match directly. Searching generic buttons...")
            buttons = page.locator("button").all()
            for b in buttons:
                txt = b.inner_text().strip()
                if "back up" in txt.lower() or "wake" in txt.lower():
                    print(f"[ACTION] Clicking button with text: '{txt}'")
                    b.click()
                    time.sleep(25)
                    break
        else:
            print("[SUCCESS] App is active and awake (no sleep dialog detected).")

        # Capture status screenshot for GitHub Actions artifacts
        try:
            page.screenshot(path=screenshot_path, full_page=False)
            print(f"[INFO] Saved status screenshot to {screenshot_path}")
        except Exception as err:
            print(f"[WARN] Failed to capture screenshot: {err}")

        browser.close()

    print("[DONE] Keep-alive check completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
