"""Export a static HTML presentation to PDF via headless Chrome."""
from __future__ import annotations

import argparse
import base64
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "deliverables" / "presentation.html"
DEFAULT_OUTPUT = REPO_ROOT / "deliverables" / "presentation.pdf"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export presentation.html to presentation.pdf.")
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT.relative_to(REPO_ROOT)),
        help="Source HTML presentation path",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT.relative_to(REPO_ROOT)),
        help="Output PDF path",
    )
    return parser.parse_args(argv)


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def export_html_to_pdf(*, html_path: Path, pdf_path: Path) -> None:
    if not html_path.exists():
        raise FileNotFoundError(f"Presentation HTML not found: {html_path}")

    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1600,900")

    driver = webdriver.Chrome(options=options)
    try:
        driver.get(html_path.resolve().as_uri())
        driver.execute_async_script(
            """
            const done = arguments[0];
            Promise.all([
              document.fonts ? document.fonts.ready : Promise.resolve(),
              new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)))
            ]).then(() => done()).catch(() => done());
            """
        )
        payload = driver.execute_cdp_cmd(
            "Page.printToPDF",
            {
                "printBackground": True,
                "preferCSSPageSize": True,
                "landscape": False,
                "marginTop": 0,
                "marginBottom": 0,
                "marginLeft": 0,
                "marginRight": 0,
            },
        )
        pdf_path.write_bytes(base64.b64decode(payload["data"]))
    finally:
        driver.quit()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = _resolve_repo_path(args.input)
    output_path = _resolve_repo_path(args.output)
    export_html_to_pdf(html_path=input_path, pdf_path=output_path)
    print(f"Saved PDF presentation to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
