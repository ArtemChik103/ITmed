"""Generate a standalone HTML presentation and export it to PDF via headless Chrome."""
from __future__ import annotations

import argparse
import base64
import csv
from dataclasses import dataclass
from html import escape
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from scripts.submission_common import collect_test_objects

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HTML_OUTPUT = REPO_ROOT / "deliverables" / "presentation.html"
DEFAULT_PDF_OUTPUT = REPO_ROOT / "deliverables" / "presentation.pdf"
DEFAULT_SUMMARY_CSV = REPO_ROOT / "deliverables" / "results_test_done" / "summary.csv"


@dataclass(slots=True)
class PresentationStats:
    object_count: int
    positive_count: int
    negative_count: int
    runtime_loaded_count: int
    keypoint_ready_count: int
    mean_confidence: float
    max_confidence: float


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate HTML presentation and export it to PDF.")
    parser.add_argument("--test-root", default="../test_done", help="Path to test_done")
    parser.add_argument(
        "--html-output",
        default=str(DEFAULT_HTML_OUTPUT.relative_to(REPO_ROOT)),
        help="Output HTML path",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_PDF_OUTPUT.relative_to(REPO_ROOT)),
        help="Output PDF path",
    )
    parser.add_argument(
        "--summary-csv",
        default=str(DEFAULT_SUMMARY_CSV.relative_to(REPO_ROOT)),
        help="Optional summary.csv path used for live counters",
    )
    parser.add_argument(
        "--skip-pdf",
        action="store_true",
        help="Generate only the HTML file without PDF export",
    )
    return parser.parse_args(argv)


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def _load_stats(*, test_root: Path, summary_csv: Path) -> PresentationStats:
    objects = collect_test_objects(test_root)
    object_count = len(objects)
    positive_count = 0
    negative_count = 0
    runtime_loaded_count = 0
    keypoint_ready_count = 0
    confidences: list[float] = []

    if summary_csv.exists():
        with summary_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                positive_count += int(row.get("class", "0") == "1")
                negative_count += int(row.get("class", "0") == "0")
                runtime_loaded_count += int(row.get("runtime_model_loaded", "0") == "1")
                keypoint_ready_count += int(row.get("keypoint_model_loaded", "0") == "1")
                try:
                    confidences.append(float(row.get("confidence", "0")))
                except ValueError:
                    continue
    else:
        negative_count = object_count

    mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    max_confidence = max(confidences) if confidences else 0.0
    return PresentationStats(
        object_count=object_count,
        positive_count=positive_count,
        negative_count=negative_count,
        runtime_loaded_count=runtime_loaded_count,
        keypoint_ready_count=keypoint_ready_count,
        mean_confidence=mean_confidence,
        max_confidence=max_confidence,
    )


def _slides(stats: PresentationStats) -> list[dict[str, object]]:
    return [
        {
            "eyebrow": "Финальная защита",
            "title": "ИТ+Мед 2026",
            "subtitle": "Classifier-first система анализа DICOM-снимков тазобедренных суставов",
            "layout": "hero",
            "lead": (
                "Сдача собрана как единый воспроизводимый пакет: чистый публичный репозиторий, "
                "HTML/PDF-презентация, обязательный `id,class` и текстовый архив результатов."
            ),
            "chips": ["FastAPI", "Streamlit", "DICOM", "Classifier-first", "Explainability"],
            "metrics": [
                ("Объектов в test_done", str(stats.object_count)),
                ("Патология", str(stats.positive_count)),
                ("Норма", str(stats.negative_count)),
            ],
        },
        {
            "eyebrow": "Задача",
            "title": "Что нужно было сдать",
            "subtitle": "Не исследовательскую свалку, а компактный экспертный пакет.",
            "layout": "split",
            "bullets": [
                "Публичный репозиторий с понятной точкой входа и без противоречивых материалов.",
                "Файл `predictions.csv` в строгом формате `id,class`.",
                "Архив `results_test_done.zip` с машиночитаемыми JSON/TXT-отчетами по каждому объекту.",
                "Слайды, которые можно показать за 5 минут без перегруза и псевдомедицинских обещаний.",
            ],
            "side_title": "Ключевое решение",
            "side_text": (
                "Архив результатов сделан текстовым, а не набором картинок. "
                "Так его легче проверить автоматически, открыть локально и показать экспертам."
            ),
        },
        {
            "eyebrow": "Архитектура",
            "title": "Classifier-first runtime",
            "subtitle": "Основной verdict дает модель, explainability подключается отдельно.",
            "layout": "architecture",
            "columns": [
                (
                    "1. DICOM intake",
                    "Backend принимает `.dcm`, валидирует структуру и извлекает метаданные без ручной подготовки файла.",
                ),
                (
                    "2. Plugin runtime",
                    "Плагин `hip_dysplasia` считает `confidence`, `threshold`, `class` и флаги загрузки runtime.",
                ),
                (
                    "3. Explainability",
                    "Keypoints включаются только в режиме `education` и не меняют итоговый бинарный verdict.",
                ),
            ],
            "footer_note": "Такой разрез проще защищать: диагноз, confidence и ограничения разделены явно.",
        },
        {
            "eyebrow": "Интерфейс",
            "title": "Два режима для двух сценариев",
            "subtitle": "Один экран для врача, другой для демонстрации и обучения.",
            "layout": "two-panel",
            "panels": [
                (
                    "Режим врача",
                    [
                        "краткая сводка по диагнозу",
                        "confidence и threshold",
                        "warnings и runtime status",
                        "без лишней визуальной нагрузки",
                    ],
                ),
                (
                    "Режим обучения",
                    [
                        "keypoint overlay",
                        "расширенный JSON/PDF-слой",
                        "объяснение статуса geometry",
                        "отдельный explainability-контур",
                    ],
                ),
            ],
        },
        {
            "eyebrow": "Ограничение",
            "title": "Почему geometry отключена",
            "subtitle": "Мы не подменяем клинический смысл неподтвержденными числами.",
            "layout": "quote",
            "quote": (
                "Количественная geometry автоматически не рассчитывается, "
                "потому что семантика raw MTDDH keypoints пока не валидирована для клинического использования."
            ),
            "bullets": [
                "Keypoints остаются в продукте как anatomy/explainability layer.",
                "UI, TXT, JSON и PDF пишут об ограничении честно и одинаково.",
                "Это снижает риск ложных clinical claims на защите.",
            ],
        },
        {
            "eyebrow": "Batch pipeline",
            "title": "Один запуск по всему test_done",
            "subtitle": "От объекта до готовых deliverables без ручной склейки.",
            "layout": "process",
            "steps": [
                "собрать все DICOM внутри объекта",
                "прогнать локальный runtime",
                "агрегировать объектный verdict",
                "сохранить `predictions.csv` и `summary.csv`",
                "сохранить `reports/{id}.json` и `reports/{id}.txt`",
                "упаковать `results_test_done.zip`",
            ],
            "metrics": [
                ("Runtime loaded", str(stats.runtime_loaded_count)),
                ("Keypoints ready", str(stats.keypoint_ready_count)),
                ("Средний confidence", f"{stats.mean_confidence:.3f}"),
                ("Максимальный confidence", f"{stats.max_confidence:.3f}"),
            ],
        },
        {
            "eyebrow": "Артефакты",
            "title": "Что получает эксперт",
            "subtitle": "Пакет собран так, чтобы его можно было проверить и глазами, и скриптом.",
            "layout": "deliverables",
            "cards": [
                ("Репозиторий", "Чистый `main`, актуальный README, запуск API/frontend и финальные команды."),
                ("presentation.html", "Человеческая версия слайдов: красиво открывается в браузере и умеет печататься в PDF."),
                ("presentation.pdf", "Экспортируется из того же HTML, поэтому визуально совпадает со слайдами."),
                ("results_test_done.zip", "Внутри `predictions.csv`, `summary.csv`, `README_results.txt` и per-object JSON/TXT."),
            ],
        },
        {
            "eyebrow": "Финал",
            "title": "Что важно на защите",
            "subtitle": "Не обещать лишнего и показывать воспроизводимость.",
            "layout": "final",
            "bullets": [
                "Classifier runtime выдает бинарный verdict для всех объектов `test_done`.",
                "Explainability сохранен, но не влияет на диагностическое решение.",
                "Все итоговые артефакты можно пересобрать одной командой из текущего репозитория.",
            ],
            "closing": "HTML-first презентация решает проблему качества рендера и дает нормальный PDF без артефактов шрифтов.",
        },
    ]


def _render_metrics(metrics: list[tuple[str, str]]) -> str:
    return "".join(
        (
            "<div class=\"metric-card\">"
            f"<span class=\"metric-value\">{escape(value)}</span>"
            f"<span class=\"metric-label\">{escape(label)}</span>"
            "</div>"
        )
        for label, value in metrics
    )


def _render_list(items: list[str], *, class_name: str) -> str:
    return "".join(f"<li class=\"{class_name}\">{escape(item)}</li>" for item in items)


def _render_slide(index: int, total: int, slide: dict[str, object]) -> str:
    layout = str(slide["layout"])
    eyebrow = escape(str(slide["eyebrow"]))
    title = escape(str(slide["title"]))
    subtitle = escape(str(slide["subtitle"]))

    common_intro = (
        f"<div class=\"slide-meta\"><span class=\"eyebrow\">{eyebrow}</span>"
        f"<span class=\"page-no\">{index:02d}/{total:02d}</span></div>"
        f"<h2 class=\"slide-title\">{title}</h2>"
        f"<p class=\"slide-subtitle\">{subtitle}</p>"
    )

    if layout == "hero":
        lead = escape(str(slide["lead"]))
        chips = "".join(
            f"<span class=\"chip\">{escape(str(chip))}</span>" for chip in slide["chips"]
        )
        metrics = _render_metrics(slide["metrics"])
        body = (
            "<div class=\"hero-layout\">"
            "<div class=\"hero-copy\">"
            f"{common_intro}<p class=\"hero-lead\">{lead}</p>"
            f"<div class=\"chip-row\">{chips}</div>"
            "</div>"
            "<div class=\"hero-aside\">"
            "<div class=\"radar-card\">"
            "<span class=\"radar-label\">Фокус защиты</span>"
            "<strong>Практический pipeline, честные ограничения, чистые артефакты.</strong>"
            "</div>"
            f"<div class=\"metrics-grid\">{metrics}</div>"
            "</div>"
            "</div>"
        )
    elif layout == "split":
        bullets = _render_list(slide["bullets"], class_name="bullet-item")
        side_title = escape(str(slide["side_title"]))
        side_text = escape(str(slide["side_text"]))
        body = (
            "<div class=\"split-layout\">"
            f"<div class=\"content-block\">{common_intro}<ul class=\"bullet-list\">{bullets}</ul></div>"
            "<aside class=\"side-note\">"
            f"<span class=\"note-kicker\">{side_title}</span>"
            f"<p>{side_text}</p>"
            "</aside>"
            "</div>"
        )
    elif layout == "architecture":
        columns = "".join(
            (
                "<article class=\"arch-card\">"
                f"<h3>{escape(str(title_text))}</h3>"
                f"<p>{escape(str(text))}</p>"
                "</article>"
            )
            for title_text, text in slide["columns"]
        )
        footer_note = escape(str(slide["footer_note"]))
        body = (
            f"{common_intro}"
            f"<div class=\"arch-grid\">{columns}</div>"
            f"<p class=\"footer-note\">{footer_note}</p>"
        )
    elif layout == "two-panel":
        panels = "".join(
            (
                "<article class=\"panel-card\">"
                f"<h3>{escape(str(panel_title))}</h3>"
                f"<ul class=\"panel-list\">{_render_list(items, class_name='panel-item')}</ul>"
                "</article>"
            )
            for panel_title, items in slide["panels"]
        )
        body = f"{common_intro}<div class=\"panel-grid\">{panels}</div>"
    elif layout == "quote":
        quote = escape(str(slide["quote"]))
        bullets = _render_list(slide["bullets"], class_name="bullet-item")
        body = (
            f"{common_intro}<blockquote class=\"hero-quote\">{quote}</blockquote>"
            f"<ul class=\"bullet-list compact\">{bullets}</ul>"
        )
    elif layout == "process":
        steps = "".join(
            (
                "<div class=\"step-card\">"
                f"<span class=\"step-index\">{step_index:02d}</span>"
                f"<p>{escape(str(step_text))}</p>"
                "</div>"
            )
            for step_index, step_text in enumerate(slide["steps"], start=1)
        )
        metrics = _render_metrics(slide["metrics"])
        body = (
            f"{common_intro}<div class=\"process-layout\">"
            f"<div class=\"steps-grid\">{steps}</div>"
            f"<div class=\"metrics-grid wide\">{metrics}</div>"
            "</div>"
        )
    elif layout == "deliverables":
        cards = "".join(
            (
                "<article class=\"deliverable-card\">"
                f"<h3>{escape(str(card_title))}</h3>"
                f"<p>{escape(str(card_text))}</p>"
                "</article>"
            )
            for card_title, card_text in slide["cards"]
        )
        body = f"{common_intro}<div class=\"deliverables-grid\">{cards}</div>"
    else:
        bullets = _render_list(slide["bullets"], class_name="bullet-item")
        closing = escape(str(slide["closing"]))
        body = (
            f"{common_intro}<ul class=\"bullet-list\">{bullets}</ul>"
            f"<p class=\"closing-note\">{closing}</p>"
        )

    return f"<section class=\"slide slide-{layout}\" id=\"slide-{index}\">{body}</section>"


def render_html(*, stats: PresentationStats) -> str:
    slides = _slides(stats)
    rendered_slides = "\n".join(
        _render_slide(index, len(slides), slide) for index, slide in enumerate(slides, start=1)
    )
    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ИТ+Мед 2026 · Финальная презентация</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=Unbounded:wght@400;500;700;800&display=swap&subset=cyrillic" rel="stylesheet">
  <style>
    :root {{
      --bg: #f4ede3;
      --surface: rgba(255, 250, 244, 0.88);
      --surface-strong: rgba(255, 245, 235, 0.98);
      --ink: #16212f;
      --muted: #5e6a77;
      --accent: #b4491d;
      --accent-deep: #7d2e10;
      --line: rgba(22, 33, 47, 0.12);
      --shadow: 0 30px 80px rgba(89, 58, 36, 0.14);
      --display: "Unbounded", "Trebuchet MS", sans-serif;
      --body: "IBM Plex Sans", "Segoe UI", sans-serif;
    }}

    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      font-family: var(--body);
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(180, 73, 29, 0.12), transparent 36%),
        radial-gradient(circle at 85% 15%, rgba(22, 33, 47, 0.08), transparent 28%),
        linear-gradient(180deg, #fbf6ef 0%, #f4ede3 100%);
      min-height: 100vh;
      overflow-x: hidden;
    }}

    body::before {{
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background-image:
        linear-gradient(rgba(255,255,255,0.12) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.12) 1px, transparent 1px);
      background-size: 28px 28px;
      mask-image: linear-gradient(180deg, rgba(0,0,0,0.28), transparent 85%);
      opacity: 0.3;
    }}

    .toolbar {{
      position: fixed;
      top: 20px;
      right: 20px;
      z-index: 20;
      display: flex;
      gap: 10px;
      align-items: center;
      padding: 10px 12px;
      border: 1px solid rgba(22, 33, 47, 0.08);
      border-radius: 999px;
      background: rgba(255, 249, 241, 0.86);
      backdrop-filter: blur(12px);
      box-shadow: 0 20px 35px rgba(89, 58, 36, 0.12);
    }}

    .toolbar button,
    .toolbar a {{
      border: 0;
      border-radius: 999px;
      padding: 10px 14px;
      font: 600 13px/1 var(--body);
      color: var(--ink);
      background: rgba(22, 33, 47, 0.06);
      text-decoration: none;
      cursor: pointer;
      transition: transform .2s ease, background .2s ease, color .2s ease;
    }}

    .toolbar button.primary {{
      background: var(--accent);
      color: #fff7f1;
    }}

    .toolbar button:hover,
    .toolbar a:hover {{
      transform: translateY(-1px);
      background: rgba(22, 33, 47, 0.12);
    }}

    .toolbar button.primary:hover {{ background: var(--accent-deep); }}

    .deck {{
      width: min(100%, 1560px);
      margin: 0 auto;
      padding: 88px 28px 72px;
      display: grid;
      gap: 28px;
    }}

    .slide {{
      position: relative;
      min-height: calc(100vh - 120px);
      padding: 42px 44px;
      border: 1px solid rgba(22, 33, 47, 0.08);
      border-radius: 34px;
      background:
        linear-gradient(180deg, rgba(255,255,255,0.70), rgba(255,255,255,0.88)),
        linear-gradient(135deg, rgba(228, 179, 147, 0.12), transparent 45%);
      box-shadow: var(--shadow);
      overflow: hidden;
      isolation: isolate;
    }}

    .slide::after {{
      content: "";
      position: absolute;
      width: 380px;
      height: 380px;
      right: -120px;
      top: -140px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(180, 73, 29, 0.12), rgba(180, 73, 29, 0));
      z-index: -1;
    }}

    .slide-meta {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 20px;
      margin-bottom: 18px;
      text-transform: uppercase;
      letter-spacing: 0.16em;
      font-size: 12px;
      font-weight: 700;
      color: var(--accent);
    }}

    .page-no {{ color: rgba(22, 33, 47, 0.36); }}

    .slide-title {{
      margin: 0;
      max-width: 15ch;
      font: 800 clamp(42px, 6vw, 86px)/1.04 var(--display);
      letter-spacing: -0.05em;
    }}

    .slide-subtitle,
    .hero-lead,
    .content-block p,
    .side-note p,
    .arch-card p,
    .deliverable-card p,
    .panel-card p,
    .step-card p,
    .footer-note,
    .closing-note {{
      margin: 0;
      color: var(--muted);
      font-size: clamp(18px, 2vw, 24px);
      line-height: 1.45;
    }}

    .slide-subtitle {{
      max-width: 44rem;
      margin-top: 12px;
    }}

    .hero-layout,
    .split-layout,
    .process-layout {{
      display: grid;
      gap: 28px;
      margin-top: 30px;
    }}

    .hero-layout {{
      grid-template-columns: 1.3fr .9fr;
      align-items: end;
    }}

    .hero-copy {{
      display: grid;
      gap: 20px;
    }}

    .hero-lead {{
      max-width: 42rem;
      font-size: clamp(21px, 2.4vw, 30px);
      color: var(--ink);
    }}

    .chip-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
    }}

    .chip {{
      display: inline-flex;
      align-items: center;
      padding: 10px 14px;
      border-radius: 999px;
      background: rgba(22, 33, 47, 0.06);
      color: var(--ink);
      font-size: 14px;
      font-weight: 600;
    }}

    .hero-aside {{
      display: grid;
      gap: 18px;
      align-self: stretch;
    }}

    .radar-card,
    .side-note,
    .arch-card,
    .panel-card,
    .step-card,
    .deliverable-card,
    .metric-card {{
      border: 1px solid var(--line);
      border-radius: 24px;
      background: var(--surface);
      padding: 22px;
      backdrop-filter: blur(10px);
    }}

    .radar-card {{
      min-height: 220px;
      display: grid;
      align-content: space-between;
      background:
        linear-gradient(180deg, rgba(180, 73, 29, 0.12), rgba(255, 255, 255, 0.36)),
        var(--surface);
    }}

    .radar-card strong {{
      font: 700 clamp(28px, 3vw, 40px)/1.12 var(--display);
      letter-spacing: -0.04em;
    }}

    .radar-label,
    .note-kicker {{
      font-size: 13px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.18em;
      color: var(--accent);
    }}

    .metrics-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }}

    .metrics-grid.wide {{ align-content: start; }}

    .metric-card {{
      min-height: 124px;
      display: grid;
      align-content: space-between;
      background: var(--surface-strong);
    }}

    .metric-value {{
      font: 700 clamp(32px, 3.1vw, 52px)/1 var(--display);
      letter-spacing: -0.05em;
    }}

    .metric-label {{
      color: var(--muted);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.14em;
    }}

    .split-layout {{
      grid-template-columns: 1.25fr .75fr;
      align-items: start;
    }}

    .content-block {{
      display: grid;
      gap: 26px;
    }}

    .bullet-list,
    .panel-list {{
      margin: 0;
      padding: 0;
      list-style: none;
      display: grid;
      gap: 14px;
    }}

    .bullet-item,
    .panel-item {{
      position: relative;
      padding-left: 30px;
      color: var(--ink);
      font-size: clamp(18px, 2.1vw, 28px);
      line-height: 1.35;
    }}

    .bullet-item::before,
    .panel-item::before {{
      content: "";
      position: absolute;
      left: 0;
      top: 0.62em;
      width: 12px;
      height: 12px;
      border-radius: 999px;
      background: var(--accent);
      box-shadow: 0 0 0 6px rgba(180, 73, 29, 0.12);
    }}

    .arch-grid,
    .panel-grid,
    .deliverables-grid {{
      display: grid;
      gap: 18px;
      margin-top: 32px;
    }}

    .arch-grid {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    .panel-grid,
    .deliverables-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}

    .arch-card h3,
    .panel-card h3,
    .deliverable-card h3 {{
      margin: 0 0 10px;
      font: 700 clamp(24px, 2.4vw, 34px)/1.12 var(--display);
      letter-spacing: -0.04em;
    }}

    .footer-note,
    .closing-note {{
      margin-top: 24px;
      max-width: 44rem;
    }}

    .hero-quote {{
      margin: 26px 0 0;
      padding: 24px 28px;
      border-left: 8px solid var(--accent);
      border-radius: 0 24px 24px 0;
      background: rgba(180, 73, 29, 0.08);
      color: var(--ink);
      font: 600 clamp(26px, 3vw, 42px)/1.28 var(--body);
    }}

    .bullet-list.compact {{ margin-top: 24px; gap: 12px; }}

    .process-layout {{
      grid-template-columns: 1.1fr .9fr;
      align-items: start;
    }}

    .steps-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      margin-top: 30px;
    }}

    .step-card {{
      min-height: 150px;
      display: grid;
      align-content: space-between;
    }}

    .step-index {{
      font: 800 18px/1 var(--display);
      letter-spacing: -0.04em;
      color: var(--accent);
    }}

    .step-card p {{
      color: var(--ink);
      font-size: clamp(18px, 1.7vw, 24px);
    }}

    .deliverable-card {{
      min-height: 220px;
      display: grid;
      align-content: start;
      gap: 10px;
    }}

    .slide-final .closing-note {{
      font-size: clamp(22px, 2.6vw, 34px);
      color: var(--accent-deep);
      max-width: 36rem;
    }}

    @media (max-width: 1100px) {{
      .hero-layout,
      .split-layout,
      .process-layout,
      .arch-grid,
      .panel-grid,
      .deliverables-grid,
      .steps-grid {{
        grid-template-columns: 1fr;
      }}

      .slide {{ min-height: auto; }}
      .metrics-grid {{ grid-template-columns: 1fr 1fr; }}
    }}

    @media (max-width: 720px) {{
      .deck {{ padding: 88px 16px 32px; }}
      .slide {{ padding: 26px 22px; border-radius: 24px; }}

      .toolbar {{
        right: 12px;
        left: 12px;
        justify-content: space-between;
        flex-wrap: wrap;
        border-radius: 24px;
      }}

      .metrics-grid {{ grid-template-columns: 1fr; }}
    }}

    @page {{
      size: 1600px 900px;
      margin: 0;
    }}

    @media print {{
      html, body {{ background: #fff; }}
      body::before, .toolbar {{ display: none !important; }}
      .deck {{ width: auto; margin: 0; padding: 0; gap: 0; }}

      .slide {{
        width: 1600px;
        min-height: 900px;
        height: 900px;
        border-radius: 0;
        border: 0;
        box-shadow: none;
        page-break-after: always;
        break-after: page;
      }}
    }}

    * {{
      -webkit-print-color-adjust: exact;
      print-color-adjust: exact;
    }}
  </style>
</head>
<body>
  <div class="toolbar" aria-label="Управление презентацией">
    <button type="button" data-action="prev">Назад</button>
    <button type="button" data-action="next">Вперед</button>
    <button type="button" class="primary" data-action="export">Экспорт в PDF</button>
    <a href="#slide-1">К началу</a>
  </div>

  <main class="deck">
    {rendered_slides}
  </main>

  <script>
    const slides = Array.from(document.querySelectorAll('.slide'));
    let activeIndex = 0;

    function scrollToSlide(index) {{
      const clamped = Math.max(0, Math.min(index, slides.length - 1));
      activeIndex = clamped;
      slides[clamped].scrollIntoView({{ behavior: 'smooth', block: 'start' }});
    }}

    const observer = new IntersectionObserver((entries) => {{
      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (!visible) return;
      activeIndex = slides.indexOf(visible.target);
    }}, {{ threshold: [0.4, 0.65, 0.9] }});

    slides.forEach((slide) => observer.observe(slide));

    document.querySelector('[data-action="prev"]').addEventListener('click', () => scrollToSlide(activeIndex - 1));
    document.querySelector('[data-action="next"]').addEventListener('click', () => scrollToSlide(activeIndex + 1));
    document.querySelector('[data-action="export"]').addEventListener('click', () => window.print());

    window.addEventListener('keydown', (event) => {{
      if (event.key === 'ArrowDown' || event.key === 'PageDown' || event.key === 'ArrowRight') {{
        event.preventDefault();
        scrollToSlide(activeIndex + 1);
      }}
      if (event.key === 'ArrowUp' || event.key === 'PageUp' || event.key === 'ArrowLeft') {{
        event.preventDefault();
        scrollToSlide(activeIndex - 1);
      }}
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'p') {{
        event.preventDefault();
        window.print();
      }}
    }});
  </script>
</body>
</html>
"""


def _write_html(path: Path, *, stats: PresentationStats) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(stats=stats), encoding="utf-8")


def _export_pdf(*, html_path: Path, pdf_path: Path) -> None:
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
        pdf_base64 = driver.execute_cdp_cmd(
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
        )["data"]
        pdf_path.write_bytes(base64.b64decode(pdf_base64))
    finally:
        driver.quit()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    test_root = _resolve_repo_path(args.test_root)
    html_output = _resolve_repo_path(args.html_output)
    pdf_output = _resolve_repo_path(args.output)
    summary_csv = _resolve_repo_path(args.summary_csv)

    stats = _load_stats(test_root=test_root, summary_csv=summary_csv)
    _write_html(html_output, stats=stats)
    print(f"Saved HTML presentation to {html_output}")

    if not args.skip_pdf:
        _export_pdf(html_path=html_output, pdf_path=pdf_output)
        print(f"Saved PDF presentation to {pdf_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
