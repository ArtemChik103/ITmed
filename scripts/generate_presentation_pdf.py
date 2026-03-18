"""Generate a concise defense PDF for the final submission package."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reportlab.lib.colors import HexColor, white
from reportlab.lib.pagesizes import landscape
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

from scripts.submission_common import collect_test_objects

REPO_ROOT = Path(__file__).resolve().parents[1]
SLIDE_SIZE = landscape((1280, 720))
MARGIN_X = 72
MARGIN_Y = 56
TITLE_COLOR = HexColor("#132238")
ACCENT = HexColor("#C44900")
TEXT_COLOR = HexColor("#243447")
MUTED = HexColor("#5D6B7A")
CARD = HexColor("#F5EFE6")
BACKGROUND = HexColor("#FFF9F2")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deliverables/presentation.pdf.")
    parser.add_argument("--test-root", default="../test_done", help="Path to test_done")
    parser.add_argument(
        "--output",
        default="deliverables/presentation.pdf",
        help="Output PDF path",
    )
    return parser.parse_args(argv)


def _draw_background(pdf: canvas.Canvas, *, title: str, subtitle: str, page_width: float, page_height: float) -> None:
    pdf.setFillColor(BACKGROUND)
    pdf.rect(0, 0, page_width, page_height, stroke=0, fill=1)
    pdf.setFillColor(CARD)
    pdf.roundRect(40, 38, page_width - 80, page_height - 76, 24, stroke=0, fill=1)
    pdf.setFillColor(ACCENT)
    pdf.roundRect(40, page_height - 122, page_width - 80, 56, 24, stroke=0, fill=1)
    pdf.setFillColor(white)
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(MARGIN_X, page_height - 100, "ИТ+Мед 2026 · Финальная защита")
    pdf.setFillColor(TITLE_COLOR)
    pdf.setFont("Helvetica-Bold", 28)
    pdf.drawString(MARGIN_X, page_height - 170, title)
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica", 16)
    pdf.drawString(MARGIN_X, page_height - 198, subtitle)


def _draw_bullets(
    pdf: canvas.Canvas,
    *,
    items: list[str],
    x: float,
    y: float,
    width: float,
    font_name: str = "Helvetica",
    font_size: int = 18,
    line_gap: int = 12,
) -> None:
    current_y = y
    bullet_indent = 20
    wrap_width = width - bullet_indent - 10
    for item in items:
        words = item.split()
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if stringWidth(candidate, font_name, font_size) <= wrap_width or not current:
                current = candidate
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)

        pdf.setFillColor(ACCENT)
        pdf.circle(x + 5, current_y + 5, 3, stroke=0, fill=1)
        pdf.setFillColor(TEXT_COLOR)
        pdf.setFont(font_name, font_size)
        for index, line in enumerate(lines):
            line_x = x + bullet_indent
            pdf.drawString(line_x, current_y - index * (font_size + 4), line)
        current_y -= len(lines) * (font_size + 4) + line_gap


def _draw_footer(pdf: canvas.Canvas, *, page_index: int, page_width: float) -> None:
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica", 12)
    pdf.drawRightString(page_width - MARGIN_X, 28, f"Слайд {page_index}")


def _slides(object_count: int) -> list[tuple[str, str, list[str]]]:
    return [
        (
            "Задача и формат сдачи",
            "Нужно отдать чистый публичный репозиторий и три проверяемых артефакта.",
            [
                "Целевой сценарий: анализ DICOM-снимков тазобедренных суставов с бинарным verdict 0/1.",
                "Финальный пакет: репозиторий, presentation.pdf, predictions.csv, results_test_done.zip.",
                "Архив результатов сделан текстовым и машиночитаемым, без зависимости от jpg-рендеринга.",
            ],
        ),
        (
            "Архитектура решения",
            "Classifier-first runtime с отдельным explainability-слоем для обучения и демонстрации.",
            [
                "FastAPI backend принимает DICOM, валидирует метаданные и запускает plugin runtime.",
                "Плагин hip_dysplasia сначала вычисляет бинарный classifier verdict и confidence.",
                "Keypoints подключаются отдельно и не меняют итоговый диагноз, а только помогают объяснению.",
            ],
        ),
        (
            "Runtime и режимы",
            "Система работает и с обученными весами, и в fallback-режиме.",
            [
                "Режим врача показывает краткую клиническую сводку: verdict, confidence, threshold, warnings.",
                "Режим обучения добавляет anatomy overlay и подробный JSON/PDF-отчет.",
                "Если веса недоступны, pipeline остается рабочим, но отчет честно помечается как non-diagnostic.",
            ],
        ),
        (
            "Почему geometry отключена",
            "Ограничение зафиксировано явно, чтобы не подменять анализ псевдоклиническими числами.",
            [
                "В raw MTDDH keypoints нет подтвержденной клинической семантики для автоматического расчета углов и расстояний.",
                "Поэтому keypoints используются только как explainability layer, а quantitative geometry не публикуется автоматически.",
                "README, UI, PDF и TXT-отчеты явно сообщают об этом ограничении.",
            ],
        ),
        (
            "Batch pipeline по test_done",
            "Один скрипт проходит по всем объектам test_done и собирает сразу все нужные deliverables.",
            [
                f"В текущем наборе test_done обнаружено {object_count} объектов.",
                "Для каждого объекта обрабатываются все DICOM, считается object-level probability и финальный class.",
                "Скрипт сохраняет predictions.csv, summary.csv, reports/{id}.json, reports/{id}.txt и ZIP-архив.",
            ],
        ),
        (
            "Итоговые артефакты",
            "Эксперт получает и человекочитаемый, и машиночитаемый слой результатов.",
            [
                "deliverables/predictions.csv: обязательный файл id,class без пропусков и дублей.",
                "deliverables/results_test_done.zip: summary.csv плюс подробные JSON/TXT-отчеты по каждому объекту.",
                "deliverables/presentation.pdf: короткая защита на 5 минут без research-шума.",
            ],
        ),
        (
            "Что смотреть эксперту",
            "Входная точка для проверки собрана в README и повторяется в deliverables.",
            [
                "README описывает запуск, режимы работы, ограничения и команды для пересборки выгрузки.",
                "ZIP позволяет быстро открыть summary.csv и затем провалиться в конкретный JSON/TXT по нужному id.",
                "Публичный репозиторий очищается до файлов, которые нужны для запуска, проверки и защиты.",
            ],
        ),
        (
            "Вывод",
            "Сдача собрана как воспроизводимый пакет, а не как набор вручную подготовленных файлов.",
            [
                "Classifier runtime выдает бинарный verdict для всех объектов test_done.",
                "Explainability-слой сохраняется, но не подменяет собой диагностику и не искажает clinical scope.",
                "Все финальные артефакты можно пересобрать локально из одного репозитория и одного batch-сценария.",
            ],
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    test_root = Path(args.test_root).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    object_count = len(collect_test_objects(test_root))
    page_width, page_height = SLIDE_SIZE
    pdf = canvas.Canvas(str(output_path), pagesize=SLIDE_SIZE)

    for index, (title, subtitle, bullets) in enumerate(_slides(object_count), start=1):
        _draw_background(
            pdf,
            title=title,
            subtitle=subtitle,
            page_width=page_width,
            page_height=page_height,
        )
        _draw_bullets(
            pdf,
            items=bullets,
            x=MARGIN_X,
            y=page_height - 260,
            width=page_width - 2 * MARGIN_X,
        )
        _draw_footer(pdf, page_index=index, page_width=page_width)
        pdf.showPage()

    pdf.save()
    print(f"Saved PDF presentation to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
