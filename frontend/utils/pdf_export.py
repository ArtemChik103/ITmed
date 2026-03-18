"""PDF report generation utilities."""
from __future__ import annotations

import io
from datetime import datetime
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from frontend.utils.report_formatting import (
    disease_label,
    model_probability,
    model_threshold,
    runtime_status_text,
    metadata_summary,
    compact_metrics,
)
from frontend.utils.clinical_report_builder import geometry_metric_rows, geometry_reason
from frontend.utils.medical_text import get_pdf_report_text

def _register_fonts():
    try:
        pdfmetrics.registerFont(TTFont('Arial', 'arial.ttf'))
        pdfmetrics.registerFont(TTFont('Arial-Bold', 'arialbd.ttf'))
        return 'Arial', 'Arial-Bold'
    except Exception:
        return 'Helvetica', 'Helvetica-Bold'

def generate_pdf_report(result: dict[str, Any], filename: str) -> bytes:
    font_regular, font_bold = _register_fonts()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )
    
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='PremiumTitle',
        parent=styles['Heading1'],
        fontName=font_bold,
        fontSize=24,
        spaceAfter=20,
        textColor=colors.HexColor('#1e293b')
    ))
    styles.add(ParagraphStyle(
        name='PremiumHeading',
        parent=styles['Heading2'],
        fontName=font_bold,
        fontSize=16,
        spaceBefore=15,
        spaceAfter=10,
        textColor=colors.HexColor('#334155')
    ))
    styles.add(ParagraphStyle(
        name='PremiumBody',
        parent=styles['Normal'],
        fontName=font_regular,
        fontSize=11,
        spaceAfter=8,
        textColor=colors.HexColor('#475569')
    ))
    
    elements = []
    
    # Header
    elements.append(Paragraph("Медицинское заключение (ИТ+Мед 2026)", styles['PremiumTitle']))
    elements.append(Paragraph(f"Дата формирования: {datetime.now().strftime('%d.%m.%Y %H:%M')}", styles['PremiumBody']))
    elements.append(Spacer(1, 20))
    
    # Patient / File Info
    elements.append(Paragraph("Информация об исследовании", styles['PremiumHeading']))
    elements.append(Paragraph(f"Файл: <b>{filename}</b>", styles['PremiumBody']))
    
    meta_items = metadata_summary(result)
    for label, val in meta_items:
        elements.append(Paragraph(f"{label}: <b>{val}</b>", styles['PremiumBody']))
        
    elements.append(Spacer(1, 15))
    
    # Results
    elements.append(Paragraph("Результаты анализа", styles['PremiumHeading']))
    
    diag_label = disease_label(result)
    diag_color = colors.HexColor('#ef4444') if result.get("disease_detected") else colors.HexColor('#10b981')
    
    # We create a custom paragraph style for the diagnosis
    styles.add(ParagraphStyle(
        name='DiagnosisStyle',
        parent=styles['Normal'],
        fontName=font_bold,
        fontSize=14,
        textColor=diag_color,
        spaceAfter=10
    ))
    
    elements.append(Paragraph(f"Диагноз: {diag_label}", styles['DiagnosisStyle']))
    
    conf_text = f"{model_probability(result) * 100:.1f}%"
    threshold = model_threshold(result)
    threshold_text = f"{threshold * 100:.1f}%" if threshold is not None else "не указан"
    
    elements.append(Paragraph(f"Уверенность нейросети: <b>{conf_text}</b>", styles['PremiumBody']))
    elements.append(Paragraph(f"Порог классификации: <b>{threshold_text}</b>", styles['PremiumBody']))
    elements.append(Paragraph(f"Статус системы: <i>{runtime_status_text(result)}</i>", styles['PremiumBody']))
    
    elements.append(Spacer(1, 15))
    
    # Detailed Report
    elements.append(Paragraph("Детальное описание", styles['PremiumHeading']))
    elements.append(Paragraph(get_pdf_report_text(result), styles['PremiumBody']))
    
    elements.append(Spacer(1, 15))

    elements.append(Paragraph("Геометрические показатели", styles['PremiumHeading']))
    elements.append(Paragraph(geometry_reason(result), styles['PremiumBody']))
    geometry_data = [["Показатель", "Значение"]]
    for label, value in geometry_metric_rows(result):
        geometry_data.append([label, value])

    geometry_table = Table(geometry_data, colWidths=[240, 160])
    geometry_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#0f172a')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,0), font_bold),
        ('FONTSIZE', (0,0), (-1,0), 12),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.white),
        ('TEXTCOLOR', (0,1), (-1,-1), colors.HexColor('#334155')),
        ('FONTNAME', (0,1), (-1,-1), font_regular),
        ('FONTSIZE', (0,1), (-1,-1), 10),
        ('GRID', (0,0), (-1,-1), 1, colors.HexColor('#e2e8f0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')])
    ]))
    elements.append(geometry_table)

    elements.append(Spacer(1, 15))
    
    # Metrics Table
    elements.append(Paragraph("Подробные метрики", styles['PremiumHeading']))
    metrics_data = [["Метрика", "Значение"]]
    for label, val in compact_metrics(result):
        metrics_data.append([label, str(val)])
        
    if metrics_data:
        t = Table(metrics_data, colWidths=[200, 200])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#0f172a')),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('FONTNAME', (0,0), (-1,0), font_bold),
            ('FONTSIZE', (0,0), (-1,0), 12),
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
            ('BACKGROUND', (0,1), (-1,-1), colors.white),
            ('TEXTCOLOR', (0,1), (-1,-1), colors.HexColor('#334155')),
            ('FONTNAME', (0,1), (-1,-1), font_regular),
            ('FONTSIZE', (0,1), (-1,-1), 10),
            ('GRID', (0,0), (-1,-1), 1, colors.HexColor('#e2e8f0')),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')])
        ]))
        elements.append(t)
        
    elements.append(Spacer(1, 40))
    elements.append(Paragraph("Внимание: Данный отчет сформирован автоматически алгоритмом ИИ и не является окончательным медицинским диагнозом. Требуется верификация врачом-специалистом.", styles['PremiumBody']))
    
    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
