from __future__ import annotations
from pathlib import Path
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak


def _fmt(v, digits=4):
    if v is None:
        return "N/A"
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return str(v)


def build_pdf_report(result: dict, model_info: dict, report_path: str | Path, overlay_path=None, burden_path=None):
    report_path = Path(report_path)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CenterTitle", parent=styles["Title"], alignment=TA_CENTER, spaceAfter=12))
    styles.add(ParagraphStyle(name="SectionHead", parent=styles["Heading2"], spaceBefore=10, spaceAfter=6))
    doc = SimpleDocTemplate(str(report_path), pagesize=letter, rightMargin=42, leftMargin=42, topMargin=42, bottomMargin=42)
    story = [
        Paragraph("BT-MRI DSS - Research Decision-Support Report", styles["CenterTitle"]),
        Paragraph("Brain Tumor MRI Decision-Support System", styles["Normal"]),
        Spacer(1, 8),
    ]

    meta = [
        ["Analysis ID", result.get("analysis_id", "N/A")],
        ["Case ID", result.get("case_id", "N/A")],
        ["Generated", result.get("timestamp", datetime.now().isoformat(timespec="seconds"))],
        ["Mode", result.get("analysis_mode", "Research Analysis")],
        ["Model", model_info.get("model_name", "Model 3 - 2D U-Net++")],
        ["Operating threshold", _fmt(result.get("threshold"), 2)],
        ["Best training epoch", str(model_info.get("best_epoch", "37"))],
    ]
    t = Table(meta, colWidths=[1.8*inch, 4.7*inch])
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(0,-1),colors.HexColor("#EAF0F8")),("GRID",(0,0),(-1,-1),0.25,colors.grey),("VALIGN",(0,0),(-1,-1),"TOP"),("FONTNAME",(0,0),(-1,-1),"Helvetica"),("FONTSIZE",(0,0),(-1,-1),9),("BOTTOMPADDING",(0,0),(-1,-1),5),("TOPPADDING",(0,0),(-1,-1),5)]))
    story += [t, Spacer(1, 10)]

    if overlay_path and Path(overlay_path).exists():
        story += [Paragraph("MRI Segmentation Visualization", styles["SectionHead"]), Image(str(overlay_path), width=6.5*inch, height=2.2*inch)]
    if burden_path and Path(burden_path).exists():
        story += [Paragraph("Axial Tumor-Burden Profile", styles["SectionHead"]), Image(str(burden_path), width=6.3*inch, height=3.2*inch)]

    story += [Paragraph("Imaging-Derived Measurements", styles["SectionHead"])]
    measures = [
        ["Predicted tumor burden (B)", _fmt(result.get("predicted_tumor_burden"))],
        ["Affected-slice ratio (A)", _fmt(result.get("affected_slice_ratio"))],
        ["Maximum slice tumor ratio (M)", _fmt(result.get("max_slice_tumor_ratio"))],
        ["Mean prediction confidence (C)", _fmt(result.get("mean_confidence_in_prediction"))],
        ["Uncertainty ratio (U)", _fmt(result.get("uncertainty_ratio"))],
    ]
    mt = Table(measures, colWidths=[3.8*inch, 2.7*inch])
    mt.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey),("BACKGROUND",(0,0),(0,-1),colors.HexColor("#F5F7FA")),("FONTSIZE",(0,0),(-1,-1),9),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]))
    story += [mt]

    story += [Paragraph("Severity and Recommendation", styles["SectionHead"])]
    if result.get("severity_score") is None:
        story += [Paragraph("Severity calibration has not yet been generated for the final U-Net++ model. Segmentation results are valid, but severity and recommendation output are intentionally withheld until calibration is completed.", styles["Normal"])]
    else:
        rows = [
            ["Severity score", f"{result.get('severity_score')}/5"],
            ["Severity label", result.get("severity_label", "")],
            ["Confidence", result.get("confidence_category", "")],
            ["Recommendation category", result.get("urgency_category", "")],
        ]
        rt = Table(rows, colWidths=[2.2*inch, 4.3*inch])
        rt.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey),("BACKGROUND",(0,0),(0,-1),colors.HexColor("#EAF0F8")),("FONTSIZE",(0,0),(-1,-1),9),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]))
        story += [rt, Spacer(1, 6), Paragraph("<b>Summary:</b> " + result.get("structured_summary", ""), styles["Normal"]), Spacer(1, 4), Paragraph("<b>Recommended next step:</b> " + result.get("recommended_next_step", ""), styles["Normal"]), Spacer(1, 4), Paragraph("<b>Safety modifiers:</b> " + result.get("safety_modifiers", ""), styles["Normal"])]

    eval_keys = [("case_dice","Dice"),("case_iou","IoU"),("case_precision","Precision"),("case_recall_sensitivity","Sensitivity"),("case_specificity","Specificity"),("case_mcc","MCC"),("case_volume_similarity","Volume similarity"),("case_relative_volume_error","RVE")]
    available = [(label, result.get(key)) for key,label in eval_keys if key in result]
    if available:
        story += [Paragraph("Evaluation Metrics (Ground Truth Supplied)", styles["SectionHead"])]
        et = Table([[a,_fmt(v)] for a,v in available], colWidths=[3.8*inch,2.7*inch])
        et.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey),("FONTSIZE",(0,0),(-1,-1),9)])); story += [et]

    story += [Spacer(1, 12), Paragraph("Research-use statement", styles["SectionHead"]), Paragraph(result.get("disclaimer", "This output is generated by a research prototype and is not a medical diagnosis. A qualified radiologist or clinical specialist should review the complete MRI study and clinical context."), styles["Normal"])]
    doc.build(story)
    return report_path
