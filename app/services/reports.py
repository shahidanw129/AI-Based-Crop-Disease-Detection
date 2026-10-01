from io import BytesIO

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def detection_pdf(detection):
    output = BytesIO()
    page = canvas.Canvas(output, pagesize=letter)
    page.setTitle(f"Crop scan report {detection.id}")
    page.setFont("Helvetica-Bold", 18)
    page.drawString(54, 742, "Fieldnote AI | Crop scan report")
    page.setFont("Helvetica", 11)
    rows = [
        ("Report ID", str(detection.id)),
        ("User", detection.user.name),
        ("Crop", detection.disease.crop_name if detection.disease else "Not mapped"),
        ("Prediction", detection.predicted_label.replace("_", " ").replace("|", " / ")),
        ("Model confidence", f"{detection.confidence * 100:.1f}%"),
        ("Date (UTC)", detection.created_at.strftime("%Y-%m-%d %H:%M UTC")),
    ]
    y = 690
    for label, value in rows:
        page.setFont("Helvetica-Bold", 11)
        page.drawString(54, y, f"{label}:")
        page.setFont("Helvetica", 11)
        page.drawString(170, y, value[:75])
        y -= 26
    page.setFont("Helvetica-Bold", 11)
    page.drawString(54, y - 8, "Important")
    page.setFont("Helvetica", 10)
    page.drawString(54, y - 28, "This AI output is a screening prediction, not a confirmed diagnosis.")
    page.drawString(54, y - 44, "Consult a qualified agriculture expert before making crop-treatment decisions.")
    page.save()
    output.seek(0)
    return output