"""Package the LaTeX slides as a PowerPoint slideshow without changing layout."""
from pathlib import Path
import argparse

from pptx import Presentation
from pptx.util import Inches

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("rendered_pages", type=Path, help="PNG pages rendered from main.pdf")
args = parser.parse_args()
pages = sorted(args.rendered_pages.glob("page-*.png"))
assert len(pages) == 9, "Expected the nine reviewed slides."

deck = Presentation()
deck.slide_width, deck.slide_height = Inches(16), Inches(9)
deck.core_properties.title = "模型融合的后处理防御"
deck.core_properties.subject = "两种方案与攻击评估"
for page in pages:
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    slide.shapes.add_picture(str(page), 0, 0, width=deck.slide_width, height=deck.slide_height)
    slide.notes_slide.notes_text_frame.text = "LaTeX-generated slide. Edit presentation/main.tex and regenerate."
output = Path(__file__).with_name("research_report.pptx")
deck.save(output)
print(output)

