"""One-shot: extract text from the two FITWIN patent docx files."""
import sys
import zipfile
import re
from pathlib import Path

FILES = [
    r"E:\[주식회사 피트윈_특허2]다중 센서 융합 기반 근골격계 데이터 분석 관련 특허 출원 자료_260315_유기웅.docx",
    r"E:\[주식회사 피트윈_특허1]다중 센서 융합 기반 쿼드 모달(Quad-Modal) 하드웨어 시스템 관련 특허 출원 자료_260315_유기웅.docx",
]

OUT = Path(r"C:\Users\FITWIN\Desktop\biomech-mocap\_tmp_patent")
OUT.mkdir(exist_ok=True)


def extract(p: Path) -> str:
    with zipfile.ZipFile(p, "r") as z:
        xml = z.read("word/document.xml").decode("utf-8")
    # Paragraph-preserving: insert newlines at paragraph boundaries
    xml = xml.replace("</w:p>", "</w:p>\n")
    text = re.sub(r"<[^>]+>", "", xml)
    text = text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


for idx, f in enumerate(FILES, 1):
    p = Path(f)
    if not p.exists():
        print(f"[miss] {p}")
        continue
    t = extract(p)
    out = OUT / f"patent_{idx}.txt"
    out.write_text(t, encoding="utf-8")
    print(f"[ok] patent_{idx} — {len(t)} chars — {out}")
