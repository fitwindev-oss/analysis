"""
Verify that Korean fonts resolve to the bundled TTFs (not system fallback).

Run after placing NanumGothic-Regular.ttf / NanumGothic-Bold.ttf into
``src/reports/resources/fonts/``. Prints which concrete font file each
renderer (matplotlib + reportlab) is actually using, generates a tiny
test PDF, and exits with non-zero if the bundled TTFs are not picked up.

Usage:
    python scripts/verify_fonts.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _check_files() -> tuple[list[Path], list[Path], bool]:
    """Return (regular candidates found, bold candidates found, all-ok).

    Accepts both the hyphenated Naver form (NanumGothic-Regular.ttf) and
    the legacy terse form (NanumGothic.ttf / NanumGothicBold.ttf).
    """
    base = Path(__file__).resolve().parents[1] / "src" / "reports" / "resources" / "fonts"
    reg_candidates = [base / "NanumGothic-Regular.ttf",
                      base / "NanumGothic.ttf"]
    bold_candidates = [base / "NanumGothic-Bold.ttf",
                       base / "NanumGothicBold.ttf"]
    reg_found  = [p for p in reg_candidates  if p.exists()]
    bold_found = [p for p in bold_candidates if p.exists()]
    both_ok = bool(reg_found) and bool(bold_found)
    return reg_found, bold_found, both_ok


def main() -> int:
    reg_found, bold_found, files_ok = _check_files()
    fonts_dir = (Path(__file__).resolve().parents[1]
                  / "src" / "reports" / "resources" / "fonts")
    print("── TTF file presence ──────────────────────────────────────────")
    if reg_found:
        print(f"  Regular : OK       {reg_found[0].name}")
    else:
        print(f"  Regular : MISSING  (expected NanumGothic-Regular.ttf or "
              f"NanumGothic.ttf in {fonts_dir})")
    if bold_found:
        print(f"  Bold    : OK       {bold_found[0].name}")
    else:
        print(f"  Bold    : MISSING  (expected NanumGothic-Bold.ttf or "
              f"NanumGothicBold.ttf in {fonts_dir})")
    if not files_ok:
        print()
        print("  ⚠ TTF files not bundled — PDF will fall back to Malgun "
              "Gothic (or Helvetica if unavailable).")
        print("    See src/reports/resources/fonts/README.md for download "
              "instructions.")
    print()

    # ── Trigger font resolution ─────────────────────────────────────
    from src.reports.fonts import setup_korean_fonts, _resolve_paths

    state = setup_korean_fonts()
    resolved_reg, resolved_bold = _resolve_paths()

    print("── Resolved paths (first existing wins) ───────────────────────")
    print(f"  Regular → {resolved_reg}")
    print(f"  Bold    → {resolved_bold}")
    print()

    # "Using bundle" iff the resolved path lives inside the fonts dir.
    def _in_bundle(p):
        if p is None:
            return False
        try:
            return fonts_dir.resolve() in p.resolve().parents
        except Exception:
            return False

    using_bundle_reg  = _in_bundle(resolved_reg)
    using_bundle_bold = _in_bundle(resolved_bold)

    print("── Rendering engine state ─────────────────────────────────────")
    print(f"  matplotlib family: {state.get('matplotlib_family')}")
    print(f"  reportlab family : {state.get('pdf_family')}")
    print()

    print("── Source verdict ────────────────────────────────────────────")
    if using_bundle_reg and using_bundle_bold:
        print("  ✅ Bundled NanumGothic — deterministic rendering across "
              "machines.")
    elif resolved_reg is not None:
        print("  🟡 Using system font fallback — works on this PC, may "
              "render differently elsewhere.")
    else:
        print("  ❌ No Korean font resolved — PDF will show boxes for "
              "Korean glyphs.")
    print()

    # ── Quick PDF test ──────────────────────────────────────────────
    try:
        from reportlab.platypus import SimpleDocTemplate, Paragraph
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.pagesizes import A4
        from src.reports.fonts import pdf_font_family
        family = pdf_font_family()

        tmp = Path(tempfile.mktemp(suffix=".pdf", prefix="font_test_"))
        doc = SimpleDocTemplate(str(tmp), pagesize=A4)
        style = ParagraphStyle("t", fontName=family, fontSize=14, leading=20)
        doc.build([
            Paragraph("한글 렌더링 테스트", style),
            Paragraph("가나다라마바사 아자차카타파하", style),
            Paragraph("FITWIN MoCap Clinic — 바이오메카닉스 리포트", style),
        ])
        size = tmp.stat().st_size
        print(f"── PDF smoke test ─────────────────────────────────────────────")
        print(f"  Generated : {tmp}")
        print(f"  Size      : {size} bytes")
        print("  (열어서 한글이 □ 박스로 보이면 폴백 실패입니다.)")
    except Exception as e:
        print(f"  ⚠ PDF test failed: {e}")
        return 2

    # Non-zero exit if bundle missing (for CI / automated checks)
    return 0 if (using_bundle_reg and using_bundle_bold) else 1


if __name__ == "__main__":
    sys.exit(main())
