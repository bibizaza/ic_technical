"""Put the Technical Update on market.herculis.ai from the intern PC (Wednesday run, SKILL.md step 8).

    python tools/market_publish.py pdf <deck.pptx>            -> <deck>.pdf next to it, made by PowerPoint
    python tools/market_publish.py check <deck.pdf>           -> every number of the summary and notes is in the PDF?
    python tools/market_publish.py publish <deck.pdf>         -> check, then upload the PDF, summary and notes
    python tools/market_publish.py selftest [deck.pptx]       -> PowerPoint, PDF reader, OCR, login: OK or FAIL
                                                                 (default deck: template.pptx in IC_DROPBOX_PATH)

The summary and notes files sit next to the PDF: <deck>.summary.md (Committee page text) and <deck>.notes.txt
(one line per slide, "s5 · title · message", read by Ask Market). Login: .env MARKET_PUBLISHER_EMAIL and
MARKET_PUBLISHER_PASSWORD, an account that may only add and correct committee items. Same check as
market-herculis/web-codex/scripts/committee-publish.mjs (the Mac version): a number must be in the PDF's text or
in what OCR reads on its pictures (most of this deck's figures are images). Needs: pip install pymupdf.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

SUPABASE = "https://lipzcqyogckyvkpjywni.supabase.co"
PUBLISHABLE_KEY = "sb_publishable_DZfAmwSdzGQ5fRydpA19Gg_Hy243Xqf"  # public by design (the website ships it)
TOOLS = Path(__file__).resolve().parent
NOTE = re.compile(r"^s(\d+)\s*·\s*(.+)$")
NUMBER = re.compile(r"\d[\d,']*(?:\.\d+)?")
TOPICS = {"Technical Update": "Equities,Rates,Commodities,Currencies,Crypto",
          "Macro Update": "Inflation in the US,Rates,Equities,Energy"}


# ---------------------------------------------------------------- the check
def numbers(text: str) -> list[str]:
    found = (m.replace(",", "").replace("'", "") for m in NUMBER.findall(text))
    return [x for x in found if re.fullmatch(r"\d+(\.\d+)?", x)]


def unsupported(text: str, data: str) -> list[str]:
    """Numbers of `text` (brackets removed) found nowhere in `data` at their own precision (rounding allowed)."""
    have = sorted(float(x) for x in numbers(data))
    out = []
    for a in dict.fromkeys(numbers(re.sub(r"\[[^\]]*\]", "", text))):
        tol = 0.5 * 10 ** -(len(a.split(".")[1]) if "." in a else 0) + 1e-9
        i = bisect.bisect_left(have, float(a) - tol)
        if not (i < len(have) and have[i] <= float(a) + tol):
            out.append(a)
    return out


def check_notes(pages: list[str], notes: str, summary: str) -> list[str]:
    errors, seen, text = [], set(), "\n".join(pages)
    for line in (l.strip() for l in notes.splitlines() if l.strip()):
        m = NOTE.match(line)
        if not m:
            errors.append(f'not "s<page> · text": {line[:60]}')
            continue
        p = int(m.group(1))
        if not 1 <= p <= len(pages):
            errors.append(f"s{p}: the deck has {len(pages)} pages")
        if p in seen:
            errors.append(f"s{p}: two lines")
        seen.add(p)
        if len(line) > 400:
            errors.append(f"s{p}: {len(line)} characters (keep it under 400)")
        bad = unsupported(m.group(2), text)
        if bad:
            errors.append(f"s{p}: not in the deck: {' '.join(bad)}")
    bad = unsupported(summary, text)
    if bad:
        errors.append("summary: not in the deck: " + " ".join(bad))
    return errors


# ---------------------------------------------------------------- PDF, text, OCR
def powershell(script: str, *args: str) -> str:
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(TOOLS / script), *args],
                       capture_output=True, timeout=900)
    if r.returncode:
        raise RuntimeError(f"{script} failed: {r.stderr.decode('utf-8', 'replace')[-600:]}")
    return r.stdout.decode("utf-8", "replace")


def export_pdf(pptx: Path, pdf: Path | None = None) -> Path:
    pdf = pdf or pptx.with_suffix(".pdf")
    powershell("export_pdf.ps1", str(pptx.resolve()), str(pdf.resolve()))
    if not pdf.exists():
        raise RuntimeError("PowerPoint made no PDF")
    return pdf


def pdf_pages(pdf: Path) -> list[str]:
    import pymupdf as fitz
    with fitz.open(pdf) as doc:
        return [page.get_text() for page in doc]


def ocr_pages(pdf: Path) -> list[str]:
    """OCR of every page at 250 dpi; cached by the PDF's hash, so a rerun of the check is instant.
    Windows' own OCR (windows_ocr.ps1); env MARKET_OCR replaces it (a command: images in, texts out split by \\f)."""
    import pymupdf as fitz
    engine = os.environ.get("MARKET_OCR") or "windows_ocr.ps1"  # in the cache name: two OCR tools never share a cache
    cache = Path(tempfile.gettempdir()) / "committee-ocr" / f"py-{hashlib.sha1(pdf.read_bytes()).hexdigest()}-{hashlib.sha1(engine.encode()).hexdigest()[:8]}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp, fitz.open(pdf) as doc:
        images = []
        for i, page in enumerate(doc, 1):
            images.append(str(Path(tmp) / f"p{i:03d}.png"))
            page.get_pixmap(dpi=250).save(images[-1])  # 2,500 px wide: older Windows OCR refuses over 2,600
        if os.environ.get("MARKET_OCR"):
            out = subprocess.run([*shlex.split(os.environ["MARKET_OCR"]), *images], capture_output=True, check=True).stdout.decode("utf-8", "replace")
        else:
            out = powershell("windows_ocr.ps1", *images)
    texts = out.split("\f")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(texts), encoding="utf-8")
    return texts


# ---------------------------------------------------------------- upload
def login() -> str:
    import requests
    email, password = os.environ.get("MARKET_PUBLISHER_EMAIL"), os.environ.get("MARKET_PUBLISHER_PASSWORD")
    if not email or not password:
        raise RuntimeError("MARKET_PUBLISHER_EMAIL / MARKET_PUBLISHER_PASSWORD missing from .env")
    r = requests.post(f"{SUPABASE}/auth/v1/token?grant_type=password", headers={"apikey": PUBLISHABLE_KEY},
                      json={"email": email, "password": password}, timeout=30)
    if not r.ok:
        raise RuntimeError(f"login refused ({r.status_code})")
    return r.json()["access_token"]


def upload(pdf: Path, title: str, date: str, topics: list[str], summary: str, notes: str) -> str:
    import requests
    auth = {"apikey": PUBLISHABLE_KEY, "Authorization": "Bearer " + login()}
    # Same file name = the same deck re-exported after an edit: the file is replaced.
    r = requests.post(f"{SUPABASE}/storage/v1/object/committee/{pdf.name}", data=pdf.read_bytes(), timeout=300,
                      headers={**auth, "Content-Type": "application/pdf", "x-upsert": "true"})
    if not r.ok:
        raise RuntimeError(f"upload refused ({r.status_code}): {r.text[:300]}")
    row = {"kind": "presentation", "title": title, "published_on": date, "topics": topics, "body": summary.strip(),
           "notes": notes.strip(), "file_path": pdf.name}
    found = requests.get(f"{SUPABASE}/rest/v1/committee_items", params={"select": "id", "file_path": f"eq.{pdf.name}"},
                         headers=auth, timeout=30)
    found.raise_for_status()
    rid = found.json()[0]["id"] if found.json() else None
    headers = {**auth, "Content-Type": "application/json", "Prefer": "return=representation"}
    r = (requests.patch(f"{SUPABASE}/rest/v1/committee_items", params={"id": f"eq.{rid}"}, json=row, headers=headers, timeout=30)
         if rid else requests.post(f"{SUPABASE}/rest/v1/committee_items", json=row, headers=headers, timeout=30))
    if not r.ok or not r.json():
        raise RuntimeError(f"row refused ({r.status_code}): {r.text[:300]}")
    return ("Updated" if rid else "Published") + f": {title} · {date} · row {r.json()[0]['id']}"


# ---------------------------------------------------------------- commands
def deck_facts(pdf: Path, title: str | None, date: str | None) -> tuple[str, str]:
    m = re.match(r"(\d{2})(\d{2})(\d{4})_.*?_(Technical|Macro)_Update\.pdf$", pdf.name)
    title = title or (f"{m.group(4)} Update" if m else None)
    date = date or (f"{m.group(3)}-{m.group(2)}-{m.group(1)}" if m else None)
    if not title or not date:
        raise SystemExit("Give --title and --date (the file name is not DDMMYYYY_..._Technical_Update.pdf).")
    return title, date


def run_check(pdf: Path, summary_file: Path, notes_file: Path) -> tuple[str, str]:
    pages = pdf_pages(pdf)
    ocr = ocr_pages(pdf)
    pages = [t + "\n" + (ocr[i] if i < len(ocr) else "") for i, t in enumerate(pages)]
    summary, notes = summary_file.read_text(encoding="utf-8-sig"), notes_file.read_text(encoding="utf-8-sig")
    errors = check_notes(pages, notes, summary)
    lines = sum(1 for l in notes.splitlines() if l.strip())
    print(f"{pdf.name}: {len(pages)} pages · {lines} slide notes, {len(notes)} characters · summary {len(summary)} characters")
    if errors:
        print("NOT READY:\n- " + "\n- ".join(errors))
        raise SystemExit(1)
    print("Check passed.")
    return summary, notes


def selftest(pptx: Path) -> None:
    ok = True
    def step(name, fn):
        nonlocal ok
        try:
            print(f"OK   {name}: {fn()}")
        except Exception as e:  # report every step, then fail once
            ok = False
            print(f"FAIL {name}: {e}")
    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(tmp) / "selftest.pdf"
        step("PDF reader (PyMuPDF)", lambda: __import__("pymupdf").__doc__.split(":")[0])
        step("PowerPoint makes the PDF", lambda: f"{len(pdf_pages(export_pdf(pptx, pdf)))} pages")
        step("Windows OCR reads page 1", lambda: f"{len(powershell('windows_ocr.ps1', page_png(pdf, tmp)).strip())} characters")
        step("Market login", lambda: "token received" if login() else "")
    raise SystemExit(0 if ok else 1)


def page_png(pdf: Path, tmp: str) -> str:
    import pymupdf as fitz
    with fitz.open(pdf) as doc:
        path = str(Path(tmp) / "p1.png")
        doc[0].get_pixmap(dpi=250).save(path)
    return path


def logic_check() -> None:
    """python tools/market_publish.py logic: the number check and note format, no files needed."""
    pages = ["Title", "US CPI 3.6% consensus, core 2.5%", "Brent 93 USD"]
    assert check_notes(pages, "s2 · CPI expected at 3.6%, core 2.5%\ns3 · Brent at $93", "- CPI 3.6% [p. 2]") == []
    assert check_notes(pages, "s2 · CPI at 3.7%\ns2 · again\ns9 · late\nfree text", "Brent 95 [p. 3]") == [
        "s2: not in the deck: 3.7", "s2: two lines", "s9: the deck has 3 pages", 'not "s<page> · text": free text',
        "summary: not in the deck: 95"]
    assert unsupported("7,659 and 1.1%", "7,659.44 · -1.09%") == []
    print("logic OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["pdf", "check", "publish", "selftest", "logic"])
    ap.add_argument("file", nargs="?")
    ap.add_argument("--title")
    ap.add_argument("--date")
    ap.add_argument("--topics")
    ap.add_argument("--summary")
    ap.add_argument("--notes")
    a = ap.parse_args()
    try:  # the Wednesday launcher already loads .env; a run by hand needs it too
        from dotenv import load_dotenv
        load_dotenv(TOOLS.parent / ".env")
    except ImportError:
        pass
    if a.command == "logic":
        return logic_check()
    if a.command == "selftest":
        return selftest(Path(a.file) if a.file else Path(os.environ.get("IC_DROPBOX_PATH", ".")) / "template.pptx")
    if a.command == "pdf":
        pdf = export_pdf(Path(a.file))
        return print(f"{pdf} · {len(pdf_pages(pdf))} pages")
    pdf = Path(a.file)
    title, date = deck_facts(pdf, a.title, a.date)
    summary, notes = run_check(pdf, Path(a.summary or pdf.with_suffix(".summary.md")), Path(a.notes or pdf.with_suffix(".notes.txt")))
    if a.command == "publish":
        topics = [t.strip() for t in (a.topics or TOPICS.get(title, "")).split(",") if t.strip()][:10]
        print(upload(pdf, title, date, topics, summary, notes) + ". Live on market.herculis.ai.")


if __name__ == "__main__":
    if os.name == "nt":  # the Wednesday run prints to a Windows console: keep "·" and "€" intact
        sys.stdout.reconfigure(encoding="utf-8")
    main()
