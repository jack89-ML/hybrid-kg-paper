#!/usr/bin/env python3
"""
Download arXiv papers for experiments.
"""
import json, os, sys, time, xml.etree.ElementTree as ET
from pathlib import Path

PAPERS = [
    {"id": "2502.09956", "name": "KGGen"},
    {"id": "2501.00309", "name": "GraphRAG_Survey"},
    {"id": "2407.04363", "name": "AriGraph"},
    {"id": "2306.08302", "name": "LLM_KG_Roadmap"},
    {"id": "2510.20345", "name": "LLM_KGC_Survey"},
]

def download_abstract(arxiv_id: str) -> str:
    """Scarica abstract da arXiv API."""
    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    resp = __import__("requests").get(url, timeout=30)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(resp.text)
    entry = root.find("a:entry", ns)
    title = entry.find("a:title", ns).text.strip().replace("\n", " ")
    abstract = entry.find("a:summary", ns).text.strip()
    return f"# {title}\n\n{abstract}"

def download_pdf_text(arxiv_id: str) -> str:
    """Scarica PDF e tenta estrazione testo. Fallback a abstract."""
    try:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
        resp = __import__("requests").get(pdf_url, timeout=60)
        # Salva PDF
        pdf_path = f"corpus/{arxiv_id}.pdf"
        Path(pdf_path).parent.mkdir(parents=True, exist_ok=True)
        with open(pdf_path, "wb") as f:
            f.write(resp.content)
        # Tenta estrazione testo via PyMuPDF
        try:
            import fitz
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            if len(text) > 500:
                return text
        except ImportError:
            print("  [warn] PyMuPDF not installed. Install: pip install pymupdf")
    except Exception as e:
        print(f"  [warn] PDF download failed: {e}")
    return download_abstract(arxiv_id)

def main():
    corpus_dir = Path("corpus")
    corpus_dir.mkdir(exist_ok=True)

    for paper in PAPERS:
        name = paper["name"]
        aid = paper["id"]
        txt_path = corpus_dir / f"{name}.txt"

        if txt_path.exists():
            print(f"✓ {name} already downloaded ({os.path.getsize(txt_path)} bytes)")
            continue

        print(f"→ Downloading {name} ({aid})...")
        text = download_pdf_text(aid)
        with open(txt_path, "w") as f:
            f.write(text)
        print(f"  Saved: {txt_path} ({len(text)} chars)")
        time.sleep(3)  # arXiv rate limit

    print("\nDone. Corpus files:")
    for f in sorted(corpus_dir.glob("*.txt")):
        print(f"  {f} ({os.path.getsize(f)} bytes)")

if __name__ == "__main__":
    main()
