"""Isolated PDF extraction worker. No network or application imports."""
import json
import sys
from pathlib import Path


def extract(path):
    from pypdf import PdfReader
    try:
        reader = PdfReader(path)
        if reader.is_encrypted:
            return {"error": "Password-protected PDFs are not supported. Upload an unlocked copy."}
        if len(reader.pages) > 150:
            return {"error": "PDFs must contain no more than 150 pages. Split this document."}
        pages, total = [], 0
        for page in reader.pages:
            text = page.extract_text() or ""
            total += len(text)
            if total > 1_500_000:
                return {"error": "This PDF contains too much extracted text. Split it into smaller files."}
            pages.append(text)
        return {"pages": pages}
    except Exception:
        return {"error": "The PDF is damaged or uses an unsupported encoding. Re-export it and try again."}


if __name__ == "__main__":
    # Linux/macOS worker only. Parent also enforces a wall-clock deadline.
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (768 * 1024 * 1024,) * 2)
        resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
    except (ImportError, ValueError, OSError):
        pass
    Path(sys.argv[2]).write_text(json.dumps(extract(sys.argv[1])), encoding="utf-8")
