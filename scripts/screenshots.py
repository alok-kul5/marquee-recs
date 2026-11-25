"""Capture README screenshots from a running server with headless Edge/Chrome (Windows paths).

usage: python scripts/screenshots.py [base_url] [viewer_id]
"""
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from PIL import Image

OUT = Path(__file__).resolve().parent.parent / "docs" / "screenshots"
BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
]


def shoot(url: str, dest: Path, height: int = 4905, tries: int = 4) -> Image.Image:
    exe = next(b for b in BROWSERS if os.path.exists(b))
    for _ in range(tries):
        dest.unlink(missing_ok=True)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as prof:
            subprocess.run([exe, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                            "--force-device-scale-factor=1", f"--user-data-dir={prof}",
                            f"--window-size=1440,{height}", "--virtual-time-budget=20000",
                            "--no-first-run", "--no-default-browser-check",
                            f"--screenshot={dest}", url], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=120)
            for _ in range(30):
                if dest.exists() and dest.stat().st_size > 1_500_000:
                    time.sleep(0.5)
                    img = Image.open(dest).convert("RGB")
                    dest.unlink()
                    return img
                time.sleep(0.5)
    raise RuntimeError(f"could not capture {url}")


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8077"
    viewer = sys.argv[2] if len(sys.argv) > 2 else "5705"
    OUT.mkdir(parents=True, exist_ok=True)
    f = shoot(f"{base}/?demo&shot", OUT / "_full.png")
    f.crop((0, 0, 1440, 936)).save(OUT / "hero.png", optimize=True)
    f.crop((0, 936, 1440, 2000)).save(OUT / "picker.png", optimize=True)
    f.crop((0, 2620, 1440, 3830)).save(OUT / "results.png", optimize=True)
    f.crop((0, 3830, 1440, 4800)).save(OUT / "pipeline-metrics.png", optimize=True)
    v = shoot(f"{base}/?viewer={viewer}&shot", OUT / "_viewer.png")
    v.crop((0, 2620, 1440, 3830)).save(OUT / "viewer.png", optimize=True)
    print("saved", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
