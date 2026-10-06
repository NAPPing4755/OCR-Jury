"""Reads the remote Zenodo zip's index via HTTP range requests (no full download)
and summarizes its folder structure, so we only fetch what we need."""
from collections import Counter
from remotezip import RemoteZip

URL = "https://zenodo.org/records/18597235/files/OCR_18th_century.zip?download=1"

with RemoteZip(URL) as z:
    infos = z.infolist()
    print(f"total entries: {len(infos)}")

    top_dirs = Counter()
    for i in infos:
        parts = i.filename.split("/")
        key = "/".join(parts[:3]) if len(parts) > 3 else "/".join(parts[:-1])
        top_dirs[key] += 1

    print("\nfolders (first 3 levels) -> file count:")
    for d, n in sorted(top_dirs.items()):
        print(f"  {n:7d}  {d}")

    print("\nsample entries:")
    for i in infos[:15]:
        print(f"  {i.file_size:10d}  {i.filename}")
