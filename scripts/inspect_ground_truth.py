"""Profiles the English ground truth before any code is written against it:
sources, line types, lengths, missing images, and every non-ASCII character."""
import collections
import csv
import io
import statistics
import unicodedata
import zipfile

z = zipfile.ZipFile("data/raw/OCR_Testing.zip")
text = z.read("full_english_dataset_relative.csv").decode("utf-8-sig")
rows = list(csv.DictReader(io.StringIO(text), delimiter="\t"))
print("rows:", len(rows), "| columns:", list(rows[0].keys()))

print("by source folder:", dict(collections.Counter(r["file_name"].split("\\")[0] for r in rows)))

kinds = collections.Counter()
for r in rows:
    parts = r["file_name"].rsplit("_", 3)
    kinds[parts[-3] if len(parts) >= 4 else "?"] += 1
print("by line type:", dict(kinds.most_common()))

lens = [len(r["text"]) for r in rows]
print(f"line length (chars): min={min(lens)} median={statistics.median(lens)} max={max(lens)}")
print("empty-text rows:", sum(1 for r in rows if not r["text"].strip()))

names = {i.filename.replace("/", "\\") for i in z.infolist()}
print("rows whose image is missing from the zip:", sum(1 for r in rows if r["file_name"] not in names))

chars = collections.Counter(c for r in rows for c in r["text"] if not (c.isascii() and c.isprintable()))
print("non-ASCII characters (count, char, name):")
for c, n in chars.most_common(25):
    print(f"  {n:6d}  {c!r:6}  {unicodedata.name(c, '?')}")

long_s_lines = sum(1 for r in rows if "ſ" in r["text"])
print(f"lines containing long-s: {long_s_lines} ({100 * long_s_lines / len(rows):.1f}%)")
