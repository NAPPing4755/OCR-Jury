"""Downloads only OCR_Testing.zip (~1.07 GB) out of the 7 GB Zenodo archive.

remotezip's many small range requests stalled against Zenodo, so this pulls
the inner file's exact compressed byte range as ONE continuous stream
(resumable if the connection drops), then inflates it locally.

Offsets come from the outer zip's central directory (read via remotezip).
"""
import os
import struct
import subprocess
import zlib
import requests
from remotezip import RemoteZip

URL = "https://zenodo.org/records/18597235/files/OCR_18th_century.zip?download=1"
TARGET = "OCR_Testing.zip"
RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
OUT = os.path.join(RAW_DIR, TARGET)
os.makedirs(RAW_DIR, exist_ok=True)


def locate_member():
    with RemoteZip(URL) as z:
        info = z.getinfo(TARGET)
    # Local file header is 30 fixed bytes + filename + extra field. Its
    # lengths can differ from the central directory's, so read it directly.
    header = requests.get(URL, headers={"Range": f"bytes={info.header_offset}-{info.header_offset + 29}"},
                          timeout=60).content
    assert header[:4] == b"PK\x03\x04", "not a local file header"
    name_len, extra_len = struct.unpack("<HH", header[26:30])
    data_start = info.header_offset + 30 + name_len + extra_len
    return data_start, info.compress_size, info.compress_type, info.file_size


CHUNK = 100_000_000


def chunk_paths(size):
    return [os.path.join(RAW_DIR, f"{TARGET}.part{i:02d}") for i in range((size + CHUNK - 1) // CHUNK)]


def download_range(start, size):
    """Fetches the range as fixed 100 MB chunks with curl. A chunk that
    already exists at full size is skipped, so re-running resumes cheaply.
    (Python `requests` stalled on this Zenodo redirect; curl did not.)"""
    for i, path in enumerate(chunk_paths(size)):
        a = start + i * CHUNK
        b = min(start + size, a + CHUNK) - 1
        want = b - a + 1
        if os.path.exists(path) and os.path.getsize(path) == want:
            continue
        for attempt in range(1, 6):
            print(f"chunk {i + 1}/{len(chunk_paths(size))} ({want / 1e6:.0f} MB), attempt {attempt}", flush=True)
            # --speed-limit/--speed-time: abort if the transfer stalls (<1 KB/s
            # for 60 s) so the retry loop takes over instead of hanging forever.
            rc = subprocess.run(["curl.exe", "-s", "-L", "--fail", "--speed-limit", "1000", "--speed-time", "60",
                                 "-r", f"{a}-{b}", "-o", path, URL]).returncode
            if rc == 0 and os.path.getsize(path) == want:
                break
        else:
            raise RuntimeError(f"chunk {i} failed after 5 attempts")


def inflate(size, expected_size):
    print("inflating ...", flush=True)
    d = zlib.decompressobj(-15)  # raw deflate stream, as stored inside zip
    written = 0
    with open(OUT, "wb") as dst:
        for path in chunk_paths(size):
            with open(path, "rb") as src:
                while chunk := src.read(1 << 20):
                    out = d.decompress(chunk)
                    dst.write(out)
                    written += len(out)
        tail = d.flush()
        dst.write(tail)
        written += len(tail)
    assert written == expected_size, f"size mismatch: {written} != {expected_size}"
    print(f"inflated OK: {written / 1e6:.0f} MB -> {OUT}", flush=True)


if __name__ == "__main__":
    start, csize, ctype, fsize = locate_member()
    assert ctype == 8, f"expected deflate (8), got {ctype}"
    download_range(start, csize)
    inflate(csize, fsize)
    for p in chunk_paths(csize):
        os.remove(p)
    print("Done.")
