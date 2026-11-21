"""Write installer/torch_lib_manifest.json: every file of the venv's torch/lib folder matched to its member in the official PyTorch wheel.

  python tools/make_torch_manifest.py [--wheel-url URL] [--venv-lib PATH] [--out PATH]

The wheel is read with ranged requests only. Its zip central directory gives each member's size, CRC-32 and compressed size; each member's
local header gives the offset of its deflated data. Every local file must match its member's size and CRC-32, and the two file lists must be
identical. Any mismatch fails the script and writes nothing. SHA256 is taken from the local file, which the CRC-32 check ties to the wheel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import urllib.request
import zipfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import runtime_libs  # noqa: E402

# ---------------------------------------------------------------- PUBLIC
WHEEL_URL = "https://download.pytorch.org/whl/cu126/torch-2.14.0%2Bcu126-cp312-cp312-win_amd64.whl"
VENV_LIB = ROOT / ".venv-seedvc" / "Lib" / "site-packages" / "torch" / "lib"
OUT = ROOT / "installer" / "torch_lib_manifest.json"
PREFIX = "torch/lib/"


class ManifestError(RuntimeError):
    """The local torch/lib folder does not match the wheel. The message names the file and the difference."""


def Build(wheel_url: str, venv_lib: Path) -> dict:
    total = _WheelSize(wheel_url)
    reader = runtime_libs.HttpRangeFile(wheel_url, total)
    with zipfile.ZipFile(reader) as wheel:
        members = {info.filename[len(PREFIX):]: info for info in wheel.infolist() if info.filename.startswith(PREFIX) and not info.is_dir()}
        local = {path.relative_to(venv_lib).as_posix(): path for path in venv_lib.rglob("*") if path.is_file()}
        if set(local) != set(members):
            only_local = sorted(set(local) - set(members))
            only_wheel = sorted(set(members) - set(local))
            raise ManifestError(f"the file lists differ: only local {only_local[:5]}, only in the wheel {only_wheel[:5]}")
        files = []
        for rel in sorted(local):
            info = members[rel]
            if info.compress_type != zipfile.ZIP_DEFLATED:
                raise ManifestError(f"{rel}: the wheel stores it with compression {info.compress_type}; only deflate is supported")
            crc, sha256 = _Digest(local[rel])
            size = local[rel].stat().st_size
            if size != info.file_size or crc != info.CRC:
                raise ManifestError(f"{rel}: local size and CRC-32 {size}/{crc:08x} do not match the wheel {info.file_size}/{info.CRC:08x}")
            files.append({"relpath": rel, "member": info.filename, "size": size, "sha256": sha256, "crc32": info.CRC,
                          "compress_size": info.compress_size, "data_offset": _DataOffset(reader, info)})
    return {"format": runtime_libs.MANIFEST_FORMAT, "wheel": {"url": wheel_url, "size": total}, "files": files}


def Main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wheel-url", default=WHEEL_URL)
    ap.add_argument("--venv-lib", type=Path, default=VENV_LIB)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    try:
        doc = Build(args.wheel_url, args.venv_lib)
    except ManifestError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    args.out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    total = sum(f["size"] for f in doc["files"])
    print(f"wrote {args.out}: {len(doc['files'])} files, {total} bytes, every size and CRC-32 matches the wheel")
    return 0


# ---------------------------------------------------------------- PRIVATE
def _WheelSize(url: str) -> int:
    request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 206:
            raise ManifestError(f"the wheel server ignored the byte range (HTTP {response.status})")
        return int(response.headers["Content-Range"].partition("/")[2])


def _DataOffset(reader: runtime_libs.HttpRangeFile, info: zipfile.ZipInfo) -> int:
    reader.seek(info.header_offset)
    head = reader.read(30)
    if head[:4] != b"PK\x03\x04":
        raise ManifestError(f"{info.filename}: no local file header at offset {info.header_offset}")
    name_len, extra_len = struct.unpack("<HH", head[26:30])
    return info.header_offset + 30 + name_len + extra_len


def _Digest(path: Path) -> tuple[int, str]:
    crc = 0
    digest = hashlib.sha256()
    with open(path, "rb", buffering=0) as handle:
        while True:
            block = handle.read(8 << 20)
            if not block:
                return crc & 0xFFFFFFFF, digest.hexdigest()
            crc = zlib.crc32(block, crc)
            digest.update(block)


if __name__ == "__main__":
    sys.exit(Main())
