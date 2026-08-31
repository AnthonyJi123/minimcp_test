"""Pull files/dirs off the gate-data Modal volume.

The `modal volume get` CLI can't download directories on Windows (it opens
the destination as a file, then fails with WinError 32), so go through the
SDK and fetch each file concurrently instead.

    python scripts/pull_gate_files.py <dest_dir> <remote_path> [remote_path ...]
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import modal

vol = modal.Volume.from_name("gate-data")


def _files(remotes):
    """Expand remote paths to file entries, recursing into any directories.

    One listdir per parent directory rather than per path -- listing once
    per argument trips the volume's VolumeListFiles rate limit.
    """
    is_file, out = {}, []
    parents = {str(PurePosixPath(r).parent) for r in remotes}
    for parent in {"/" if p in (".", "") else p for p in parents}:
        for e in vol.listdir(parent):
            is_file[e.path] = e.type == modal.volume.FileEntryType.FILE
    for r in remotes:
        if is_file.get(r.strip("/")) or is_file.get(r):
            out.append(r)
            continue
        out += [e.path for e in vol.listdir(r, recursive=True)
                if e.type == modal.volume.FileEntryType.FILE]
    return out


def _fetch(remote, dest):
    out = dest / remote
    if out.exists() and out.stat().st_size > 0:
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    with open(tmp, "wb") as f:
        vol.read_file_into_fileobj(remote, f)
    tmp.replace(out)
    return out.stat().st_size


def main():
    dest, remotes = Path(sys.argv[1]), sys.argv[2:]
    todo = _files(remotes)
    print(f"{len(todo)} files -> {dest}", flush=True)
    total = done = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        for n in ex.map(lambda p: _fetch(p, dest), todo):
            total += n
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(todo)}  {total/1e6:.0f} MB", flush=True)
    print(f"done: {done} files, {total/1e6:.1f} MB new")


if __name__ == "__main__":
    main()
