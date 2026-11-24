"""
Parallel pull of the exp70 chunk corpus from S3 to NODE-LOCAL /tmp (scratch is over
quota). Downloads + decompresses all 24 parts concurrently, reconstructing
passages/ and extracted_text/ under /tmp/exp70_filter/chunks/. Resumable (per-part
done markers).

s3://molerag/exp70_chunks/molecule/parts/part_00..15.tar.gz  (16)
s3://molerag/exp70_chunks/property/parts/part_00..07.tar.gz  (8)
"""
import os, sys, tarfile, io, time, threading
from concurrent.futures import ThreadPoolExecutor
import boto3

BUCKET = "molerag"
BASE = "/tmp/exp70_filter"
CHUNKS = os.path.join(BASE, "chunks")
DONE = os.path.join(BASE, "parts_done")
PARTS = ([("molecule", "exp70_chunks/molecule/parts/part_%02d.tar.gz" % i) for i in range(16)] +
         [("property", "exp70_chunks/property/parts/part_%02d.tar.gz" % i) for i in range(8)])

_lock = threading.Lock()
_ctr = dict(parts=0, bytes=0, files=0)


def _done_marker(key):
    return os.path.join(DONE, key.replace("/", "_") + ".done")


def pull_one(grp, key):
    if os.path.exists(_done_marker(key)):
        return "cached"
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    data = obj["Body"].read()
    nfiles = 0
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        members = [m for m in tf.getmembers() if m.isfile()]
        for m in members:
            # safe: our own tars; strip any leading slashes / .. just in case
            name = m.name.lstrip("/").replace("..", "")
            dest = os.path.join(CHUNKS, name)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with tf.extractfile(m) as src, open(dest, "wb") as out:
                out.write(src.read())
            nfiles += 1
    open(_done_marker(key), "w").write("%d files %d bytes\n" % (nfiles, len(data)))
    with _lock:
        _ctr["parts"] += 1; _ctr["bytes"] += len(data); _ctr["files"] += nfiles
    return "ok:%d" % nfiles


def run(workers=12):
    os.makedirs(CHUNKS, exist_ok=True)
    os.makedirs(DONE, exist_ok=True)
    print("pulling %d parts -> %s" % (len(PARTS), CHUNKS), flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(pull_one, g, k): (g, k) for g, k in PARTS}
        for f in futs:
            g, k = futs[f]
            try:
                r = f.result()
                print("[%s] %s -> %s (%.0fs, %d/%d parts, %d files)" %
                      (time.strftime("%H:%M:%S"), k, r, time.time() - t0,
                       _ctr["parts"], len(PARTS), _ctr["files"]), flush=True)
            except Exception as e:
                print("ERR %s: %r" % (k, e), flush=True)
    # write a completion marker + counts
    with open(os.path.join(BASE, "PULL_DONE.txt"), "w") as fo:
        fo.write("parts=%d bytes=%d files=%d elapsed=%.0fs at %s\n" %
                 (_ctr["parts"], _ctr["bytes"], _ctr["files"], time.time() - t0, time.ctime()))
    print("PULL COMPLETE: %d parts, %d files, %.1f GB in %.0fs" %
          (_ctr["parts"], _ctr["files"], _ctr["bytes"] / 2**30, time.time() - t0), flush=True)


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
