"""Combine the 5 per-stream PROGRESS_<stream>.json files into one top-level PROGRESS.json
(each stream's done/total + calls/s + passages/s + ETA, plus an overall rollup) and push it
to s3://molerag/exp70_chunks/PROGRESS.json. Detached loop; cheap (reads 5 small JSONs)."""
import os, sys, json, time, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

BASE = "/tmp/exp70_filter"
STREAMS = ["claude", "mistral", "deepseek", "qwen", "ministral"]
OUT = os.path.join(BASE, "PROGRESS.json")
S3 = "s3://molerag/exp70_chunks/PROGRESS.json"


def combine():
    streams = {}
    tot_done = tot_total = 0
    agg_pps = agg_cps = agg_aps = 0.0
    keep = review = exclude = pending = 0
    for s in STREAMS:
        p = os.path.join(BASE, "PROGRESS_%s.json" % s)
        try:
            d = json.load(open(p))
        except Exception:
            d = None
        if not d:
            streams[s] = dict(status="not_started")
            continue
        streams[s] = dict(
            backend=d.get("backend"), model=d.get("model"),
            articles_done=d.get("articles_done"), articles_total=d.get("articles_total"),
            pct=d.get("pct"), articles_per_s=d.get("articles_per_s"),
            passages_per_s=d.get("passages_per_s"), calls_per_s=d.get("calls_per_s"),
            passages_judged=d.get("passages_judged"),
            keep=d.get("keep"), review=d.get("review"), exclude=d.get("exclude"),
            pending=d.get("pending"), parse_fail=d.get("parse_fail"),
            throttle_events=d.get("throttle_events"), eta_min=d.get("eta_min"),
            updated_at=d.get("updated_at"))
        tot_done += d.get("articles_done", 0) or 0
        tot_total += d.get("articles_total", 0) or 0
        agg_pps += d.get("passages_per_s", 0) or 0
        agg_cps += d.get("calls_per_s", 0) or 0
        agg_aps += d.get("articles_per_s", 0) or 0
        keep += d.get("keep", 0) or 0; review += d.get("review", 0) or 0
        exclude += d.get("exclude", 0) or 0; pending += d.get("pending", 0) or 0
    # overall ETA = max over streams of their own ETA (they run in parallel)
    etas = [streams[s].get("eta_min") for s in STREAMS if isinstance(streams[s].get("eta_min"), (int, float))]
    out = dict(updated_at=C.now_iso(), experiment="exp70_bbbp_judge_hybrid_5stream",
               articles_done=tot_done, articles_total=tot_total,
               pct=round(100.0 * tot_done / tot_total, 2) if tot_total else None,
               overall_articles_per_s=round(agg_aps, 3), overall_passages_per_s=round(agg_pps, 1),
               overall_calls_per_s=round(agg_cps, 2),
               eta_min_slowest_stream=(round(max(etas), 1) if etas else None),
               retention=dict(keep=keep, review=review, exclude=exclude, pending=pending),
               streams=streams)
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, OUT)
    return out


def main():
    i = 0
    while True:
        try:
            combine()
            if i % 6 == 0:
                subprocess.run(["aws", "s3", "cp", OUT, S3, "--only-show-errors"],
                               timeout=120, check=False)
        except Exception as e:
            C.log("WARN", "progress_combine", "%r" % e)
        i += 1
        time.sleep(20)


if __name__ == "__main__":
    main()
