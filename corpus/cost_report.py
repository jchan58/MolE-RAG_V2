"""Real measured $ cost of the CLAUDE judge stream, from the live usage ledger
(judge_usage_claude.jsonl): per-model input/output/cache tokens x each model's Bedrock rate,
plus cache-hit rate. Projects to the full 10k from articles done so far.

Usage accounting (Anthropic/Bedrock Converse): inputTokens = NON-cached input; cacheReadInputTokens
billed at ~10% of input; cacheWriteInputTokens (5-min) billed at ~1.25x input; outputTokens at
the output rate. $ per 1M tokens below."""
import os, sys, json, glob

BASE = "/tmp/exp70_filter"
# $ per 1M tokens: (input, output, cache_read, cache_write)
RATES = {
    "us.anthropic.claude-opus-4-5-20251101-v1:0": (5.0, 25.0, 0.5, 6.25),
    "us.anthropic.claude-opus-4-6-v1":            (5.0, 25.0, 0.5, 6.25),
    "us.anthropic.claude-sonnet-4-6":             (3.0, 15.0, 0.3, 3.75),
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0": (3.0, 15.0, 0.3, 3.75),
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": (0.8, 4.0, 0.08, 1.0),
}
DEFAULT_RATE = (5.0, 25.0, 0.5, 6.25)


def _count_done(stream="claude"):
    jout = os.path.join(BASE, "judgments_%s" % stream, "done")
    n = 0
    try:
        for sh in os.scandir(jout):
            if sh.is_dir():
                n += sum(1 for e in os.scandir(sh.path) if e.name.endswith(".done"))
    except FileNotFoundError:
        pass
    return n


def main(stream="claude", target_total=10000):
    ledger = os.path.join(BASE, "judge_usage_%s.jsonl" % stream)
    per = {}
    tot_in = tot_out = tot_cr = tot_cw = 0
    calls = 0
    try:
        with open(ledger) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                m = r.get("model") or "unknown"
                d = per.setdefault(m, dict(calls=0, input=0, output=0, cache_read=0, cache_write=0))
                d["calls"] += 1; calls += 1
                d["input"] += r.get("input", 0) or 0
                d["output"] += r.get("output", 0) or 0
                d["cache_read"] += r.get("cache_read", 0) or 0
                d["cache_write"] += r.get("cache_write", 0) or 0
                tot_in += r.get("input", 0) or 0
                tot_out += r.get("output", 0) or 0
                tot_cr += r.get("cache_read", 0) or 0
                tot_cw += r.get("cache_write", 0) or 0
    except FileNotFoundError:
        print("no ledger yet: %s" % ledger); return

    total_cost = 0.0
    model_rows = []
    for m, d in sorted(per.items()):
        ri, ro, rcr, rcw = RATES.get(m, DEFAULT_RATE)
        c = (d["input"] * ri + d["output"] * ro + d["cache_read"] * rcr + d["cache_write"] * rcw) / 1e6
        total_cost += c
        model_rows.append(dict(model=m, calls=d["calls"], input=d["input"], output=d["output"],
                               cache_read=d["cache_read"], cache_write=d["cache_write"], cost_usd=round(c, 2)))
    done = _count_done(stream)
    cache_hit = (tot_cr / (tot_cr + tot_in)) if (tot_cr + tot_in) else 0.0
    proj = (total_cost / done * target_total) if done else None
    out = dict(stream=stream, articles_done=done, target_total=target_total,
               calls=calls, input_tokens=tot_in, output_tokens=tot_out,
               cache_read_tokens=tot_cr, cache_write_tokens=tot_cw,
               cache_hit_rate=round(cache_hit, 3),
               tokens_per_article=(round((tot_in + tot_out + tot_cr) / done, 0) if done else None),
               running_cost_usd=round(total_cost, 2),
               projected_cost_full_usd=(round(proj, 2) if proj is not None else None),
               per_model=model_rows)
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    s = sys.argv[1] if len(sys.argv) > 1 else "claude"
    t = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
    main(s, t)
