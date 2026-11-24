"""Monitor SageMaker training jobs: status + CloudWatch log tail.
Usage:
  python monitor.py status <job_name> [<job_name> ...]
  python monitor.py logs   <job_name> [n_lines]   # tail CloudWatch
  python monitor.py grep   <job_name> <pattern>   # grep CloudWatch
"""
import sys, boto3, time
REGION = "us-east-1"
sm = boto3.client("sagemaker", region_name=REGION)
logs = boto3.client("logs", region_name=REGION)

def status(names):
    for n in names:
        d = sm.describe_training_job(TrainingJobName=n)
        sec = d.get("BillableTimeInSeconds")
        sm_ = d.get("SecondaryStatus")
        reason = d.get("FailureReason", "")
        print(f"{n}\n  status={d['TrainingJobStatus']} secondary={sm_} billable_s={sec} inst={d['ResourceConfig']['InstanceType']}")
        if reason:
            print("  FailureReason:", reason[:500])

def _events(name, limit=None, pattern=None):
    group = "/aws/sagemaker/TrainingJobs"
    # find stream(s) for this job
    streams = []
    tok = None
    while True:
        kw = dict(logGroupName=group, logStreamNamePrefix=name, orderBy="LogStreamName", limit=50)
        if tok: kw["nextToken"] = tok
        r = logs.describe_log_streams(**kw)
        streams += [s["logStreamName"] for s in r.get("logStreams", [])]
        tok = r.get("nextToken")
        if not tok: break
    out = []
    for st in streams:
        tok = None
        while True:
            kw = dict(logGroupName=group, logStreamName=st, startFromHead=True)
            if tok: kw["nextToken"] = tok
            r = logs.get_log_events(**kw)
            evs = r.get("events", [])
            for e in evs:
                msg = e["message"]
                if pattern is None or pattern in msg:
                    out.append(msg)
            nt = r.get("nextForwardToken")
            if nt == tok: break
            tok = nt
    return out

def logs_tail(name, n=120):
    evs = _events(name)
    for m in evs[-n:]:
        print(m)

def grep(name, pattern):
    for m in _events(name, pattern=pattern):
        print(m)

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "status":
        status(sys.argv[2:])
    elif cmd == "logs":
        logs_tail(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 120)
    elif cmd == "grep":
        grep(sys.argv[2], sys.argv[3])
