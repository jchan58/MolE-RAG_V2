"""setsid watcher for the MolLM full run: polls the main job (job_full.json) via boto3 (no ps|grep),
periodically refreshing RESULTS.md from live S3 incremental results, and runs a final aggregate when
the job reaches a terminal state. Logs to watch.log."""
import json, time, subprocess, sys, os
import boto3

HERE = os.path.dirname(os.path.abspath(__file__))
sm = boto3.client("sagemaker", region_name="us-east-1")
PY = sys.executable


def agg():
    try:
        subprocess.run([PY, os.path.join(HERE, "aggregate.py")], cwd=HERE,
                       timeout=300, check=False)
    except Exception as e:
        print("aggregate error:", str(e)[:200], flush=True)


def main():
    job = json.load(open(os.path.join(HERE, "job_full.json")))["job"]
    print("watching", job, flush=True)
    while True:
        try:
            st = sm.describe_training_job(TrainingJobName=job)["TrainingJobStatus"]
        except Exception as e:
            print("describe error:", str(e)[:150], flush=True); time.sleep(60); continue
        print(time.strftime("%H:%M:%S"), job, "->", st, flush=True)
        agg()  # refresh partial RESULTS.md from incremental
        if st in ("Completed", "Failed", "Stopped"):
            print("TERMINAL:", st, "-> final aggregate", flush=True)
            agg()
            break
        time.sleep(300)
    print("WATCH DONE", flush=True)


if __name__ == "__main__":
    main()
