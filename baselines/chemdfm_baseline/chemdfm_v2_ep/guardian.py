"""Billing-safety guardian for the chemdfm-v2-14b endpoint. Deletes the endpoint (and its config)
when run_logs/ALL_DONE.flag appears OR an absolute deadline (DEADLINE_EPOCH env) is reached, so a
dropped agent session can never leave a GPU endpoint idle-billing. Writes guardian_status.json.
Usage: DEADLINE_EPOCH=<unix_ts> python guardian.py"""
import boto3, json, os, time
EP = "chemdfm-v2-14b"
REGION = "us-east-1"
HERE = os.path.dirname(os.path.abspath(__file__))
FLAG = os.path.join(HERE, "run_logs", "ALL_DONE.flag")
STATUS = os.path.join(HERE, "guardian_status.json")
DEADLINE = float(os.environ.get("DEADLINE_EPOCH", str(time.time() + 8 * 3600)))
sm = boto3.client("sagemaker", region_name=REGION)


def exists():
    try:
        sm.describe_endpoint(EndpointName=EP); return True
    except sm.exceptions.ClientError:
        return False


def teardown(reason):
    cfg = None
    try:
        cfg = sm.describe_endpoint(EndpointName=EP).get("EndpointConfigName")
    except Exception:
        pass
    try:
        if exists(): sm.delete_endpoint(EndpointName=EP)
    except Exception as e:
        print("delete_endpoint err", str(e)[:160], flush=True)
    if cfg:
        try: sm.delete_endpoint_config(EndpointConfigName=cfg)
        except Exception: pass
    json.dump({"deleted": True, "reason": reason, "at": time.time()}, open(STATUS, "w"), indent=1)
    print(f"GUARDIAN: endpoint deleted (reason={reason})", flush=True)


def main():
    json.dump({"deleted": False, "deadline_epoch": DEADLINE}, open(STATUS, "w"), indent=1)
    while True:
        if os.path.exists(FLAG):
            teardown("ALL_DONE"); return
        if time.time() > DEADLINE:
            teardown("deadline"); return
        if not exists():
            json.dump({"deleted": True, "reason": "already_gone", "at": time.time()}, open(STATUS, "w"), indent=1)
            print("GUARDIAN: endpoint already gone; exiting", flush=True); return
        time.sleep(60)


if __name__ == "__main__":
    main()
