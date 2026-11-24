"""Deploy OpenDFM/ChemDFM-v2.0-14B as a SageMaker real-time endpoint `chemdfm-v2-14b`.

Reuses chemdfm_baseline infra (role/bucket/region). Weights are downloaded in-container from the
HF hub (HF_HOME=/tmp/hf); the model.tar.gz carries only the inference code. 14B bf16 (~28GB) needs
a 48GB GPU -> fallback order ml.g6e.2xlarge -> g6e.8xlarge -> g6e.12xlarge -> g5.12xlarge.

CAPACITY CAVEAT: g6e quota is 1/type and capacity-contended. Total wall-clock budget DEPLOY_BUDGET_S
(default 1500s ~25min). If no instance type reaches InService within budget, give up, delete any
half-created endpoint, and write status=capacity_blocked. Does NOT spin indefinitely.

Writes live status to deploy_status.json. Usage: python deploy.py
"""
import sagemaker, boto3, json, os, time, tarfile, io
from sagemaker.pytorch import PyTorchModel

ROLE = "arn:aws:iam::833984135295:role/service-role/AmazonSageMaker-ExecutionRole-20260922T220415"
BUCKET = "molerag"; REGION = "us-east-1"
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "code")
ENDPOINT = "chemdfm-v2-14b"
FALLBACK = os.environ.get("FALLBACK", "ml.g6e.2xlarge,ml.g6e.8xlarge,ml.g6e.12xlarge,ml.g5.12xlarge").split(",")
BUDGET_S = int(os.environ.get("DEPLOY_BUDGET_S", "1500"))
HEALTH_TIMEOUT = 1800  # seconds SageMaker waits for /ping (covers 28GB download+load)
STATUS = os.path.join(HERE, "deploy_status.json")

boto_sess = boto3.Session(region_name=REGION)
sess = sagemaker.Session(boto_sess)
sm = boto_sess.client("sagemaker")

T0 = time.time()
_state = {"endpoint": ENDPOINT, "status": "starting", "tried": [], "t_start": T0}


def save(**kw):
    _state.update(kw); _state["elapsed_s"] = round(time.time() - T0, 1)
    json.dump(_state, open(STATUS, "w"), indent=1)
    print("STATUS", json.dumps({k: _state[k] for k in ("status", "tried", "elapsed_s") if k in _state}), flush=True)


def remaining():
    return BUDGET_S - (time.time() - T0)


def endpoint_exists():
    try:
        sm.describe_endpoint(EndpointName=ENDPOINT); return True
    except sm.exceptions.ClientError:
        return False


def delete_endpoint_blocking():
    if not endpoint_exists():
        return
    try:
        sm.delete_endpoint(EndpointName=ENDPOINT)
    except Exception as e:
        print("delete_endpoint err", str(e)[:160], flush=True)
    for _ in range(40):
        if not endpoint_exists():
            return
        time.sleep(5)


def poll_until(name, budget_s):
    """Poll DescribeEndpoint until InService/Failed or budget exhausted."""
    t = time.time()
    last = None
    while True:
        d = sm.describe_endpoint(EndpointName=name)
        st = d["EndpointStatus"]
        if st != last:
            print(f"  endpoint {name}: {st}", flush=True); last = st
            save(cur_endpoint_status=st)
        if st == "InService":
            return "InService", d
        if st == "Failed":
            return "Failed", d
        if time.time() - t > budget_s or remaining() <= 0:
            return "Timeout", d
        time.sleep(15)


def main():
    save(status="packaging_model")
    # minimal placeholder model.tar.gz (real weights pulled from HF in-container)
    dummy = os.path.join(HERE, "dummy_model.tar.gz")
    with tarfile.open(dummy, "w:gz") as t:
        info = tarfile.TarInfo("placeholder.txt"); b = b"chemdfm-v2-14b: weights loaded from HF hub\n"
        info.size = len(b); t.addfile(info, io.BytesIO(b))
    model_data = sess.upload_data(path=dummy, bucket=BUCKET, key_prefix="chemdfm_baseline/chemdfm_v2_ep/model")
    print("model_data ->", model_data, flush=True)

    env = {
        "MODEL_NAME": "OpenDFM/ChemDFM-v2.0-14B",
        "HF_HOME": "/tmp/hf", "HF_HUB_CACHE": "/tmp/hf/hub", "TRANSFORMERS_CACHE": "/tmp/hf",
        "TOKENIZERS_PARALLELISM": "false", "HF_HUB_ENABLE_HF_TRANSFER": "0",
        "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
        "SAGEMAKER_MODEL_SERVER_TIMEOUT": "1800", "TS_DEFAULT_RESPONSE_TIMEOUT": "1800",
    }
    model = PyTorchModel(
        model_data=model_data, role=ROLE, entry_point="inference.py", source_dir=CODE,
        framework_version="2.3", py_version="py311", env=env, sagemaker_session=sess,
    )
    # Register the SageMaker Model resource once (repacks code into model.tar.gz).
    model.create(instance_type=FALLBACK[0])
    save(status="model_created", model_name=model.name)
    print("model resource:", model.name, flush=True)

    # clear any pre-existing endpoint with our name
    delete_endpoint_blocking()

    for inst in FALLBACK:
        if remaining() <= 120:
            save(status="capacity_blocked", reason=f"budget exhausted before trying {inst}")
            print("BUDGET EXHAUSTED before", inst, flush=True)
            break
        _state["tried"].append(inst)
        cfg = f"chemdfm-v2-14b-{inst.replace('.', '-')}-{int(time.time())}"
        try:
            sm.create_endpoint_config(EndpointConfigName=cfg, ProductionVariants=[{
                "VariantName": "AllTraffic", "ModelName": model.name, "InstanceType": inst,
                "InitialInstanceCount": 1,
                "ContainerStartupHealthCheckTimeoutInSeconds": HEALTH_TIMEOUT,
            }])
        except Exception as e:
            print(f"  {inst}: create_endpoint_config failed {str(e)[:200]}", flush=True)
            save(status="trying", last_error=f"{inst}: cfg {str(e)[:200]}"); continue
        save(status="creating_endpoint", instance=inst, cfg=cfg)
        try:
            sm.create_endpoint(EndpointName=ENDPOINT, EndpointConfigName=cfg)
        except Exception as e:
            print(f"  {inst}: create_endpoint failed {str(e)[:200]}", flush=True)
            save(status="trying", last_error=f"{inst}: create {str(e)[:200]}"); continue

        # let each attempt resolve naturally (InsufficientInstanceCapacity surfaces as Failed in ~30 min);
        # cap at 45 min so a pathological Creating can't eat the whole budget.
        budget_this = min(2700, max(60, remaining()))
        res, d = poll_until(ENDPOINT, budget_this)
        if res == "InService":
            save(status="InService", instance=inst, cfg=cfg,
                 endpoint_arn=d.get("EndpointArn"))
            print("ENDPOINT IN SERVICE on", inst, flush=True)
            return
        reason = d.get("FailureReason", "")
        print(f"  {inst}: result={res} reason={str(reason)[:260]}", flush=True)
        save(status="trying", instance=inst, last_result=res, last_failure=str(reason)[:260])
        # tear down before trying the next type (frees the quota slot / clears name)
        delete_endpoint_blocking()

    # fell through: no type succeeded within budget
    delete_endpoint_blocking()
    if _state.get("status") != "InService":
        save(status="capacity_blocked",
             note="No fallback instance reached InService within budget. g6e quota 1/type is capacity-contended.")
        print("CAPACITY BLOCKED", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        save(status="error", error=f"{type(e).__name__}: {str(e)[:300]}", trace=traceback.format_exc()[:1200])
        print("DEPLOY ERROR", traceback.format_exc()[:1500], flush=True)
        raise
