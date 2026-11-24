"""
PMC Open Data Cloud Service access (sanctioned automated-retrieval route).

Bucket: s3://pmc-oa-opendata (us-east-1), PUBLIC, accessed ANONYMOUSLY (botocore
UNSIGNED) — no AWS credentials. Layout: per-article root prefixes `PMC<id>.<ver>/`
containing the JATS full-text `PMC<id>.<ver>.xml` (+ .txt/.json/.pdf/assets).

Presence of a PMCID prefix == OA full-text available. Retrieval is OFF the
E-utilities 10/s cap; it is bounded by per-process thread concurrency plus a gentle
central token bucket ("s3") and exponential backoff on throttle (SlowDown/503).
"""
import os, sys, re, time, random, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

BUCKET = "pmc-oa-opendata"
REGION = "us-east-1"

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError, BotoCoreError, EndpointConnectionError

_CLIENT = None
_CLIENT_LOCK = threading.Lock()


def client():
    global _CLIENT
    if _CLIENT is None:
        with _CLIENT_LOCK:
            if _CLIENT is None:
                cfg = Config(signature_version=UNSIGNED, region_name=REGION,
                             retries={"max_attempts": 4, "mode": "adaptive"},
                             max_pool_connections=64)
                _CLIENT = boto3.client("s3", config=cfg)
    return _CLIENT


def _is_throttle(e):
    if isinstance(e, (EndpointConnectionError, BotoCoreError)):
        return True
    if isinstance(e, ClientError):
        code = e.response.get("Error", {}).get("Code", "")
        return code in ("SlowDown", "503", "ServiceUnavailable", "RequestTimeout",
                        "InternalError", "ThrottlingException")
    return False


def list_prefix_keys(pmcid, max_retries=C.MAX_RETRIES):
    """List all object keys under PMC<id>. ; returns (keys, error)."""
    prefix = "PMC%s." % pmcid
    keys = []
    token = None
    attempt = 0
    while True:
        C.limiter("s3").acquire()
        kw = dict(Bucket=BUCKET, Prefix=prefix, MaxKeys=1000)
        if token:
            kw["ContinuationToken"] = token
        try:
            r = client().list_objects_v2(**kw)
        except (ClientError, BotoCoreError) as e:
            if _is_throttle(e) and attempt < max_retries:
                _backoff(attempt); attempt += 1; continue
            return keys, "list: %r" % e
        for o in r.get("Contents", []):
            keys.append(o["Key"])
        if r.get("IsTruncated"):
            token = r.get("NextContinuationToken")
        else:
            break
    return keys, None


def pick_xml_key(keys):
    """Choose the JATS .xml key, preferring the highest version."""
    xmls = [k for k in keys if k.lower().endswith(".xml")]
    if not xmls:
        return None
    def ver(k):
        m = re.search(r"PMC\d+\.(\d+)", k)
        return int(m.group(1)) if m else 0
    return sorted(xmls, key=ver)[-1]


def get_object_bytes(key, max_retries=C.MAX_RETRIES):
    """GET an object; returns (bytes, error)."""
    attempt = 0
    while True:
        C.limiter("s3").acquire()
        try:
            obj = client().get_object(Bucket=BUCKET, Key=key)
            return obj["Body"].read(), None
        except (ClientError, BotoCoreError) as e:
            if _is_throttle(e) and attempt < max_retries:
                _backoff(attempt); attempt += 1; continue
            return None, "get: %r" % e


def _backoff(attempt):
    time.sleep(min(C.BACKOFF_MAX, C.BACKOFF_BASE * (2 ** attempt)) + random.uniform(0, 0.5))
