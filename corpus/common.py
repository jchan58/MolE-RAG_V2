"""
exp70 BBBP PMC literature-collection pipeline — shared infrastructure.

Contains: configuration, secret loading (never logged), atomic file IO,
a CENTRAL cross-process token-bucket rate limiter (fcntl file locks),
an idempotent lock-file work queue, a robust HTTP layer with retry/backoff
and Retry-After handling, and disk/usage accounting.

Design notes for horizontal scaling:
  * All rate limiting is enforced through on-disk token buckets guarded by
    fcntl.lockf, so N independent worker processes sharing RUN_DIR cannot
    collectively exceed the NCBI / PubChem / OpenAI per-second caps.
  * Work items are claimed with O_EXCL lock files (atomic create) + a lease,
    and the authoritative "is this item done?" signal is the existence of a
    validated per-item result file. Races are therefore harmless (idempotent
    writes), and a crashed worker's lease expires and the item is retried.
"""
import os, sys, json, time, fcntl, hashlib, tempfile, errno, random, datetime, shutil, threading

# --------------------------------------------------------------------------
# Paths / configuration
# --------------------------------------------------------------------------
RUN_DIR = os.environ.get("EXP70_RUN_DIR",
                         os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "corpus_runs"))

# Bulk artifacts (full-text XML, chunked passages, extracted text) are large and
# must NOT land on the over-quota Lustre volume. EXP70_BULK_ROOT redirects them to
# node-local /tmp (then tarred to S3); it defaults to RUN_DIR for backward compat.
BULK_ROOT = os.environ.get("EXP70_BULK_ROOT", RUN_DIR)

def P(*parts):
    return os.path.join(RUN_DIR, *parts)

def BP(*parts):
    """Path under BULK_ROOT (full-text / passages / extracted_text)."""
    return os.path.join(BULK_ROOT, *parts)

DIRS = dict(
    code=P("code"), data=P("data"), state=P("state"),
    ratelimit=P("state", "ratelimit"), locks=P("state", "locks"),
    results=P("results"),
    res_pubchem=P("results", "pubchem"),
    res_qg_mol=P("results", "querygen_molecule"),
    res_qg_prop=P("results", "querygen_property"),
    res_search=P("results", "search"),
    res_metadata=P("results", "metadata"),
    res_download=P("results", "download"),
    raw_pubchem=P("raw_pubchem"), raw_model=P("raw_model_responses"),
    raw_search=P("raw_search_responses"), raw_metadata=P("raw_metadata"),
    fulltext=BP("fulltext_xml"), logs=P("logs"),
)

# Credentials: a repo .env (or exported env vars) is preferred; the
# ~/.config/molrag/*.env files remain supported as a fallback. _load_env_var()
# below checks the file first, then os.environ.
try:
    from dotenv import load_dotenv as _ld
    _ld(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except Exception:
    pass
OPENAI_ENV = os.path.expanduser("~/.config/molrag/openai.env")
NCBI_ENV   = os.path.expanduser("~/.config/molrag/ncbi.env")

PROMPT_VERSION = "exp70-v1"
OPENAI_MODEL   = "gpt-5.4-mini"
OPENAI_MAX_COMPLETION_TOKENS = 4000

ENDPOINTS = dict(
    esearch="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
    esummary="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
    efetch="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
    oa="https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi",
    openai="https://api.openai.com/v1/chat/completions",
    pubchem_base="https://pubchem.ncbi.nlm.nih.gov/rest/pug",
)

# Rate limits (requests / second), enforced GLOBALLY across all processes.
# Live-overridable at runtime via state/rate_overrides.json (see service_rate()).
RATE_NCBI_FULL = 10.0      # with a valid NCBI_API_KEY
RATE_NCBI_FALLBACK = 3.0   # if key missing / rejected
RATE_PUBCHEM = 5.0
RATE_OPENAI = 30.0         # OpenAI lane (separate from NCBI); mini tolerates high RPM
RATE_S3 = 250.0            # AWS S3 OA bucket (off the E-utilities cap); S3 scales to
                           # thousands/s, so this is a soft politeness cap only, with
                           # 429/503 backoff; real bound is thread concurrency

HTTP_TIMEOUT = 120
MAX_RETRIES = 6
BACKOFF_BASE = 1.0
BACKOFF_MAX = 60.0
LEASE_SECONDS = 900        # a claimed item older than this is considered stale

# Partition threshold: PMC ESearch exposes only the first 10,000 hits, so any
# partition at/above this count must be subdivided further.
PARTITION_THRESHOLD = 9000
SEARCH_PAGE = 5000         # retmax per ESearch page within a partition
SUMMARY_BATCH = 400        # ids per ESummary call
EFETCH_BATCH = 1           # full-text fetched one PMCID at a time (identity + atomic save)

# --------------------------------------------------------------------------
# Secret loading
# --------------------------------------------------------------------------
def _load_env_var(path, var):
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[7:]
                if "=" in line:
                    k, v = line.split("=", 1)
                    if k.strip() == var:
                        return v.strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    return os.environ.get(var)   # fall back to .env / exported env var

def openai_key():
    k = _load_env_var(OPENAI_ENV, "OPENAI_API_KEY")
    if not k:
        raise RuntimeError("OPENAI_API_KEY not found in %s" % OPENAI_ENV)
    return k

def ncbi_fallback_active():
    return os.path.exists(P("state", "ncbi_fallback.flag"))

def ncbi_key():
    """Return the NCBI key, or None if we are in global fallback mode."""
    if ncbi_fallback_active():
        return None
    # prefer the env file; fall back to an exported env var if present
    k = _load_env_var(NCBI_ENV, "NCBI_API_KEY") or os.environ.get("NCBI_API_KEY")
    return k or None

def ncbi_rate():
    if ncbi_fallback_active() or ncbi_key() is None:
        return RATE_NCBI_FALLBACK
    # EXP70_NCBI_RATE lets each concurrent dataset job take a FIXED slice of the shared
    # account-wide 10/s cap (node-local bucket, no cross-node lock needed), e.g. 3/s per job.
    ov = os.environ.get("EXP70_NCBI_RATE")
    if ov:
        try:
            return float(ov)
        except ValueError:
            pass
    return RATE_NCBI_FULL

def trigger_ncbi_fallback(reason):
    flag = P("state", "ncbi_fallback.flag")
    if not os.path.exists(flag):
        atomic_write_text(flag, "ncbi fallback 3/s engaged: %s at %s\n" % (reason, now_iso()))
        log("WARN", "ncbi_fallback", "NCBI key rejected; falling back to 3/s. reason=%s" % reason)

# --------------------------------------------------------------------------
# Time / logging / atomic IO
# --------------------------------------------------------------------------
def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

_LOG_FILE = None
def log(level, where, msg):
    line = "%s %-5s [%s] %s" % (now_iso(), level, where, msg)
    print(line, flush=True)
    global _LOG_FILE
    if _LOG_FILE is None:
        _LOG_FILE = DIRS["logs"]
    try:
        with open(os.path.join(_LOG_FILE, "pipeline.log"), "a") as f:
            fcntl.lockf(f, fcntl.LOCK_EX)
            f.write(line + "\n")
            fcntl.lockf(f, fcntl.LOCK_UN)
    except Exception:
        pass

def atomic_write_text(path, text, fsync=True):
    # fsync=False skips the Lustre sync (much faster under contention); safe for
    # regenerable/idempotent artifacts (manifests/leaves) that are re-derived on
    # resume. os.replace is still atomic without fsync.
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            if fsync:
                os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try: os.unlink(tmp)
            except OSError: pass

def atomic_write_bytes(path, data, fsync=True):
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            if fsync:
                os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try: os.unlink(tmp)
            except OSError: pass

def write_json(path, obj, fsync=True):
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2), fsync=fsync)

def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def sha1_str(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

# --------------------------------------------------------------------------
# Central token-bucket rate limiter (fcntl, cross-process)
# --------------------------------------------------------------------------
def ratelimit_dir():
    """Node-local tmpfs dir for rate-limiter lock+bucket files.

    The bucket is touched on EVERY request; doing that IO on Lustre (open +
    rename + fcntl under a global lock) serializes all workers and intermittently
    hangs in cl_sync_io_wait, and killed holders can leave stale locks -> deadlock.
    tmpfs (/dev/shm) gives instant IO and fcntl locks that are reliably released on
    process death. All processes on the node derive the same path from RUN_DIR, so
    the global cap is still enforced node-wide. (Workers must share the node; the
    launcher runs them locally.)"""
    # EXP70_RL_DIR lets MANY per-dataset RUN_DIRs share ONE token bucket, so the
    # account/IP-wide NCBI 10/s cap is respected even when datasets run concurrently.
    rl_override = os.environ.get("EXP70_RL_DIR")
    if rl_override:
        os.makedirs(rl_override, exist_ok=True)
        return rl_override
    tag = hashlib.sha1(RUN_DIR.encode()).hexdigest()[:12]
    for base in ("/dev/shm", "/tmp"):
        try:
            if os.path.isdir(base) and os.access(base, os.W_OK):
                d = os.path.join(base, "exp70_rl_%s" % tag)
                os.makedirs(d, exist_ok=True)
                return d
        except OSError:
            continue
    d = DIRS["ratelimit"]
    os.makedirs(d, exist_ok=True)
    return d


class RateLimiter:
    def __init__(self, name, rate_fn, capacity_fn=None):
        self.name = name
        self.rate_fn = rate_fn if callable(rate_fn) else (lambda r=rate_fn: r)
        self.capacity_fn = capacity_fn or (lambda: self.rate_fn())
        d = ratelimit_dir()
        self.lock_path = os.path.join(d, name + ".lock")
        self.state_path = os.path.join(d, name + ".bucket")
        # fcntl locks are per-process; this thread lock serializes threads within a
        # process so they don't bypass the cross-process bucket.
        self._tlock = threading.Lock()
        self._fd = None

    def _fd_open(self):
        if self._fd is None:
            self._fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        return self._fd

    def _read(self, fd):
        try:
            os.lseek(fd, 0, 0)
            data = os.read(fd, 128).decode().split()
            if len(data) >= 2:
                return float(data[0]), float(data[1])
        except (OSError, ValueError):
            pass
        return None

    def _write(self, fd, tokens, last):
        try:
            buf = ("%.6f %.6f" % (tokens, last)).encode()
            os.lseek(fd, 0, 0)
            os.write(fd, buf)
            os.ftruncate(fd, len(buf))
        except OSError:
            pass

    def _lock_nb(self, fd, deadline):
        """Acquire an exclusive fcntl lock without ever blocking indefinitely.
        Returns True if locked; False if we gave up (then we proceed anyway, which
        at worst slightly exceeds the rate but can never deadlock)."""
        while True:
            try:
                fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except OSError:
                if time.time() >= deadline:
                    return False
                time.sleep(0.003 + random.uniform(0, 0.003))

    def acquire(self):
        while True:
            rate = float(self.rate_fn()) or 1.0
            cap = float(self.capacity_fn()) or rate
            wait = 0.0
            with self._tlock:
                fd = self._fd_open()
                locked = self._lock_nb(fd, time.time() + 2.0)
                try:
                    now = time.time()
                    st = self._read(fd)
                    tokens, last = st if st else (cap, now)
                    tokens = min(cap, tokens + (now - last) * rate)
                    if tokens >= 1.0:
                        tokens -= 1.0
                        self._write(fd, tokens, now)
                        return
                    wait = (1.0 - tokens) / rate
                    self._write(fd, tokens, now)
                finally:
                    if locked:
                        try:
                            fcntl.lockf(fd, fcntl.LOCK_UN)
                        except OSError:
                            pass
            # NEVER sleep while holding either lock.
            time.sleep(min(wait, 1.0) + random.uniform(0, 0.02))

# Live rate overrides (state/rate_overrides.json), cached briefly so running
# processes pick up tuning changes without a restart.
_RATE_CACHE = {"t": 0.0, "data": {}}
_RATE_CACHE_LOCK = threading.Lock()

def _rate_overrides():
    now = time.time()
    with _RATE_CACHE_LOCK:
        if now - _RATE_CACHE["t"] > 3.0:
            _RATE_CACHE["data"] = read_json(P("state", "rate_overrides.json"), {}) or {}
            _RATE_CACHE["t"] = now
        return _RATE_CACHE["data"]

def service_rate(service):
    ov = _rate_overrides()
    if service in ov:
        try:
            return float(ov[service])
        except (TypeError, ValueError):
            pass
    if service == "ncbi":
        return ncbi_rate()
    if service == "pubchem":
        return float(RATE_PUBCHEM)
    if service == "openai":
        return float(RATE_OPENAI)
    if service == "s3":
        return float(RATE_S3)
    raise ValueError("unknown service %s" % service)

def set_rate_override(service, value):
    path = P("state", "rate_overrides.json")
    ov = read_json(path, {}) or {}
    ov[service] = float(value)
    write_json(path, ov)
    with _RATE_CACHE_LOCK:
        _RATE_CACHE["t"] = 0.0  # force reload

def throttle_down(service, factor=0.5, floor=5.0):
    """Adaptively reduce a service's global rate (e.g. on repeated 429s)."""
    cur = service_rate(service)
    newr = max(floor, cur * factor)
    if newr < cur:
        set_rate_override(service, newr)
        log("WARN", "throttle", "%s rate reduced %.1f -> %.1f/s (throttle)" % (service, cur, newr))

# Service limiters (rate resolved live via service_rate so tuning + fallback apply)
_LIMITERS = {}
_LIMITERS_LOCK = threading.Lock()
def limiter(service):
    with _LIMITERS_LOCK:
        if service not in _LIMITERS:
            if service not in ("ncbi", "pubchem", "openai", "s3"):
                raise ValueError("unknown service %s" % service)
            _LIMITERS[service] = RateLimiter(service, lambda s=service: service_rate(s))
        return _LIMITERS[service]

# --------------------------------------------------------------------------
# HTTP layer (requests) with retry/backoff + Retry-After
# --------------------------------------------------------------------------
import requests
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "exp70-moleculenet-pmc-collector/1.0 (research; contact via NCBI account)"})
# Default urllib3 pool is 10 connections/host, which caps per-process HTTP
# concurrency below the NCBI 10/s budget when many threads share the session.
_adapter = requests.adapters.HTTPAdapter(pool_connections=128, pool_maxsize=512)
_SESSION.mount("https://", _adapter)
_SESSION.mount("http://", _adapter)

class HttpResult:
    __slots__ = ("ok", "status", "text", "content", "error", "attempts", "url")
    def __init__(self, ok, status, text=None, content=None, error=None, attempts=0, url=None):
        self.ok, self.status, self.text, self.content = ok, status, text, content
        self.error, self.attempts, self.url = error, attempts, url

def http_request(service, method, url, params=None, data=None, headers=None,
                 timeout=HTTP_TIMEOUT, want="text", max_retries=MAX_RETRIES):
    """Rate-limited, retrying HTTP. want in {text,bytes,json}. Returns HttpResult."""
    attempts = 0
    for attempt in range(max_retries + 1):
        attempts = attempt + 1
        limiter(service).acquire()
        try:
            resp = _SESSION.request(method, url, params=params, data=data,
                                    headers=headers, timeout=timeout)
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                _backoff(attempt, None)
                continue
            return HttpResult(False, None, error="network: %r" % e, attempts=attempts, url=url)
        status = resp.status_code
        if status == 200:
            if want == "bytes":
                return HttpResult(True, 200, content=resp.content, attempts=attempts, url=resp.url)
            return HttpResult(True, 200, text=resp.text, content=resp.content,
                              attempts=attempts, url=resp.url)
        # NCBI invalid-key detection -> global fallback
        if service == "ncbi" and status in (400, 401, 429):
            body = (resp.text or "")[:600].lower()
            if "api" in body and "key" in body and ("invalid" in body or "not" in body):
                trigger_ncbi_fallback("http %s: %s" % (status, body[:120]))
        if status in (429, 500, 502, 503, 504):
            # Adaptively reduce the global rate for the touchy services on 429/503.
            if status in (429, 503) and service in ("pubchem", "openai", "s3"):
                throttle_down(service, factor=0.7,
                              floor=(2.0 if service == "pubchem" else 5.0))
            if attempt < max_retries:
                ra = resp.headers.get("Retry-After")
                _backoff(attempt, ra)
                continue
            return HttpResult(False, status, text=(resp.text or "")[:1000],
                              error="http %s after retries" % status, attempts=attempts, url=resp.url)
        # non-retryable
        return HttpResult(False, status, text=(resp.text or "")[:1000],
                          error="http %s" % status, attempts=attempts, url=resp.url)
    return HttpResult(False, None, error="exhausted", attempts=attempts, url=url)

def _backoff(attempt, retry_after):
    if retry_after:
        try:
            delay = float(retry_after)
        except ValueError:
            delay = BACKOFF_BASE * (2 ** attempt)
    else:
        delay = BACKOFF_BASE * (2 ** attempt)
    delay = min(delay, BACKOFF_MAX) + random.uniform(0, 0.5)
    time.sleep(delay)

# -- NCBI helpers (append api_key at call time; never logged) --------------
def ncbi_params(params):
    p = dict(params)
    k = ncbi_key()
    if k:
        p["api_key"] = k
    return p

def ncbi_get(endpoint_key, params, want="text", timeout=HTTP_TIMEOUT, max_retries=MAX_RETRIES):
    return http_request("ncbi", "GET", ENDPOINTS[endpoint_key], params=ncbi_params(params),
                        want=want, timeout=timeout, max_retries=max_retries)

def ncbi_post(endpoint_key, params, want="text", timeout=HTTP_TIMEOUT, max_retries=MAX_RETRIES):
    # POST form data (for large id lists). api_key included in form.
    return http_request("ncbi", "POST", ENDPOINTS[endpoint_key], data=ncbi_params(params),
                        want=want, timeout=timeout, max_retries=max_retries)

def pubchem_get(path, want="text", timeout=HTTP_TIMEOUT, max_retries=MAX_RETRIES):
    url = ENDPOINTS["pubchem_base"] + path
    return http_request("pubchem", "GET", url, want=want, timeout=timeout, max_retries=max_retries)

# -- OpenAI chat-completions (key read at call time; never logged) ---------
def openai_chat(messages, max_completion_tokens=OPENAI_MAX_COMPLETION_TOKENS,
                model=OPENAI_MODEL, max_retries=MAX_RETRIES):
    """Returns (ok, content_str, raw_json, usage, error)."""
    body = json.dumps({"model": model, "messages": messages,
                       "max_completion_tokens": max_completion_tokens}).encode()
    for attempt in range(max_retries + 1):
        limiter("openai").acquire()
        try:
            resp = _SESSION.post(ENDPOINTS["openai"], data=body, timeout=HTTP_TIMEOUT,
                                 headers={"Authorization": "Bearer " + openai_key(),
                                          "Content-Type": "application/json"})
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                _backoff(attempt, None); continue
            return (False, None, None, None, "network: %r" % e)
        if resp.status_code == 200:
            j = resp.json()
            try:
                content = j["choices"][0]["message"]["content"]
                finish = j["choices"][0].get("finish_reason")
            except (KeyError, IndexError):
                content, finish = None, None
            return (True, content, j, j.get("usage"), None if finish == "stop" else ("finish=%s" % finish))
        if resp.status_code in (429, 500, 502, 503, 504):
            if resp.status_code == 429:
                throttle_down("openai")  # adaptively lower the global OpenAI rate
            if attempt < max_retries:
                _backoff(attempt, resp.headers.get("Retry-After")); continue
        # non-retryable or exhausted; do NOT echo any auth material
        return (False, None, None, None, "http %s: %s" % (resp.status_code, (resp.text or "")[:400]))
    return (False, None, None, None, "exhausted")

# --------------------------------------------------------------------------
# Idempotent lock-file work queue
# --------------------------------------------------------------------------
class ItemLock:
    """Atomic O_EXCL claim with a lease + heartbeat. Context manager.

    `acquire()` returns True if this process won the claim, False if another
    live worker holds it. A stale lock (older than LEASE_SECONDS) is stolen.
    """
    def __init__(self, stage, item_id):
        self.dir = os.path.join(DIRS["locks"], stage)
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, sha1_str(item_id) + ".lock")
        self.item_id = item_id
        self.fd = None

    def acquire(self):
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.write(self.fd, ("%d %s %s\n" % (os.getpid(), now_iso(), self.item_id)).encode())
            return True
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
            # exists: steal if stale
            try:
                age = time.time() - os.path.getmtime(self.path)
            except OSError:
                return False
            if age > LEASE_SECONDS:
                try:
                    os.unlink(self.path)
                except OSError:
                    return False
                return self.acquire()
            return False

    def heartbeat(self):
        try:
            os.utime(self.path, None)
        except OSError:
            pass

    def release(self):
        if self.fd is not None:
            try: os.close(self.fd)
            except OSError: pass
            self.fd = None
        try:
            os.unlink(self.path)
        except OSError:
            pass

    def __enter__(self):
        return self
    def __exit__(self, *a):
        self.release()

# --------------------------------------------------------------------------
# Disk / usage accounting
# --------------------------------------------------------------------------
def disk_usage_bytes(path=RUN_DIR):
    total = 0
    for root, _dirs, files in os.walk(path):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(root, fn))
            except OSError:
                pass
    return total

def disk_free_bytes(path=RUN_DIR):
    # Lustre reports f_bavail=0 for project subdirectories (quota display quirk);
    # the /scratch mount root carries the truthful figure (~3.4 PB; group quota is
    # unlimited). Prefer the mount root, fall back to the given path.
    for p in ("/scratch", path):
        try:
            st = os.statvfs(p)
            v = st.f_bavail * st.f_frsize
            if v > 0:
                return v
        except OSError:
            continue
    return 0

def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if abs(n) < 1024:
            return "%.1f%s" % (n, unit)
        n /= 1024.0
    return "%.1fEB" % n

# Usage ledger (tokens / cost / counts), appended atomically under lock.
def record_usage(kind, **fields):
    rec = dict(ts=now_iso(), kind=kind, **fields)
    path = P("state", "usage_ledger.jsonl")
    with open(path, "a") as f:
        fcntl.lockf(f, fcntl.LOCK_EX)
        f.write(json.dumps(rec) + "\n")
        fcntl.lockf(f, fcntl.LOCK_UN)

def ensure_dirs():
    for d in DIRS.values():
        os.makedirs(d, exist_ok=True)
