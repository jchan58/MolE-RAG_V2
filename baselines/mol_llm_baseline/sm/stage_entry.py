"""STAGE job (CPU): fetch the MolLM MoleculePrediction checkpoint zip from HuggingFace,
list its members, extract the pretrained graph-encoder checkpoint(s), and upload the small
ckpt(s) + a manifest to s3://molerag/mol_llm_baseline/ckpt/ so the GPU training job pulls a
small file instead of the 13 GB archive. Public HF repo -> no auth/token needed.
"""
import os, json, zipfile, sys

HF_REPO = "andrewt28/MolLM"
HF_FILE = "MoleculePrediction/MoleculePrediction-model.zip"
S3_BUCKET = os.environ.get("S3_BUCKET", "molerag")
S3_CKPT_PREFIX = os.environ.get("S3_CKPT_PREFIX", "mol_llm_baseline/ckpt")
OUTD = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
WORK = "/tmp/mollm_stage"
os.makedirs(WORK, exist_ok=True)
os.environ.setdefault("HF_HOME", "/tmp/hf")


def s3():
    import boto3
    return boto3.client("s3", region_name="us-east-1")


def main():
    from huggingface_hub import hf_hub_download
    print("Downloading", HF_REPO, HF_FILE, "(~13 GB) ...", flush=True)
    zip_path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE, local_dir=WORK,
                              local_dir_use_symlinks=False)
    print("downloaded ->", zip_path, os.path.getsize(zip_path), "bytes", flush=True)

    zf = zipfile.ZipFile(zip_path)
    members = zf.namelist()
    manifest = [{"name": m, "size": zf.getinfo(m).file_size} for m in members]
    # find candidate pretrained encoder checkpoints
    ckpts = [m for m in members if m.endswith(".ckpt")]
    print("ZIP has", len(members), "members;", len(ckpts), "ckpt files:", flush=True)
    for c in ckpts:
        print("   ", c, zf.getinfo(c).file_size, flush=True)

    cli = s3()
    cli.put_object(Bucket=S3_BUCKET, Key=f"{S3_CKPT_PREFIX}/zip_manifest.json",
                   Body=json.dumps(manifest, indent=2).encode())
    print("uploaded manifest to", f"s3://{S3_BUCKET}/{S3_CKPT_PREFIX}/zip_manifest.json", flush=True)

    # Upload every .ckpt (flattened basename). The finetune scripts load
    # all_checkpoints/model-epoch=394.ckpt; keep full relative path too.
    uploaded = []
    for c in ckpts:
        base = os.path.basename(c)
        dst = os.path.join(WORK, "extract", base)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with zf.open(c) as src, open(dst, "wb") as out:
            while True:
                buf = src.read(1 << 24)
                if not buf:
                    break
                out.write(buf)
        sz = os.path.getsize(dst)
        key = f"{S3_CKPT_PREFIX}/{base}"
        print(f"uploading {c} -> s3://{S3_BUCKET}/{key} ({sz} bytes)", flush=True)
        cli.upload_file(dst, S3_BUCKET, key)
        uploaded.append({"zip_member": c, "s3_key": key, "basename": base, "size": sz})
        os.remove(dst)

    summary = {"hf_repo": HF_REPO, "hf_file": HF_FILE, "n_members": len(members),
               "ckpts": uploaded, "all_ckpt_members": ckpts}
    os.makedirs(OUTD, exist_ok=True)
    json.dump(summary, open(os.path.join(OUTD, "stage_summary.json"), "w"), indent=2)
    cli.put_object(Bucket=S3_BUCKET, Key=f"{S3_CKPT_PREFIX}/stage_summary.json",
                   Body=json.dumps(summary, indent=2).encode())
    print("STAGE DONE:", json.dumps(summary, indent=2), flush=True)
    if not uploaded:
        print("ERROR: no .ckpt found in archive", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
