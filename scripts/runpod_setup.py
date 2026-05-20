#!/usr/bin/env python3
"""
RunPod + Cloudflare R2 setup helper for card-capture.

What you need from the Cloudflare dashboard:
  1. Account ID  — shown on the R2 overview page (top right)
  2. Bucket      — create one named e.g. "card-capture" in R2 → Create bucket
  3. API token   — R2 → Manage R2 API Tokens → Create Token → Object Read & Write
                   This gives you an Access Key ID + Secret Access Key

What you need from RunPod:
  1. API key     — runpod.io/console/user/settings → API Keys
  2. Endpoint ID — created by --create (or visible in RunPod console after deploy)

Usage:
    python3 scripts/runpod_setup.py            # check R2 + list endpoints
    python3 scripts/runpod_setup.py --create   # create RunPod endpoint + template
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent / "card_capture_config.json"
DOCKER_IMAGE = "ghcr.io/jpglick/card-capture-cuda:latest"
ENDPOINT_NAME = "card-capture"


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        sys.exit(f"Config not found: {CONFIG_PATH}")
    return json.loads(CONFIG_PATH.read_text())


def save_config(cfg: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


def check_r2(cfg: dict) -> bool:
    """Verify R2 credentials and bucket. Returns True if all OK."""
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError, EndpointConnectionError, ReadTimeoutError

    account_id = cfg.get("r2_account_id", "")
    bucket = cfg.get("r2_bucket", "")
    access_key = cfg.get("r2_access_key_id", "")
    secret_key = cfg.get("r2_secret_access_key", "")

    missing = [k for k, v in [
        ("r2_account_id", account_id), ("r2_bucket", bucket),
        ("r2_access_key_id", access_key), ("r2_secret_access_key", secret_key),
    ] if not v]
    if missing:
        print(f"  Missing config fields: {', '.join(missing)}")
        print("  See the docstring at the top of this script for setup instructions.")
        return False

    endpoint = f"https://{account_id}.r2.cloudflarestorage.com"
    print(f"  Endpoint: {endpoint}")
    print(f"  Bucket:   {bucket}")

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name="auto",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(
            connect_timeout=10,
            read_timeout=15,
            retries={"max_attempts": 1},
            s3={"addressing_style": "path"},
        ),
    )

    try:
        s3.head_bucket(Bucket=bucket)
        print(f"  R2 OK — bucket '{bucket}' accessible.")
        return True
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "404":
            print(f"  Bucket '{bucket}' not found — creating it…")
            try:
                s3.create_bucket(Bucket=bucket)
                print(f"  Bucket '{bucket}' created.")
                return True
            except Exception as ce:
                print(f"  Could not create bucket: {ce}")
                return False
        elif code == "403":
            print(f"  Access denied. Check your R2 API token has Object Read & Write permissions.")
            return False
        else:
            print(f"  R2 error ({code}): {e}")
            return False
    except (EndpointConnectionError, ReadTimeoutError) as e:
        print(f"  Cannot reach R2 endpoint: {e}")
        print(f"  Check your r2_account_id is correct.")
        return False


def list_endpoints(api_key: str) -> None:
    import runpod
    runpod.api_key = api_key
    endpoints = runpod.get_endpoints()
    if not endpoints:
        print("  No serverless endpoints found. Run with --create to create one.")
        return
    print(f"  {'ID':<20} {'Name':<30} {'Workers'}")
    print("  " + "-" * 58)
    for ep in endpoints:
        ep_id = ep.get("id", "?")
        name = ep.get("name", "?")
        workers = f"{ep.get('workersMin', 0)}-{ep.get('workersMax', 3)}"
        print(f"  {ep_id:<20} {name:<30} {workers}")
    print(f"\n  Set runpod_endpoint_id in {CONFIG_PATH.name} to use one of these.")


def create_endpoint(api_key: str, cfg: dict) -> None:
    import runpod
    runpod.api_key = api_key

    r2_account_id = cfg.get("r2_account_id", "")
    r2_access_key_id = cfg.get("r2_access_key_id", "")
    r2_secret_access_key = cfg.get("r2_secret_access_key", "")

    print(f"Creating serverless template from {DOCKER_IMAGE}…")
    template = runpod.create_template(
        name=ENDPOINT_NAME,
        image_name=DOCKER_IMAGE,
        docker_start_cmd="python3 -m app.runpod_handler",
        container_disk_in_gb=20,
        is_serverless=True,
        env={
            "PYTHONUNBUFFERED": "1",
            "R2_ACCOUNT_ID": r2_account_id,
            "R2_ACCESS_KEY_ID": r2_access_key_id,
            "R2_SECRET_ACCESS_KEY": r2_secret_access_key,
        },
    )
    template_id = (
        template.get("id")
        or template.get("data", {}).get("saveTemplate", {}).get("id")
    )
    if not template_id:
        sys.exit(f"Template creation failed: {template}")
    print(f"Template created: {template_id}")

    print("Creating serverless endpoint…")
    endpoint = runpod.create_endpoint(
        name=ENDPOINT_NAME,
        template_id=template_id,
        gpu_ids="NVIDIA GeForce RTX 4090,NVIDIA GeForce RTX 5090",
        workers_min=0,
        workers_max=3,
        idle_timeout=30,
        scaler_type="QUEUE_DELAY",
        scaler_value=4,
    )
    ep_id = (
        endpoint.get("id")
        or endpoint.get("data", {}).get("saveEndpoint", {}).get("id")
    )
    if not ep_id:
        sys.exit(f"Endpoint creation failed: {endpoint}")
    print(f"Endpoint created: {ep_id}")

    cfg = load_config()
    cfg["runpod_endpoint_id"] = ep_id
    save_config(cfg)
    print(f"Saved runpod_endpoint_id={ep_id} to {CONFIG_PATH.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RunPod + R2 setup for card-capture")
    parser.add_argument("--create", action="store_true", help="Create RunPod template + endpoint")
    args = parser.parse_args()

    cfg = load_config()
    api_key = cfg.get("runpod_api_key", "")
    if not api_key:
        sys.exit("runpod_api_key is not set in card_capture_config.json")

    print("=== Cloudflare R2 ===")
    r2_ok = check_r2(cfg)

    print("\n=== RunPod Endpoints ===")
    if args.create:
        if not r2_ok:
            sys.exit("Fix R2 credentials before creating the endpoint (R2 creds are baked into the template).")
        create_endpoint(api_key, cfg)
    else:
        list_endpoints(api_key)


if __name__ == "__main__":
    main()
