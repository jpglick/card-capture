#!/usr/bin/env python3
"""
RunPod setup helper — lists or creates the endpoint needed for card-capture.

RunPod's S3-compatible API wraps Network Volumes, not general object storage.
The "bucket" is your Network Volume ID, and the endpoint URL is datacenter-specific.

Steps to get fully configured:
  1. Create a Network Volume in a supported datacenter (RunPod console)
  2. Note its Volume ID (becomes runpod_s3_bucket) and datacenter
  3. Pick the matching endpoint URL from the table below
  4. Run this script to create the serverless endpoint

Datacenter → Endpoint URL:
  US-CA-2  https://s3api-us-ca-2.runpod.io/
  US-GA-2  https://s3api-us-ga-2.runpod.io/
  US-IL-1  https://s3api-us-il-1.runpod.io/
  US-KS-2  https://s3api-us-ks-2.runpod.io/
  US-MD-1  https://s3api-us-md-1.runpod.io/
  US-MO-1  https://s3api-us-mo-1.runpod.io/
  US-MO-2  https://s3api-us-mo-2.runpod.io/
  US-NC-1  https://s3api-us-nc-1.runpod.io/
  US-NC-2  https://s3api-us-nc-2.runpod.io/
  US-NE-1  https://s3api-us-ne-1.runpod.io/
  US-WA-1  https://s3api-us-wa-1.runpod.io/
  EU-CZ-1  https://s3api-eu-cz-1.runpod.io/
  EU-RO-1  https://s3api-eu-ro-1.runpod.io/

Usage:
    python3 scripts/runpod_setup.py            # list existing endpoints + check S3
    python3 scripts/runpod_setup.py --create   # create serverless endpoint + template

Reads from / writes to card_capture_config.json.
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


def list_endpoints(api_key: str) -> None:
    import runpod
    runpod.api_key = api_key
    endpoints = runpod.get_endpoints()
    if not endpoints:
        print("No serverless endpoints found.")
        print("Run with --create to create one.")
        return

    print(f"{'ID':<20} {'Name':<30} {'Workers'}")
    print("-" * 60)
    for ep in endpoints:
        ep_id = ep.get("id", "?")
        name = ep.get("name", "?")
        workers_min = ep.get("workersMin", 0)
        workers_max = ep.get("workersMax", 3)
        print(f"{ep_id:<20} {name:<30} {workers_min}-{workers_max}")
    print(f"\nSet runpod_endpoint_id in {CONFIG_PATH.name} to use one of these.")


def check_s3(cfg: dict) -> None:
    import boto3
    from botocore.exceptions import ClientError, EndpointConnectionError

    s3_endpoint = cfg.get("runpod_s3_endpoint_url", "")
    s3_bucket = cfg.get("runpod_s3_bucket", "")
    access_key = cfg.get("runpod_s3_access_key_id", "")
    secret_key = cfg.get("runpod_s3_secret_access_key", "")

    if not s3_endpoint:
        print("  runpod_s3_endpoint_url not set.")
        print("  Create a Network Volume, find its datacenter, then set the matching URL.")
        print("  Example: https://s3api-us-ga-2.runpod.io/")
        return
    if not s3_bucket:
        print("  runpod_s3_bucket not set — this should be your Network Volume ID.")
        return
    if not access_key or not secret_key:
        print("  S3 credentials not set.")
        print("  Generate an S3 API key at: https://www.runpod.io/console/user/settings -> Storage")
        return

    print(f"  Checking s3_endpoint={s3_endpoint}  bucket={s3_bucket}…")
    s3 = boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    try:
        s3.head_bucket(Bucket=s3_bucket)
        print("  S3 connection OK — volume accessible.")
    except EndpointConnectionError:
        print(f"  Cannot reach endpoint: {s3_endpoint}")
        print("  Check the datacenter URL matches your Network Volume's datacenter.")
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "404":
            print(f"  Volume {s3_bucket} not found. Check the Volume ID.")
        elif code == "403":
            print(f"  Access denied. Check your S3 API key credentials.")
        else:
            print(f"  S3 error: {e}")


def create_endpoint(api_key: str) -> None:
    import runpod
    runpod.api_key = api_key

    print(f"Creating serverless template from image: {DOCKER_IMAGE}")
    template = runpod.create_template(
        name=ENDPOINT_NAME,
        image_name=DOCKER_IMAGE,
        docker_start_cmd="python3 -m app.runpod_handler",
        container_disk_in_gb=20,
        is_serverless=True,
        env={"PYTHONUNBUFFERED": "1"},
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
        gpu_ids="AMPERE_24,AMPERE_48",
        workers_min=0,
        workers_max=3,
        idle_timeout=60,
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
    parser = argparse.ArgumentParser(description="RunPod setup for card-capture")
    parser.add_argument("--create", action="store_true", help="Create template + serverless endpoint")
    args = parser.parse_args()

    cfg = load_config()
    api_key = cfg.get("runpod_api_key", "")
    if not api_key:
        sys.exit("runpod_api_key is not set in card_capture_config.json")

    print("=== Serverless Endpoints ===")
    if args.create:
        create_endpoint(api_key)
    else:
        list_endpoints(api_key)

    print("\n=== S3 / Network Volume ===")
    check_s3(cfg)


if __name__ == "__main__":
    main()
