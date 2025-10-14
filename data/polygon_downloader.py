"""
Polygon Flatfiles Downloader

Download historical OHLCV and trade data files from Polygon's S3-compatible flatfiles.

Credentials and endpoint are read from environment variables (use your .env):
- POLYGON_AWS_ACCESS_KEY_ID
- POLYGON_AWS_SECRET_ACCESS_KEY
- POLYGON_ENDPOINT_URL (default: https://files.polygon.io)
- POLYGON_REGION (default: us-east-1)

Usage examples:
- List keys under a prefix:
  python data/polygon_downloader.py list --prefix us_stocks_sip/trades_v1/2024/03/

- Download a specific file to data/backtesting:
  python data/polygon_downloader.py download --object-key us_stocks_sip/trades_v1/2024/03/2024-03-07.csv.gz --out data/backtesting

By default, bucket is 'flatfiles'.
"""

import os
import sys
from pathlib import Path
import argparse

import boto3
from botocore.config import Config
from dotenv import load_dotenv


DEFAULT_BUCKET = "flatfiles"
DEFAULT_ENDPOINT = "https://files.polygon.io"
DEFAULT_REGION = "us-east-1"


def get_s3_client():
    load_dotenv()
    access_key = (
        os.getenv("POLYGON_AWS_ACCESS_KEY_ID")
        or os.getenv("POLYGON_ACCESS_KEY_ID")
        or os.getenv("aws_access_key_id")
    )
    secret_key = (
        os.getenv("POLYGON_AWS_SECRET_ACCESS_KEY")
        or os.getenv("POLYGON_SECRET_ACCESS_KEY")
        or os.getenv("aws_secret_access_key")
    )
    session_token = (
        os.getenv("POLYGON_AWS_SESSION_TOKEN")
        or os.getenv("aws_session_token")
    )
    endpoint = os.getenv("POLYGON_ENDPOINT_URL") or DEFAULT_ENDPOINT
    region = os.getenv("POLYGON_REGION") or DEFAULT_REGION

    if not access_key or not secret_key:
        raise RuntimeError(
            "Missing credentials. Set POLYGON_AWS_ACCESS_KEY_ID and POLYGON_AWS_SECRET_ACCESS_KEY in .env"
        )

    session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        aws_session_token=session_token,
        region_name=region,
    )
    s3 = session.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        config=Config(signature_version="s3v4"),
    )
    return s3


def list_objects(bucket: str, prefix: str):
    s3 = get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    print(f"Listing objects in bucket '{bucket}' with prefix '{prefix}'...")
    page_count = 0
    total = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        page_count += 1
        contents = page.get("Contents", [])
        for obj in contents:
            print(obj["Key"])  # print keys only
        total += len(contents)
    print(f"Total objects listed: {total} across {page_count} page(s)")


def download_object(bucket: str, object_key: str, out_dir: Path):
    s3 = get_s3_client()
    out_dir.mkdir(parents=True, exist_ok=True)
    local_file_name = Path(object_key).name
    local_file_path = out_dir / local_file_name
    print(f"Downloading s3://{bucket}/{object_key} -> {local_file_path}")
    try:
        # Standard transfer manager path (performs HEAD first)
        s3.download_file(bucket, object_key, str(local_file_path))
        print("Download complete.")
        return
    except Exception as e:
        from botocore.exceptions import ClientError
        # If HEAD is forbidden but GET is allowed, fallback to streaming via get_object
        if isinstance(e, ClientError):
            err = e.response.get("Error", {})
            code = err.get("Code")
            msg = err.get("Message")
            if code in {"403", "AccessDenied"}:
                print("HEAD denied; attempting GET fallback...")
                try:
                    resp = s3.get_object(Bucket=bucket, Key=object_key)
                    body = resp["Body"]
                    with open(local_file_path, "wb") as f:
                        # Stream in 8MB chunks to avoid large memory usage
                        for chunk in iter(lambda: body.read(8 * 1024 * 1024), b""):
                            f.write(chunk)
                    print("Download complete via GET fallback.")
                    return
                except Exception as e2:
                    if isinstance(e2, ClientError):
                        err2 = e2.response.get("Error", {})
                        code2 = err2.get("Code")
                        msg2 = err2.get("Message")
                        print("Access denied when downloading object. Please check:")
                        print("- Polygon flatfiles credentials (POLYGON_AWS_ACCESS_KEY_ID / POLYGON_AWS_SECRET_ACCESS_KEY)")
                        print("- If provided, POLYGON_AWS_SESSION_TOKEN is set and valid")
                        print("- Bucket 'flatfiles' and region 'us-east-1' (POLYGON_REGION) are correct")
                        print("- Your subscription entitles access to the prefix/object key")
                        print("- Verify the key exists using 'list' with the parent prefix")
                        print(f"Server message: {msg2}")
                    raise
        # Not a ClientError or different failure; re-raise original
        raise


def build_parser():
    parser = argparse.ArgumentParser(description="Polygon flatfiles downloader")
    parser.add_argument("command", choices=["list", "download"], help="Operation to perform")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET, help="S3 bucket name (default: flatfiles)")
    parser.add_argument("--prefix", help="Prefix for listing objects (required for list)")
    parser.add_argument("--object-key", help="Object key to download (required for download)")
    parser.add_argument("--out", default="data/backtesting", help="Output directory for downloads")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "list":
        if not args.prefix:
            print("--prefix is required for list")
            sys.exit(2)
        list_objects(args.bucket, args.prefix)
        return

    if args.command == "download":
        if not args.object_key:
            print("--object-key is required for download")
            sys.exit(2)
        out_dir = Path(args.out)
        download_object(args.bucket, args.object_key, out_dir)
        return


if __name__ == "__main__":
    main()