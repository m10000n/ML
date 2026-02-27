from pathlib import Path

import boto3
import botocore
import requests
from botocore.client import BaseClient
from botocore.config import Config
from botocore.exceptions import ClientError

_IMDS_URL = "http://169.254.169.254/latest"


def get_client(region_name: str, unsigned: bool = False) -> BaseClient:
    config = Config(signature_version=botocore.UNSIGNED) if unsigned else Config()
    return boto3.client("s3", config=config, region_name=region_name)


def get_content(client: BaseClient, bucket: str, folder_path: str = "") -> tuple[list[str], list[str]]:
    if folder_path and not folder_path.endswith("/"):
        folder_path += "/"

    paginator = client.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=folder_path, Delimiter="/")

    files = []
    folders = []

    for page in page_iterator:
        if "Contents" in page:
            files += [content["Key"] for content in page["Contents"] if content["Key"] != folder_path]

        if "CommonPrefixes" in page:
            folders += [prefix["Prefix"] for prefix in page["CommonPrefixes"]]

    return files, folders


def get_file(client: BaseClient, bucket: str, file_path: str) -> bytes:
    try:
        response = client.get_object(Bucket=bucket, Key=str(file_path))
        return response["Body"].read()
    except client.exceptions.NoSuchKey:
        raise FileNotFoundError(f"Could not find this object key: {file_path}")


def download(client: BaseClient, bucket: str, file_path: str, local_file_path: str | Path) -> None:
    try:
        client.download_file(bucket, file_path, str(local_file_path))
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            raise FileNotFoundError(f"Could not find this object: {file_path}")
        raise


def stop_instance(dry_run: bool) -> None:
    try:
        token = _get_imds_token()

        instance_id = _get_metadata(token, "instance-id")
        region = _get_metadata(token, "placement/region")

        ec2_client = boto3.client("ec2", region_name=region)

        ec2_client.stop_instances(InstanceIds=[instance_id], DryRun=dry_run)
    except (
        RuntimeError,
        botocore.exceptions.NoCredentialsError,
        botocore.exceptions.ClientError,
    ) as e:
        message = f"Failed to stop aws instance{' (dry run)' if dry_run else ''}"
        if isinstance(e, RuntimeError):
            raise RuntimeError(f"{message} {e.args[0]}") from e

        if isinstance(e, botocore.exceptions.NoCredentialsError):
            raise RuntimeError(f"{message} (Ensure your IAM role has ec2:StopInstances permission).")

        if isinstance(e, botocore.exceptions.ClientError) and dry_run and "DryRunOperation" in str(e):
            pass


def _get_imds_token() -> str:
    url = f"{_IMDS_URL}/api/token"
    headers = {"X-aws-ec2-metadata-token-ttl-seconds": "3600"}  # 6 hours
    try:
        resp = requests.put(url=url, headers=headers, timeout=5)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        raise RuntimeError("Failed to fetch IMDSv2 token (Ensure IMDSv2 is enabled).") from e


def _get_metadata(token: str, path: str) -> str:
    url = f"{_IMDS_URL}/meta-data/{path}"
    headers = {"X-aws-ec2-metadata-token": token}
    try:
        resp = requests.get(url=url, headers=headers, timeout=2)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch '{path}' from IMDSv2.") from e
