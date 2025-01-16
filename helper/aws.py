import json
import subprocess

import boto3
from botocore.exceptions import ClientError


def get_secret(secrete_name, region_name):
    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager", region_name=region_name)
    secrete_value = client.get_secret_value(SecretId=secrete_name)
    return json.loads(secrete_value["SecretString"])


def get_client(secrete):
    return boto3.client(
        "s3",
        aws_access_key_id=secrete["access_key_id"],
        aws_secret_access_key=secrete["secret_key"],
        region_name="eu-central-1",
    )


def get_content(client, bucket, folder_path="", verbose=False):
    folder_path = str(folder_path)
    paginator = client.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=folder_path, Delimiter="/")

    files = []
    folders = []

    for page in page_iterator:
        if "Contents" in page:
            files += [
                content["Key"]
                for content in page["Contents"]
                if content["Key"] != folder_path
            ]

        if "CommonPrefixes" in page:
            folders += [prefix["Prefix"] for prefix in page["CommonPrefixes"]]

    if verbose:
        print("Files:")
        for file in files:
            print(f"\t{file}")
        print("\nFolders:")
        for folder in folders:
            print(f"\t{folder}")

    return files, folders


def get_file(client, bucket, file_path):
    try:
        response = client.get_object(Bucket=bucket, Key=str(file_path))
        return response["Body"].read()
    except client.exceptions.NoSuchKey:
        raise FileNotFoundError(f"Could not find this object key: {file_path}")


def download(client, bucket, file_path, local_file_path):
    try:
        client.download_file(bucket, file_path, str(local_file_path))
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            raise FileNotFoundError(f"Could not find this object: {file_path}")
        raise


def set_concurrent_requests(max_requests):
    subprocess.run(
        [
            "aws",
            "configure",
            "set",
            "default.s3.max_concurrent_requests",
            str(max_requests),
        ],
        check=True,
    )
