#!/bin/bash
set -e

# server

TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
if [ -z "$TOKEN" ]; then
  echo "Error: Unable to fetch IMDSv2 token. Ensure IMDSv2 is enabled."
  exit 1
fi

INSTANCE_ID=$(curl -H "X-aws-ec2-metadata-token: $TOKEN" -s http://169.254.169.254/latest/meta-data/instance-id)
REGION=$(curl -H "X-aws-ec2-metadata-token: $TOKEN" -s http://169.254.169.254/latest/meta-data/placement/region)

echo "----->>> Stopping instance. <<<-----"
sudo pkill -HUP sshd
aws ec2 stop-instances --instance-ids "$INSTANCE_ID" --region "$REGION" $DRY_RUN
