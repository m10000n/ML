#!/bin/bash
set -e

AWS_TEST_MODE=false

case "$#" in
0) ;;
1)
	case $1 in
	--test)
		AWS_TEST_MODE=true
		;;
	*)
		echo "Usage: $0 [--test]"
		exit 1
		;;
	esac
	;;
esac

TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
if [ -z "$TOKEN" ]; then
	echo "Error: Unable to fetch IMDSv2 token. Ensure IMDSv2 is enabled."
	exit 1
fi

INSTANCE_ID=$(curl -H "X-aws-ec2-metadata-token: $TOKEN" -s http://169.254.169.254/latest/meta-data/instance-id)
REGION=$(curl -H "X-aws-ec2-metadata-token: $TOKEN" -s http://169.254.169.254/latest/meta-data/placement/region)

COMMAND="aws ec2 stop-instances --instance-ids $INSTANCE_ID --region $REGION"
if [ "$AWS_TEST_MODE" == true ]; then
	set +e
	OUTPUT=$(eval "$COMMAND --dry-run" 2>&1)
	EXIT_CODE=$?
	set -e
	if echo "$OUTPUT" | grep -q "DryRunOperation"; then
		exit 0
	else
		echo "$OUTPUT" >&2
		exit $EXIT_CODE
	fi
else
	echo "----->>> Stopping instance. <<<-----"
	sudo pkill -HUP sshd
	eval "$COMMAND"
fi
