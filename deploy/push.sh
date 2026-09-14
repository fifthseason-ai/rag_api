#!/usr/bin/env bash
set -euo pipefail

AWS_ACCOUNT_ID="861208159576"
AWS_REGION="us-east-1"
ECR_REPO="fifthseason/rag-api"
CLUSTER="PROD-API"
SERVICE="fifthseason-rag-api-service"
CONTAINER="fifhtseason-rag-api"
IMAGE_TAG="${1:-latest}"

ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}"

# Move to project root (one level up from deploy/)
cd "$(dirname "$0")/.."

echo "==> Logging in to ECR..."
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

echo "==> Building image with Dockerfile.lite (linux/amd64 for ECS X86_64)..."
docker build --platform linux/amd64 -t "${ECR_REPO}:${IMAGE_TAG}" -f Dockerfile.lite .

echo "==> Tagging image..."
docker tag "${ECR_REPO}:${IMAGE_TAG}" "${ECR_URI}:${IMAGE_TAG}"

echo "==> Pushing to ECR..."
docker push "${ECR_URI}:${IMAGE_TAG}"

echo "==> Resolving the digest that was just pushed..."
DIGEST=$(aws ecr describe-images --region "${AWS_REGION}" \
  --repository-name "${ECR_REPO}" --image-ids "imageTag=${IMAGE_TAG}" \
  --query 'imageDetails[0].imageDigest' --output text)
[ -n "${DIGEST}" ] && [ "${DIGEST}" != "None" ] \
  || { echo "could not resolve the digest for tag ${IMAGE_TAG}"; exit 1; }
IMAGE="${ECR_URI}@${DIGEST}"
echo "    ${IMAGE}"

# Derive the new revision from what the service is running RIGHT NOW, not from a
# checked-in copy: environment and secrets edited outside the repo survive, and the
# image is the only field that moves.
echo "==> Registering a task definition off the live one..."
CURRENT_TD=$(aws ecs describe-services --region "${AWS_REGION}" --cluster "${CLUSTER}" \
  --services "${SERVICE}" --query 'services[0].taskDefinition' --output text)
[ -n "${CURRENT_TD}" ] && [ "${CURRENT_TD}" != "None" ] \
  || { echo "could not read the task definition of ${SERVICE}"; exit 1; }
echo "    from ${CURRENT_TD}"

NEW_TD_JSON=$(aws ecs describe-task-definition --region "${AWS_REGION}" \
  --task-definition "${CURRENT_TD}" --query 'taskDefinition' --output json \
  | IMAGE="${IMAGE}" CONTAINER="${CONTAINER}" python3 -c '
import json, os, sys

td = json.load(sys.stdin)
for field in ("taskDefinitionArn", "revision", "status", "requiresAttributes",
              "compatibilities", "registeredAt", "registeredBy", "deregisteredAt"):
    td.pop(field, None)

name = os.environ["CONTAINER"]
target = [c for c in td["containerDefinitions"] if c["name"] == name]
if not target:
    sys.exit("container %s is not in family %s" % (name, td.get("family")))
target[0]["image"] = os.environ["IMAGE"]

json.dump(td, sys.stdout)
')

NEW_TD=$(aws ecs register-task-definition --region "${AWS_REGION}" \
  --cli-input-json "${NEW_TD_JSON}" \
  --query 'taskDefinition.taskDefinitionArn' --output text)
echo "    ${NEW_TD}"

echo "==> Updating ${SERVICE}..."
aws ecs update-service --region "${AWS_REGION}" --cluster "${CLUSTER}" \
  --service "${SERVICE}" --task-definition "${NEW_TD}" >/dev/null

echo "==> Waiting for the service to stabilize..."
aws ecs wait services-stable --region "${AWS_REGION}" --cluster "${CLUSTER}" --services "${SERVICE}"

echo "==> Done. ${SERVICE} now runs ${IMAGE}"
