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

# Derive what this image is being built FROM, because the build uses the working tree rather than
# a git ref: without this, the only record of a deployed image's source is whatever the operator
# happened to keep from this terminal. `git` may legitimately be absent (a tarball build), so a
# missing value is reported as `unknown` and never guessed.
BUILD_REVISION=$(git rev-parse HEAD 2>/dev/null || echo unknown)
if git rev-parse --git-dir >/dev/null 2>&1; then
  if [ -n "$(git status --porcelain 2>/dev/null)" ]; then BUILD_DIRTY=true; else BUILD_DIRTY=false; fi
else
  BUILD_DIRTY=unknown
fi
BUILD_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)

if [ "${BUILD_DIRTY}" = "true" ]; then
  # Not a refusal: shipping from a dirty tree is sometimes deliberate, and blocking it here would
  # only teach people to bypass this script. But the image must not claim a revision whose
  # contents it does not actually contain, so it is stamped dirty and the operator is told now
  # rather than discovering it from a digest that matches nothing reproducible.
  echo "!!! The build tree has UNCOMMITTED CHANGES."
  echo "!!! This image will be stamped ${BUILD_REVISION} with tree=dirty."
  echo "!!! It cannot be rebuilt from that revision alone. Commit first if that matters."
fi

echo "==> Building image with Dockerfile.lite (linux/amd64 for ECS X86_64)..."
echo "    revision ${BUILD_REVISION} (tree dirty=${BUILD_DIRTY})"
docker build --platform linux/amd64 -t "${ECR_REPO}:${IMAGE_TAG}" -f Dockerfile.lite \
  --build-arg "BUILD_REVISION=${BUILD_REVISION}" \
  --build-arg "BUILD_DIRTY=${BUILD_DIRTY}" \
  --build-arg "BUILD_TIME=${BUILD_TIME}" \
  .

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

# THE RECORD. A digest identifies the artifact; a revision identifies the source; only this pairing
# connects them, and until an image carries the stamp above, nothing else can reconstruct it. Paste
# these four lines into the deployment receipt -- they are what makes a later "which build is this?"
# answerable by someone who was not in this terminal.
cat <<RECEIPT

================ RECORD THIS WITH THE DEPLOYMENT ================
  service          ${SERVICE}
  task definition  ${NEW_TD}
  image digest     ${IMAGE}
  source revision  ${BUILD_REVISION}
  build tree       dirty=${BUILD_DIRTY}
  built at         ${BUILD_TIME}
  verify on the wire:  curl -sI <service-url>/health | grep -i x-build-
  verify from ECR:     docker buildx imagetools inspect ${IMAGE} | grep -i revision
=================================================================
RECEIPT
