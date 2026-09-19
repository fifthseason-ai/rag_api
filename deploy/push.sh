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
# Capture the STATUS of the measurement, not only its output. `git status --porcelain` prints
# nothing for a clean tree -- and also prints nothing when it FAILS, which the previous form read
# as the same thing and stamped `clean` on a tree it had not managed to measure. That is reachable:
# `git rev-parse HEAD` does not touch the index, `git status` does, so a stale `.git/index.lock`
# from an interrupted command or a concurrent IDE process fails the second while the first still
# answers. `unknown` is the third value for exactly this case and it must be REACHED, not merely
# defined.
if git rev-parse --git-dir >/dev/null 2>&1; then
  if PORCELAIN=$(git status --porcelain 2>/dev/null); then
    if [ -n "${PORCELAIN}" ]; then BUILD_DIRTY=true; else BUILD_DIRTY=false; fi
  else
    BUILD_DIRTY=unknown
  fi
else
  BUILD_DIRTY=unknown
fi
BUILD_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# A PRODUCTION image is refused from a dirty tree. This script deploys to the production cluster,
# and an image built from uncommitted edits names a revision whose contents it does not contain:
# it cannot be rebuilt, reviewed or reasoned about afterwards, and the digest recorded in the
# receipt would point at a source state that exists nowhere.
#
# The refusal has a deliberate, explicit escape hatch rather than being absolute: an emergency
# where the fix must ship before it can be committed is a real situation, and a ban with no way
# through would be worked around by calling `docker build` directly -- which loses the stamp
# entirely, the opposite of what this exists to protect. Taking the hatch is a decision the
# operator makes on purpose, and the image still says `tree=dirty`, so the receipt stays honest.
# `unknown` is refused on the same terms as `true`. A tree that could not be measured is not a
# clean tree -- it is a tree nobody has checked, and stamping a production image `clean` on that
# basis is the fabrication this whole change exists to prevent. Reported separately from `true`
# so the operator knows which of the two they are looking at.
if [ "${BUILD_DIRTY}" = "unknown" ]; then
  if [ "${PUSH_ALLOW_DIRTY:-0}" = "1" ]; then
    echo "!!! Building a PRODUCTION image whose tree state is UNKNOWN because PUSH_ALLOW_DIRTY=1."
    echo "!!! The image will be stamped tree=unknown. NOTHING has verified that ${BUILD_REVISION}"
    echo "!!! describes its contents. Record that in the deployment receipt."
  else
    echo "REFUSED: the build tree's state could NOT BE MEASURED." >&2
    echo "  This is not the same as clean -- git could not be asked, or failed when it was." >&2
    echo "  No git at all:         the build cannot prove what it contains." >&2
    echo "  A failed 'git status': run it by hand; a stale .git/index.lock is the usual cause." >&2
    echo "  Deliberate exception:  PUSH_ALLOW_DIRTY=1 $0 ${IMAGE_TAG}" >&2
    exit 1
  fi
fi

if [ "${BUILD_DIRTY}" = "true" ]; then
  if [ "${PUSH_ALLOW_DIRTY:-0}" = "1" ]; then
    echo "!!! Building a PRODUCTION image from a DIRTY tree because PUSH_ALLOW_DIRTY=1."
    echo "!!! The image will be stamped ${BUILD_REVISION} with tree=dirty and CANNOT be rebuilt"
    echo "!!! from that revision alone. Record that in the deployment receipt."
  else
    echo "REFUSED: the build tree has UNCOMMITTED CHANGES." >&2
    echo "  A production image must be rebuildable from the revision it names." >&2
    echo "  Commit (or stash) first:   git status --porcelain" >&2
    echo "  Deliberate exception:      PUSH_ALLOW_DIRTY=1 $0 ${IMAGE_TAG}" >&2
    exit 1
  fi
fi

if [ "${BUILD_REVISION}" = "unknown" ]; then
  # No git metadata at all (a tarball build, say). Allowed, because it is sometimes the only way
  # to ship, but never silently: an image nobody can trace back is exactly the situation this
  # change exists to end.
  echo "!!! No git revision available: this image will be stamped 'unknown' and NOTHING will tie"
  echo "!!! it to a source state. Prefer building from a checkout."
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
