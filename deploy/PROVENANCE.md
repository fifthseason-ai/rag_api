# Which image is running, and what was it built from?

Two facts identify a deployment, and they live in different places:

- the **image digest** — what ECS actually runs, recorded by the registry;
- the **source revision** — what that image was built from.

Until this change, nothing in this repo connected them. `deploy/push.sh` builds the operator's
**working tree** rather than a git ref, the image carried no label, the tag defaults to `latest`,
and the workflows that build images push *dev* images to ghcr rather than the production registry.
So the pairing existed only in whatever the operator happened to keep from one terminal.

## What this cannot do

**It cannot identify an image built before it shipped — task definition 24 included.** Those images
carry no stamp, `/health` on them reports no build, and nothing here changes that retroactively.
For an existing deployment the only routes to identity are the operator's own record of the push,
or behaviour unique to a known change. Do not read a stamped `/health` on a *future* build as
evidence about an earlier one.

## What the operator does

Nothing new during a normal deploy. `deploy/push.sh` derives the revision and tree state itself and
ends by printing:

```
================ RECORD THIS WITH THE DEPLOYMENT ================
  service          fifthseason-rag-api-service
  task definition  arn:aws:ecs:...:task-definition/...:25
  image digest     ...dkr.ecr...amazonaws.com/fifthseason/rag-api@sha256:...
  source revision  36d4fb6da44df2de12617ce83fa5e29bc50b5acc
  build tree       dirty=false
  built at         2026-09-18T19:00:00Z
  ...
=================================================================
```

**Paste those lines into the deployment receipt.** That block *is* the pairing; everything below is
how to recover it later if it was not recorded.

### If the tree is dirty, the production build is REFUSED

```
REFUSED: the build tree has UNCOMMITTED CHANGES.
  A production image must be rebuildable from the revision it names.
  Commit (or stash) first:   git status --porcelain
  Deliberate exception:      PUSH_ALLOW_DIRTY=1 ./deploy/push.sh latest
```

Nothing is built or pushed. An image built from uncommitted edits names a revision whose contents
it does not contain, so it cannot be rebuilt, reviewed or reasoned about afterwards, and the digest
in the receipt would point at a source state that exists nowhere.

**The exception exists on purpose.** An emergency where the fix must ship before it can be
committed is real, and an absolute ban would simply be worked around with a direct `docker build`
— which loses the stamp entirely, the opposite of what this protects. Taking the exception is a
deliberate act: the image is still stamped `tree=dirty`, so the receipt stays honest about what
was shipped. **Record that in the deployment receipt when it happens.**

A build with no git metadata at all (a tarball, say) is allowed and stamped `unknown`, with a
warning — but nothing will tie that image to a source state, so prefer building from a checkout.

## Reading it back, without the operator's terminal

**From the registry (needs ECR access, works on a deployed image):**

```bash
docker buildx imagetools inspect <account>.dkr.ecr.<region>.amazonaws.com/fifthseason/rag-api@sha256:<digest>
# or
aws ecr describe-images --repository-name fifthseason/rag-api --image-ids imageDigest=sha256:<digest>
```

The labels carry it: `org.opencontainers.image.revision`, `org.opencontainers.image.created`,
`ai.fifthseason.build.dirty`.

**From the wire (needs only a URL, no credentials — `/health` is public):**

```bash
curl -sI https://<service-url>/health | grep -i 'x-service-name\|x-build-'
curl -s  https://<service-url>/health          # same values in the body, under "build"
```

```
X-Service-Name: rag_api
X-Build-Revision: 36d4fb6da44d
X-Build-Tree: clean
```

**Which ECS is running right now:**

```bash
aws ecs describe-services --cluster PROD-API --services fifthseason-rag-api-service \
  --query 'services[0].taskDefinition'
aws ecs describe-task-definition --task-definition <arn> \
  --query 'taskDefinition.containerDefinitions[].image'
```

## What the values mean

| Value | Meaning |
|---|---|
| `X-Service-Name: rag_api` | this reply came from this service — **not** from an edge, proxy or load balancer answering on its behalf |
| `X-Build-Revision` | first 12 characters of the source revision, or `unknown` |
| `X-Build-Tree` | `clean` · `dirty` (built with uncommitted changes) · `unknown` (never stamped) |

`unknown` is reported wherever the value was not recorded. It is never rendered as `clean`: an
unstamped build and a clean one are different claims.

**Every response carries the headers**, including refusals (401/403), the fail-closed 500 when
`JWT_SECRET` is absent, and the 404 for a path that matches no route. That is deliberate — those
are exactly the answers a caller cannot otherwise attribute, and a 404 from this service used to be
indistinguishable from a 404 produced by an edge that never reached it. A response with **no**
`X-Service-Name` did not come from this service (or is a crash rendered above the middleware stack,
which is itself worth knowing).

## Developer builds

`docker build -f Dockerfile.lite .` with no arguments works exactly as before and reports
`unknown` for all three values, which is the truth about such an image. The build args are
optional:

```bash
docker build -f Dockerfile.lite \
  --build-arg BUILD_REVISION=$(git rev-parse HEAD) \
  --build-arg BUILD_DIRTY=$([ -n "$(git status --porcelain)" ] && echo true || echo false) \
  --build-arg BUILD_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ) .
```

The args sit **after** the dependency layers, so stamping does not invalidate the pip cache.
