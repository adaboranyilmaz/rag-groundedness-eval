# Kubernetes

`api.yaml` deploys the API: a namespace, a configmap, the deployment (startup, readiness and
liveness probes; a per-pod state volume), a ClusterIP service, and a horizontal pod
autoscaler on CPU. The image carries the serving bundle, so a pod needs no data volume.
`scripts/13_k8s_check.py` verifies it on a local kind cluster: rollout, one benchmark
answer through the service whose groundedness must equal the evaluated one, and a scale-out
under load generated inside the cluster (`results/metrics/k8s_check.json`).

```bash
kind create cluster --name rag --config k8s/kind-config.yaml
kind load docker-image ghcr.io/adaboranyilmaz/rag-groundedness-eval-api:0.2.0 --name rag
kubectl apply -f k8s/api.yaml
kubectl -n rag port-forward svc/rag-api 8080:80
# or all of it, plus metrics-server and the autoscaling check:
uv run python scripts/13_k8s_check.py --delete
```

## What would change on a real cluster

**State is per pod here, and it should not be.** Each pod keeps its request log, the
responses it paid for and its spend ledger in an `emptyDir`, so the spend cap is per pod
(three replicas can spend three times the cap), a new pod re-pays for questions another pod
already answered, and the logs die with the pod. A real deployment would keep the response
cache and the spend budget in a shared store (Redis or a database, with the budget
decremented atomically), and write the request log to stdout for the cluster's log shipper
instead of a file.

**CPU is the wrong scaling signal for live traffic.** In the replay-only check a request is
all CPU (query embedding, search, parsing), so CPU tracks load. With the API key set, a
request spends seconds waiting on three model calls while its pod's CPU idles, and a
CPU-based autoscaler would not add pods however long the queue grew. The API exports
`rag_requests_in_flight`; scaling on it (through the Prometheus adapter or KEDA), or on
request latency, follows the real bottleneck. The ceiling after that is the model API's
rate limit, not the pod count.

**The index is baked into the image.** Adding filings means building and rolling out a new
image, which is acceptable for a fixed benchmark corpus and wrong for filings that arrive
every quarter. The vector store would become a service (the Qdrant backend already exists,
behind the same adapter) fed by the ingestion pipeline, with the image carrying only code
and model weights.

**Everything in front of the service is missing.** An ingress with TLS, authentication, and
per-client rate limits; secrets from a secret manager instead of `kubectl create secret`;
the image pulled from the registry by digest, not loaded into the node by hand; a
Prometheus `ServiceMonitor` for `/metrics`; a PodDisruptionBudget and spreading replicas
across nodes. Qdrant and the MLflow server, which the compose stack runs next to the API,
would be separately managed services.
