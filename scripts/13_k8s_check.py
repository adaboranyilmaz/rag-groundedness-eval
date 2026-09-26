"""Check the Kubernetes manifests (k8s/api.yaml) on a local kind cluster.

  1. cluster     create the kind cluster (k8s/kind-config.yaml) unless it exists, load the
                 API image into it (no registry pull), install metrics-server (the HPA's
                 CPU source; `--kubelet-insecure-tls`, as kind's kubelets have self-signed
                 certificates)
  2. deploy      apply the manifests, wait for the rollout, and answer one benchmark
                 question through the service (port-forward): the served groundedness must
                 equal the Phase 5 evaluation's
  3. autoscale   run a load generator inside the cluster (a pod from the same image, sending
                 benchmark questions to the service's cluster address, a new connection per
                 request so kube-proxy spreads them over the ready pods); record the HPA's
                 CPU reading and the replica count every 10 s, until the deployment has
                 scaled out or the time limit passes; then each pod's request count from
                 its own /metrics, so it shows whether the added pods served traffic
  -> results/metrics/k8s_check.json. Replay-only (the configmap), so no API spend.

Requires docker, kind and kubectl on PATH. Single run on one machine: the autoscaling
timeline says the manifests work, not how a real cluster would perform.

Usage:
    uv run python scripts/13_k8s_check.py            # leaves the cluster running
    uv run python scripts/13_k8s_check.py --delete   # and deletes it at the end
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import yaml

CLUSTER = "rag"
NAMESPACE = "rag"
MANIFESTS = Path("k8s/api.yaml")
KIND_CONFIG = Path("k8s/kind-config.yaml")
SERVING_CONFIG = Path("configs/serving.yaml")
OUT_PATH = Path("results/metrics/k8s_check.json")
METRICS_SERVER = (
    "https://github.com/kubernetes-sigs/metrics-server/releases/download/v0.8.0/components.yaml"
)
LOADGEN_USERS = 8
AUTOSCALE_LIMIT_S = 600
POLL_S = 10


def run(*cmd: str, check: bool = True, timeout: float = 600, input: str | None = None) -> str:
    r = subprocess.run(
        list(cmd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        input=input,
        check=False,
    )
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed ({r.returncode}): {r.stderr.strip()}")
    return r.stdout


def kubectl(*args: str, **kw) -> str:
    return run("kubectl", "--context", f"kind-{CLUSTER}", *args, **kw)


def image_name() -> str:
    doc = next(d for d in yaml.safe_load_all(MANIFESTS.read_text(encoding="utf-8"))
               if d["kind"] == "Deployment")  # fmt: skip
    return doc["spec"]["template"]["spec"]["containers"][0]["image"]


def benchmark() -> tuple[list[dict], dict[str, dict]]:
    cfg = yaml.safe_load(SERVING_CONFIG.read_text(encoding="utf-8"))
    lines = Path(cfg["arm_traces"]).read_text(encoding="utf-8").splitlines()
    traces = [json.loads(x) for x in lines]
    ids = {t["trace_id"] for t in traces}
    evals = {}
    for line in Path(cfg["arm_evaluations"]).read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec["trace_id"] in ids:
            evals[rec["trace_id"]] = rec
    return traces, evals


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def step_cluster(image: str) -> dict:
    t0 = time.perf_counter()
    if CLUSTER not in run("kind", "get", "clusters").split():
        run("kind", "create", "cluster", "--name", CLUSTER, "--config", str(KIND_CONFIG),
            timeout=900)  # fmt: skip
    t1 = time.perf_counter()
    run("kind", "load", "docker-image", image, "--name", CLUSTER, timeout=1800)
    t2 = time.perf_counter()
    kubectl("apply", "-f", METRICS_SERVER)
    patch = '[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'  # noqa: E501
    args = kubectl("-n", "kube-system", "get", "deploy", "metrics-server",
                   "-o", "jsonpath={.spec.template.spec.containers[0].args}")  # fmt: skip
    if "--kubelet-insecure-tls" not in args:
        kubectl("-n", "kube-system", "patch", "deploy", "metrics-server", "--type=json",
                "-p", patch)  # fmt: skip
    kubectl("-n", "kube-system", "rollout", "status", "deploy/metrics-server", "--timeout=300s")
    return {
        "create_s": round(t1 - t0, 1),
        "image_load_s": round(t2 - t1, 1),
        "metrics_server": METRICS_SERVER,
        "versions": {
            "kind": run("kind", "version").strip(),
            "kubectl_server": json.loads(kubectl("version", "-o", "json"))["serverVersion"][
                "gitVersion"
            ],
        },
    }


def step_deploy(traces: list[dict], evals: dict) -> dict:
    t0 = time.perf_counter()
    kubectl("apply", "-f", str(MANIFESTS))
    kubectl("-n", NAMESPACE, "rollout", "status", "deploy/rag-api", "--timeout=600s")
    ready_s = time.perf_counter() - t0
    scored = [t for t in traces if (evals[t["trace_id"]]["groundedness"] or {}).get("groundedness")
              is not None]  # fmt: skip
    trace = scored[0]
    port = free_port()
    pf = subprocess.Popen(
        ["kubectl", "--context", f"kind-{CLUSTER}", "-n", NAMESPACE, "port-forward",
         "svc/rag-api", f"{port}:80"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )  # fmt: skip
    try:
        body = None
        for _ in range(30):
            try:
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/query",
                    json.dumps({"question": trace["question"]["question"]}).encode(),
                    {"Content-Type": "application/json"},
                )
                body = json.loads(urllib.request.urlopen(req, timeout=60).read())
                break
            except OSError:
                time.sleep(1)
    finally:
        pf.terminate()
    if body is None:
        raise RuntimeError("no answer through the port-forward")
    expected = evals[trace["trace_id"]]["groundedness"]["groundedness"]
    return {
        "rollout_s": round(ready_s, 1),
        "query": {
            "trace_id": trace["trace_id"],
            "status": body["status"],
            "groundedness": body["groundedness"]["score"],
            "evaluated_groundedness": expected,
            "matches_evaluation": body["groundedness"]["score"] == expected,
            "all_cached": all(c["cached"] for c in body["llm_calls"]),
        },
    }


LOADGEN = r"""
import json, random, threading, time, urllib.request
qs = json.load(open("/questions/questions.json"))
def user(i):
    rng = random.Random(i)
    while True:
        req = urllib.request.Request("http://rag-api.rag.svc/query",
            json.dumps({"question": rng.choice(qs)}).encode(), {"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=120).read()
        except Exception as e:
            print("error", type(e).__name__, flush=True); time.sleep(1)
for i in range(N):
    threading.Thread(target=user, args=(i,), daemon=True).start()
while True:
    time.sleep(60)
"""


def pod_requests() -> dict[str, float]:
    pods = kubectl("-n", NAMESPACE, "get", "pods", "-l", "app=rag-api",
                   "-o", "jsonpath={.items[*].metadata.name}").split()  # fmt: skip
    out = {}
    for pod in pods:
        code = "import urllib.request as u;print(u.urlopen('http://localhost:8000/metrics').read().decode())"  # noqa: E501
        text = kubectl("-n", NAMESPACE, "exec", pod, "--", "python", "-c", code, check=False)
        out[pod] = sum(
            float(line.rsplit(" ", 1)[1])
            for line in text.splitlines()
            if line.startswith("rag_requests_total{")
        )
    return out


def step_autoscale(traces: list[dict], image: str) -> dict:
    questions = [t["question"]["question"] for t in random.Random(0).sample(traces, 50)]
    cm = kubectl("-n", NAMESPACE, "create", "configmap", "loadgen-questions",
                 "--from-literal=questions.json=" + json.dumps(questions),
                 "--dry-run=client", "-o", "yaml")  # fmt: skip
    kubectl("apply", "-f", "-", input=cm)
    loadgen = LOADGEN.replace("range(N)", f"range({LOADGEN_USERS})")
    pod = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": "loadgen", "namespace": NAMESPACE},
        "spec": {
            "restartPolicy": "Never",
            "containers": [
                {
                    "name": "loadgen",
                    "image": image,
                    "imagePullPolicy": "IfNotPresent",
                    "command": ["python", "-c", loadgen],
                    "volumeMounts": [{"name": "q", "mountPath": "/questions"}],
                    "resources": {"requests": {"cpu": "100m"}},
                }
            ],
            "volumes": [{"name": "q", "configMap": {"name": "loadgen-questions"}}],
        },
    }
    kubectl("-n", NAMESPACE, "delete", "pod", "loadgen", "--ignore-not-found", "--wait=true")
    kubectl("apply", "-f", "-", input=json.dumps(pod))
    timeline = []
    t0 = time.perf_counter()
    scaled_at = None
    try:
        while time.perf_counter() - t0 < AUTOSCALE_LIMIT_S:
            hpa = json.loads(kubectl("-n", NAMESPACE, "get", "hpa", "rag-api", "-o", "json"))
            st = hpa.get("status", {})
            cpu = next(
                (m["resource"]["current"].get("averageUtilization")
                 for m in st.get("currentMetrics") or [] if m.get("type") == "Resource"),
                None,
            )  # fmt: skip
            dep = json.loads(kubectl("-n", NAMESPACE, "get", "deploy", "rag-api", "-o", "json"))
            ready = dep["status"].get("readyReplicas", 0)
            timeline.append(
                {
                    "t_s": round(time.perf_counter() - t0),
                    "cpu_utilization_pct": cpu,
                    "desired_replicas": st.get("desiredReplicas"),
                    "ready_replicas": ready,
                }
            )
            print(f"  t={timeline[-1]['t_s']:>4}s cpu={cpu}% desired="
                  f"{st.get('desiredReplicas')} ready={ready}")  # fmt: skip
            if scaled_at is None and ready > 1:
                scaled_at = timeline[-1]["t_s"]
            if scaled_at is not None and timeline[-1]["t_s"] - scaled_at >= 60:
                break  # scaled out, and the new pods have served for a minute
            time.sleep(POLL_S)
        per_pod = pod_requests()
    finally:
        kubectl("-n", NAMESPACE, "delete", "pod", "loadgen", "--ignore-not-found", "--wait=false")
    return {
        "load": f"{LOADGEN_USERS} closed-loop users in a pod of the same image, to the "
        "service's cluster address; 50 benchmark questions (seed 0)",
        "hpa": "cpu averageUtilization 70% of the 500m request, 1-3 replicas",
        "scaled_out": scaled_at is not None,
        "first_extra_replica_ready_s": scaled_at,
        "max_ready_replicas": max((p["ready_replicas"] for p in timeline), default=0),
        "timeline": timeline,
        "requests_per_pod": per_pod,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete", action="store_true", help="delete the cluster at the end")
    args = parser.parse_args()
    missing = [b for b in ("docker", "kind", "kubectl") if shutil.which(b) is None]
    if missing:
        sys.exit(f"not on PATH: {', '.join(missing)}")
    image = image_name()
    traces, evals = benchmark()
    print("1/3 cluster")
    cluster = step_cluster(image)  # noqa: E702
    print("2/3 deploy")
    deploy = step_deploy(traces, evals)  # noqa: E702
    print("3/3 autoscale")
    autoscale = step_autoscale(traces, image)  # noqa: E702
    report = {
        "meta": {
            "single_run": True,
            "note": "local kind cluster on one machine; replay-only (no API calls)",
            "image": image,
            "manifests": MANIFESTS.as_posix(),
        },
        "cluster": cluster,
        "deploy": deploy,
        "autoscale": autoscale,
        "passed": deploy["query"]["matches_evaluation"]
        and deploy["query"]["all_cached"]
        and autoscale["scaled_out"],
    }
    OUT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT_PATH}; passed={report['passed']}")
    if args.delete:
        run("kind", "delete", "cluster", "--name", CLUSTER, timeout=300)


if __name__ == "__main__":
    main()
