# The demo recording, docs/demo.gif: a scripted terminal session against the running API.
# Re-record (from the repository root, with the stack up in replay-only mode, so every
# answer is replayed from the baked-in response cache and nothing is paid for):
#   RAG_REPLAY_ONLY=1 docker compose up -d
#   docker build -t rag-demo-vhs docs/demo
#   docker run --rm -v "$PWD/docs:/vhs" --network container:rag-groundedness-eval-api-1 \
#       rag-demo-vhs demo/demo.tape
# `--network container:...` shares the API container's network, so localhost:8000 inside
# the recording is the API, as it is on the host.

Output demo.gif

Set Shell "bash"
Set FontSize 15
Set Width 1280
Set Height 780
Set Padding 18
Set TypingSpeed 18ms
Set Theme "Catppuccin Mocha"

Hide
Type `export PS1='$ '` Enter
Type `ask() { curl -s -X POST localhost:8000/query -H 'Content-Type: application/json' -d "{\"question\": \"$1\"}"; }` Enter
Type "clear" Enter
Show

Type "# The served pipeline: dense retrieval (bge-small, k=5) + Claude Sonnet 5, selected in MLflow" Enter
Type `curl -s localhost:8000/health | jq -c '{status, pipeline: .service.pipeline | {chunking, embedding, generator, prompt_id}}'` Enter
Sleep 3s

Type "# Every answer comes with a groundedness score: the validated LLM judge, claim by claim" Enter
Type "# (replayed from the baked-in response cache: the latency is the service's own, not the model API's)" Enter
Type `ask "How much has the effective tax rate of Corning changed between FY2021 and FY2022?" | jq '{answer, groundedness: .groundedness | {score, supported: "\(.n_supported)/\(.n_document_claims) document claims"}, latency_ms: .latency_ms.total}'` Enter
Sleep 6s

Type "clear" Enter
Type "# A correct answer that the retrieved filings do not support: right answer, wrong reasons" Enter
Type `ask "What is the quantity of restructuring costs directly outlined in AES Corporation's income statements for FY2022? If restructuring costs are not explicitly outlined then state 0." | jq '{answer, score: .groundedness.score, claims: [.groundedness.claims[] | {claim, verdict}]}'` Enter
Sleep 8s

Type "clear" Enter
Type "# How far to trust the score: the judge's agreement with hand labels travels with it" Enter
Type `ask "How much has the effective tax rate of Corning changed between FY2021 and FY2022?" | jq -c '.groundedness.validation | {claim_supported_kappa, answer_fully_grounded_kappa}'` Enter
Sleep 5s

Type "# And the service is observable: per-stage latency and the score distribution" Enter
Type `curl -s localhost:8000/metrics | grep -E '^rag_(requests_total|groundedness_score_(count|sum))'` Enter
Sleep 5s
