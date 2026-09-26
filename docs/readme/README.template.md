# Auditing Answer Groundedness in Retrieval-Augmented Question Answering over Financial Filings

*Does a retrieval-augmented system get the right answer for the right reasons?*

Accuracy on a question-answering benchmark says whether a system was right, not whether it was right for the right reasons. This project measures the second for retrieval-augmented question answering over SEC filings, on the {{fb.n_questions}} FinanceBench questions. A reproducible harness separates retrieval failure from generation failure with an oracle-context condition, scores groundedness claim by claim with an LLM judge validated against blind human labels, and serves that score with every answer. The central finding is that correctness and groundedness diverge: with the evidence guaranteed, {{q2.or_sonnet.ugc_pct}} of Claude Sonnet 5's correct answers are not fully supported by it, yet the judge measuring this agrees with a human only at κ = {{ja.human_judge.answer_k}} per answer, and RAGAS does no better (κ = {{rg.human_ragas.k}}).

## Key Findings

1. **Retrieval, not generation, is the binding constraint.** The best of {{ret.n_cells}} retrieval configurations reaches recall@5 = {{ret.best.recall5}}: some gold evidence is in its top five for only {{q1.sonnet.n_q_gt0}} of the {{ret.n_scored}} questions with aligned evidence, and the right filing is missing entirely for {{ret.best.miss_filing_pct}} of them. For the questions whose evidence retrieval misses, supplying it raises Claude Sonnet 5's answer rate by {{q1.pair_sonnet.d_answered_pp}} {{q1.pair_sonnet.d_answered_ci}} on the same questions. Under realistic retrieval the selected pipeline declines {{rep.v1.declined_pm}} of all questions (mean ± std over three runs). The retrieval method explains {{ret.share.method}} of the variance in recall@5 across the grid, and the expected headline, that chunking matters more than the embedding model, holds only because method dwarfs both: within dense retrieval it reverses.

2. **Almost a third of correct answers fail the groundedness check, and about half of those verdicts are the judge's mistakes.** With oracle context, {{q2.or_sonnet.ugc}} {{q2.or_sonnet.ugc_ci}} of Claude Sonnet 5's correct answers contain a claim the excerpts do not support. A review of {{rev.n}} of them by the author found the judge right in {{rev.right}} {{rev.right_ci}}; of the {{rev.wrong}} it got wrong, {{rev.broke_rules}} broke its own written rules and {{rev.rule_disputed}} applied a rule the author disputes. The quadrant is concentrated on questions that ask for a judgement (post hoc: {{qt.domain}} of correct answers there, against {{qt.metrics}} for computed metrics), whose evaluative conclusions the rules count as unsupported. Accuracy alone would report every one of these answers as a success; the judge alone would roughly double the size of the problem.

3. **LLM raters agree with each other far more than with a person.** The project judge agrees with itself on a re-run at κ = {{ja.retest.claim_k}} and with a second model (Haiku 4.5) at κ = {{ja.second.claim_k}}, but with the author's blind labels only at κ = {{ja.human_judge.claim}} per claim and {{ja.human_judge.answer}} per answer. RAGAS faithfulness, run on the same model through its own prompts and its own statement split, lands in the same place: κ = {{rg.human_ragas.kci}} against the author, {{rg.diff}} {{rg.diff_ci}} from the project judge. Two independent implementations meeting the same ceiling suggests the ceiling is the definition of "supported", not the implementation. Agreement between model raters is evidence of consistency, not of validity.

4. **Good retrieval predicts grounded generation only weakly.** Spearman ρ between a question's recall@5 and the groundedness of its answers is {{q1.sonnet.rho}} {{q1.sonnet.rho_ci}} for Claude Sonnet 5 and {{q1.qwen.rho}} {{q1.qwen.rho_ci}} for Qwen2.5 3B. Retrieving the evidence raises Sonnet's groundedness from {{q1.sonnet.g_eq0}} to {{q1.sonnet.g_gt0}}, but it moves the answer rate more, from {{q1.sonnet.ans_eq0}} to {{q1.sonnet.ans_gt0}}: much of retrieval's effect is on whether the model answers at all.

5. **Asking for citations changes nothing measurable.** Requiring the model to quote the excerpt it used moves Claude Sonnet 5's groundedness by {{q5.or_sonnet.v2.g}} {{q5.or_sonnet.v2.g_ci}} and its citation precision by {{q5.or_sonnet.v2.cp}} {{q5.or_sonnet.v2.cp_ci}} under oracle context, against a pre-registered prediction of a real improvement. Over three runs of the realistic pipeline, the zero-shot and citation-required prompts differ in the share of questions answered correctly and fully grounded by {{rep.pooled.diff_pp}} {{rep.pooled.diff_ci}}. Chain-of-thought is the only variant with an effect, and only for the 3B model ({{q5.or_qwen.v3.g}} in groundedness).

6. **The prompt that permits declining also stops the model correcting false premises.** On ten questions built on a figure that does not exist, Claude Sonnet 5 rejects the premise in {{q4.prem.sonnet.v1_k}}, {{q4.prem.sonnet.v2_k}} and {{q4.prem.sonnet.v3_k}} of ten answers under the first three prompts and in {{q4.prem.sonnet.v4_k}} of ten under the abstention prompt, which declines instead. It never answered an unanswerable question from its own knowledge ({{q4.mem.sonnet}} of answers), which Qwen2.5 3B did in {{q4.mem.qwen}}.

![Correct against grounded, per generator and context](results/plots/correct_vs_grounded.png)
*Correctness and groundedness are separate properties: the off-diagonal cells are populated for every generator and context. With the evidence guaranteed (bottom left), {{q2.or_sonnet.cu}} of Claude Sonnet 5's {{q2.or_sonnet.correct}} correct answers still fail the groundedness check, while under realistic retrieval most of Qwen2.5 3B's answers are wrong and unsupported at once.*

## Method

### Data

FinanceBench (Islam et al., 2023) supplies {{fb.n_questions}} human-written questions over {{fb.n_docs}} public-company filings, each with a gold answer and gold evidence text. The filings are fetched from SEC EDGAR as the HTML the SEC serves for modern 10-K and 10-Q reports; {{corpus.ingested}} are in scope ({{corpus.pages}} pages), and the rest are earnings releases and other documents outside the corpus. {{ret.n_in_corpus}} questions have their filing in the corpus and {{ret.n_scored}} of those have gold evidence aligned to the parsed text, which is what retrieval is scored against. HTML filings use `<table>` for page layout as often as for data, so parsing is a custom `lxml` walker that tells the two apart rather than a generic HTML-to-text library, and every chunk keeps its `(doc_id, page, section, char_span)`. A further {{adv.n_questions}} adversarial questions were written by hand, ten in each of four categories: answerable from one filing, answerable only by combining two, plausible but unanswerable from the corpus, and built on a false premise.

### Retrieval

The retrieval grid crosses three chunking strategies (fixed-size, `chunk_size=512, overlap=64` characters; recursive-structural, which respects sections; table-aware, which keeps each table whole), three embedding models (`bge-small-en-v1.5`, `all-MiniLM-L6-v2`, `bge-base-en-v1.5`) and four methods (dense FAISS search, BM25, hybrid by reciprocal rank fusion, and hybrid re-ranked by `bge-reranker-base`), {{ret.n_cells}} cells in all. Every cell is scored against the gold evidence at `k=5` on recall, precision, MRR, nDCG and span recall, and the winner by recall@5 was fixed before any answer was generated. FAISS and Qdrant sit behind one adapter and return equivalent top-k, so the backend is a configuration choice.

### Generation

Two generators answer every question: Claude Sonnet 5 through the API with thinking off (the model rejects sampling parameters, so its answers are reproducible from an on-disk response cache rather than from a seed), and Qwen2.5 3B Instruct locally through Ollama at `temperature=0` with a fixed seed, chosen because a 7B model does not fit the 4 GB card. Four versioned prompt files are compared: zero-shot, citation-required (the answer must quote the excerpt it used), chain-of-thought, and abstention-encouraged. Each runs under two context conditions. **Retrieved** uses the winning retriever over the whole corpus for all {{fb.n_questions}} questions. **Oracle** builds the five excerpts from the chunks that contain the gold evidence, topped up with the filing's best-scoring other chunks and ordered by retrieval score, for the {{ret.n_scored}} questions with aligned evidence. Holding generation fixed while guaranteeing the evidence separates retrieval failure from generation failure by construction. Every one of the {{gen.n_traces}} generations is a trace (question, excerpts, prompt version and hash, model, raw output, parsed answer and citations) that replays from the cache.

### Evaluation

Four metric families score every answer. *Correctness* compares numeric answers with the gold at a relative tolerance of `rel_tol=0.01` and separates unit errors (a figure in thousands read as millions) from wrong answers; other answers are graded by an LLM judge, which agrees with the numeric matcher at κ = {{ev.crosscheck_k}} where both apply. *Groundedness* is written from scratch before any library was used: a judge splits the answer into atomic claims, classed as about the company, about the excerpts, or general finance, and a second call checks each claim against the excerpts the generator saw; the score is supported company claims over company claims, and an answer is *fully grounded* when all are supported. *Citation precision* is the share of cited excerpts that support a claim. *Abstention* records whether the model declined, by a rule and by the judge (κ = {{ev.abstention_k}} between the two). The judge is Claude Sonnet 5 with thinking off, constrained to a JSON schema and run through the Message Batches API. It was validated on {{ja.n_answers}} answers ({{ja.n_claims}} claims) that the author labelled blind to the judge, the gold answer, the citations and the generator. The reliability analysis was pre-registered: every prediction was written down before any number was computed, and {{prereg.unexpected}} of the {{prereg.checked}} that could be checked turned out wrong.

### External check and repeated runs

Two additions test the harness from outside. RAGAS faithfulness (Es et al., 2023; `ragas` 0.4.3) scores the same {{ja.n_answers}} labelled answers on the same model as the project judge, so a difference reflects the method, not the model; only the transport is replaced, since RAGAS's own Anthropic path sends sampling parameters that Claude Sonnet 5 rejects. The two tied candidates for the served pipeline (retrieved context, Claude Sonnet 5, zero-shot and citation-required) were generated and judged twice more, so their numbers carry a measured spread. Everything else is a single run and is labelled as one where it is reported.

## Results

### Retrieval

{{table:retrieval_methods}}

Dense search wins in every chunking and embedding combination but one, where it ties hybrid. **BM25 finds a chunk from the right filing in its top five for only {{ret.bm25.dochit_pct}} of questions**, against {{ret.best.dochit_pct}} for the best dense cell: FinanceBench questions share templates ("positive working capital based on FY2022 data"), so lexical overlap points at other companies' identical wording, and fusing BM25 in makes dense retrieval worse. Re-ranking recovers part of that loss but does not earn its latency here. The winner (`fixed_size` + `bge-small-en-v1.5` + dense) is not statistically separable from the next three cells: its lead over the runner-up is {{ret.runner_up.diff}} [{{ret.runner_up.lo}}, {{ret.runner_up.hi}}] in recall@5. Absolute retrieval is low; the best recall@20 in the grid is {{ret.best_recall20}}.

{{table:retrieval_factors}}

**The retrieval method dominates everything else**, at {{ret.share.method}} of the between-cell variance in recall@5. The expected result was that chunking would matter more than the embedding model; that holds only because both are dwarfed by method, and holding the method at dense reverses it, with the embedding model explaining {{ret.dense.embedding}} and chunking {{ret.dense.chunking}}. Retrieval is deterministic, so these numbers carry no run-to-run variance; the interval above resamples questions.

### Correctness and groundedness

{{table:main_eval}}

With the evidence guaranteed, Claude Sonnet 5 answers {{ev.or_sonnet.acc_answered}} of its answered questions correctly and cites excerpts that support its claims with precision {{ev.or_sonnet.cp}}; Qwen2.5 3B manages {{ev.or_qwen.acc_answered}}, and its low groundedness is mostly wrong answers rather than unsupported right ones. Under realistic retrieval **Claude Sonnet 5 declines {{ev.ret_sonnet.declined}} of all answers**, the behaviour a recall@5 of {{ret.best.recall5}} calls for, and on truly unanswerable questions it declines {{ev.ret_sonnet.decl_unanswerable}}. Its answers under retrieval are almost as grounded as under oracle context; what retrieval takes away is the chance to answer.

{{table:quadrant}}

The right-answer-wrong-reasons share is the fraction of correct answers that are not fully grounded. It is high in every cell, but the next section shows how much of it is the judge.

![Retrieval quality against groundedness](results/plots/retrieval_vs_groundedness.png)
*Retrieving the evidence helps, but the two clouds overlap heavily: Claude Sonnet 5 writes fully grounded answers to many questions whose gold evidence was never retrieved, supported by the excerpts it was shown instead, and the retriever's own score says almost nothing about the 3B model's groundedness.*

### Validating the judge

{{table:judge_agreement}}

**The two LLM judges agree with each other ({{ja.second.claim_k}}) and with themselves ({{ja.retest.claim_k}}) far more than either agrees with the author.** Claim-level agreement with the author is barely above chance and its interval includes zero; answer-level agreement is moderate. RAGAS agrees with the author at {{rg.human_ragas.k}}, statistically indistinguishable from the project judge ({{rg.diff}} {{rg.diff_ci}} in κ), and correlates with the author's per-answer scores at ρ = {{rg.human_ragas.rho}}, against {{rg.human_judge.rho}} for the project judge. The comparison favours the project judge by construction, because the author labelled that judge's own claims, and RAGAS still matches it. RAGAS split each answer into {{rg.mean_statements}} statements on average and scored all {{ja.n_answers}}.

![Agreement between raters](results/plots/rater_agreement.png)
*Every model-to-model estimate sits to the right of every model-to-author estimate, although at the answer level the intervals overlap. Consistency between LLM raters is high and says little about whether any of them agrees with a careful reader.*

### The selected pipeline over three runs

{{table:replicates}}

The served pipeline was selected by a rule fixed before any arm was compared: the retrieved-condition arm with the highest share of all its questions answered correctly and fully grounded. In the original run the zero-shot and citation-required prompts tied at {{sel.v1.n_cg}} of {{sel.v1.n}} and the tie went to zero-shot on accuracy. **Over three runs the two prompts remain indistinguishable**: averaged per question across runs, zero-shot minus citation-required is {{rep.pooled.diff_pp}} {{rep.pooled.diff_ci}}, and zero-shot wins the rule in {{rep.v1.wins}} of {{rep.n_runs}} runs. The original run was the best of the three for both prompts: zero-shot answered {{rep.v1.cg_run0}} of questions correctly and fully grounded in it, against {{rep.v1.cg_pm}} over all three, so a single run of this pipeline is one draw and a flattering one here. Run to run, {{rep.v1.cg_changes}} of the {{rep.v1.n_q}} questions change between correct-and-grounded and not under zero-shot, and {{rep.v1.decl_changes}} change between declined and answered: a served answer is one sample, and so is its score.

### Prompt variants

![Prompt variants against the zero-shot prompt](results/plots/prompt_effect.png)
*Only chain-of-thought moves anything, and only for the 3B model, where it raises groundedness and citation precision together. The citation-required prompt (bold, the pre-registered contrast) sits on zero for both generators and both conditions: at most it changes how many excerpts are cited, not whether the cited ones support the answer.*

The pre-registered question was whether citation-required prompting improves groundedness or only its appearance, with citation precision separating the two. It does neither. Under oracle context Claude Sonnet 5's groundedness moves by {{q5.or_sonnet.v2.g}} {{q5.or_sonnet.v2.g_ci}}; the one thing it changes is the number of excerpts cited, which falls. The abstention prompt lowers Claude Sonnet 5's answer rate under retrieval from {{q5.ret_sonnet.v4.ans_v1}} to {{q5.ret_sonnet.v4.ans_v4}} without changing the groundedness of what it does answer.

### Adversarial questions

{{table:adversarial}}

**Two-filing questions fail at retrieval, not at reasoning**: with both filings' evidence supplied, Claude Sonnet 5 answers every one correctly under all four prompts; under retrieval its accuracy is {{adv.b.ret_sonnet.acc}} and it declines {{adv.b.ret_sonnet.declined}} of the time, because only {{adv.b.n_retrieved_any}} of the ten questions retrieves any evidence at all. On unanswerable questions it declines {{adv.c.ret_sonnet.declined}}. The false-premise results use the author's blind labels (a premise judge agrees with them at κ = {{q4.premise_k}}): Claude Sonnet 5 rejects the premise in {{q4.prem.sonnet}} {{q4.prem.sonnet_ci}} of answers, Qwen2.5 3B in {{q4.prem.qwen}}, and the 3B model accepts the invented figure and answers with it in {{q4.prem.qwen_accepts}} of its {{q4.prem.qwen_n}} answers.

![Adversarial categories](results/plots/adversarial_breakdown.png)
*Each panel shows the behaviour its category rewards. Guaranteeing the evidence turns the two-filing category from Claude Sonnet 5's worst into a perfect score, which locates the failure in retrieval; the false-premise panel is the one where neither model does well.*

### Do independent reliability signals agree?

![Agreement between reliability signals](results/plots/signal_agreement.png)
*For Claude Sonnet 5 under oracle context, the model's stated confidence tracks the judge's groundedness better than a judge-free check does: the share of an answer's figures found in its cited excerpts is uncorrelated with groundedness. Grey bands are the random baseline from permuting one signal against the other.*

Groundedness (G), citation precision (CP, from the same judge), a judge-free check of whether an answer's figures appear in its cited excerpts (NS), and stated confidence (C) were compared pairwise against a permutation baseline. The pre-registered expectation was that the judge-free check would agree with groundedness and confidence would not. The opposite happened for Claude Sonnet 5 under oracle context: G-C correlates at ρ = {{q3.or_sonnet.gc}} {{q3.or_sonnet.gc_ci}}, G-NS at {{q3.or_sonnet.gns}} {{q3.or_sonnet.gns_ci}}. Why the judge-free check fails here is open. The obvious explanation, that it cannot see derived figures (a ratio or a sum never appears verbatim in an excerpt), predicts a weaker correlation on the computational questions, and the post hoc split shows the opposite: ρ = {{q3.or_sonnet.gns_metrics}} on metrics-generated questions and {{q3.or_sonnet.gns_other}} on the rest. For the 3B model, whose stated confidence is near the maximum on almost everything, the confidence flag has no variance at all.

## What the Judge Gets Wrong

The claim-level κ of {{ja.human_judge.claim_k}} hides two opposite systematic disagreements rather than noise. On entity and period matching the author was more lenient than the judge: a list of AMD products supported only by the FY2015 10-K and claimed "as of FY22", or background facts absent from the excerpts, were labelled supported by the author and not by the judge. On figures derived by arithmetic or read across excerpts the author was stricter: Adobe's free cash flow, computed from rows of one table whose year header sat in a different excerpt, and Microsoft's total debt as the sum of two lines. Two answers account for most of the disagreement: a 26-claim AMD product list and the Adobe answer. A labelling rule adopted part-way through (a figure whose year column is cut off is unsupported unless another excerpt settles it) is stricter than the judge's prompt, and likely contributed to the second direction. Per generator, κ is unstable for a plainer reason: on Claude Sonnet 5's claims both raters call most claims supported, so raw agreement is high and κ collapses toward zero, the familiar prevalence paradox.

The author later adjudicated four answers where the blind labels had departed from the written rules, changing {{ja.adjudicated_n}} claims to the judge's verdict. Adjudicated agreement rises to κ = {{ja.adjudicated.claim_k}} per claim, but it is reported only beside the blind figure, never in place of it, because the changed claims match the judge by construction.

The review of the correct-but-ungrounded quadrant sharpens this. Of {{rev.wrong}} verdicts the author judged wrong, {{rev.broke_rules}} were the judge breaking its own rules: decomposing statements about the excerpts as facts about the company, copying a word from the question into every claim, returning a verdict that contradicts its own reason. The other {{rev.rule_disputed}} were the judge applying a rule correctly where the author thinks the rule is wrong for the case: an evaluative conclusion such as "not a high-growth company" counts as unsupported unless an excerpt says so. The first kind is a judge error; the second is a definitional choice, and it is the one no better judge would remove.

Three decisions were reversed along the way, each with the evidence on both sides. The first local model was to be a 7B or 8B model; even quantised it would not fit a 4 GB card and would have run partly on the CPU, so a 3B model was chosen that runs entirely on the GPU, at a known cost in quality. The oracle condition did not exist in the original design; it was added after the first full run showed recall@5 at {{ret.best.recall5}}, low enough that realistic retrieval alone could not separate a model that reasons badly from one that was never shown the evidence. And the rule that a question whose filing is missing from the corpus is unanswerable turned out wrong for {{ooc.answerable_elsewhere}} of {{ooc.n}} such questions, whose answer another filing in the corpus holds; an audit reclassified them before the final numbers, raising Claude Sonnet 5's measured decline rate on truly unanswerable questions to {{ev.ret_sonnet.decl_unanswerable}}.

## Deployment

The selected pipeline is served by a FastAPI service whose `POST /query` returns the answer, its citations, a latency breakdown and **the groundedness score of that answer**, computed by the same judge, prompts and model as the evaluation. Every score travels with the judge's measured agreement with the author (κ = {{ja.human_judge.answer_k}} per answer), and the service refuses to start if a judge prompt's hash differs from the validated one, since the κ would no longer describe it. Loaded from its serving bundle alone, the service reproduces all {{serve.n}} evaluated answers of the selected arm and their groundedness to the claim verdict. `GET /health`, `GET /ready` and Prometheus `GET /metrics` are exposed, every request is logged as JSON lines, and spend is capped per deployment. The architecture is drawn in [docs/architecture.md](docs/architecture.md).

![Demo](docs/demo.gif)
*A grounded answer, a correct answer the judge scores zero because the five excerpts do not support it, and the κ that travels with every score. Answers are replayed from the cache, so the latency shown is the service's own.*

{{table:load_levels}}

With every model response replayed from the cache, throughput saturates at about {{load.max_rps}} requests per second from {{load.max_rps_users}} users, and beyond that latency grows with the queue. The bottleneck is embedding the query on the CPU; the vector search takes about {{load.search_ms}} ms at the median. Live, on {{load.live_n}} questions, an answer takes {{load.live_p50_s}} at the median and {{load.live_p95_s}} at the 95th percentile, and costs {{load.live_cost_q}} on average. **The groundedness judge is {{load.judge_share_cost}} of that cost and {{load.judge_share_time}} of the time**: measuring whether an answer is grounded costs {{load.judge_ratio}} times as much as producing it. A fresh answer is a different sample, too: re-asking the {{load.live_n}} questions gave the identical text in {{load.same_text}} cases and the same groundedness in {{load.same_ground}}, most of those because neither answer had a score.

Kubernetes manifests (deployment with probes, service, configmap and a CPU autoscaler) were verified on a local `kind` cluster: the rollout took {{k8s.rollout_s}}, and under eight concurrent users the autoscaler added replicas, the first ready after {{k8s.first_replica_s}}. What a real cluster would need differently is written up in [k8s/README.md](k8s/README.md): shared rather than per-pod state for the spend cap and cache, a scaling signal based on requests in flight rather than CPU once requests wait on the model API, and the index as a service rather than baked into the image. From a fresh clone, `docker compose up` with only an API key added returned the first answer after {{deploy.clone_to_answer_s}} against a five-minute target; the {{deploy.download_gb}} GB image download dominates, so a slower connection would miss it.

## Reproducing the Results

Dependencies are managed with `uv`, and `uv sync` installs the pipeline. Development and every run used an RTX 3050 4GB; embedding also runs on CPU, more slowly. Each script below is a DVC stage, run in this order; the first four build the corpus, the embeddings and the indices from nothing, and the rest replay every model call from the response caches. The pilot runs, the adversarial set's validation and context build, and the premise-judge pilot are further stages, not timed here.

{{table:runtimes}}

The raw filings and every cached model response are published as two archives on the `data-v1` release ({{release.raw_mb}} MB and {{release.caches_mb}} MB). `scripts/19_data_release.py restore` downloads them and checks their SHA-256, fetches FinanceBench from Hugging Face at the pinned revision rather than redistributing it, and confirms that every DVC root matches its recorded hash. From there `dvc repro --force` rebuilds every stage without calling a model API (the network is needed only for the embedding models and the RAGAS environment's packages), and `scripts/11_verify_reproduction.py` compares the rebuilt results with a copy taken beforehand, field by field; the only differences it allows are listed in the script with their reasons, such as timings and the last digits of GPU embedding arithmetic. Qdrant must be running for the index stage.

```bash
uv sync
uv run python scripts/19_data_release.py restore        # raw filings and every model response
docker compose up -d qdrant                             # the second vector backend
cp -r results ../results_committed                      # the reported numbers, for comparison
dvc repro --force                                       # every stage, no model API calls
uv run python scripts/11_verify_reproduction.py ../results_committed
uv run python scripts/17_readme.py --check              # this README against the results
```

Regenerating answers or judge verdicts rather than replaying them calls the Claude API and draws new samples, since Claude Sonnet 5 accepts no seed: the whole project spent {{spend.total_usd}} across {{spend.n_calls}} calls, recorded per phase in `results/metrics/api_spend.json`. Every response is cached on disk by a hash of its request, so re-running a finished stage is free. The RAGAS comparison runs in its own environment, which its script creates on first use. To serve the pipeline, add an API key to `.env` and run `docker compose up`.

## Limitations

Every groundedness number here is a model's judgement checked by one person on {{ja.n_answers}} answers, and that check is the weakest link: claim-level agreement with the author is barely above chance, the author's own labels shifted on a second reading of four answers against the rules, and the judge and RAGAS share a model family with one of the two generators they grade. A second rater of the same family agreeing with the first says little. Only the two tied candidates for the served pipeline were run three times; every other comparison, including the whole prompt ablation, the oracle condition and the adversarial set, is a single run, and the replicates show that a single run of this pipeline moves by several questions in each direction. The benchmark is small ({{fb.n_questions}} questions, {{ret.n_scored}} with aligned evidence, {{adv.n_questions}} adversarial questions at ten per category) and covers one domain and one document type, so the intervals are wide and the findings may not carry to contracts, medical records or anything outside US public-company filings.

The metrics have blind spots of their own. Citation precision can only credit an excerpt the judge finds supporting, so it inherits every judge error; the judge-free figure check cannot see derived numbers; and a fully grounded answer can still be wrong when the excerpt it faithfully repeats is the wrong year or the wrong company, which is exactly the entity confusion that whole-corpus retrieval produces. The service is a demonstration of the measurement, not a production system: it has no authentication, no rate limiting beyond a spend cap, no monitoring beyond Prometheus counters, no process for new filings as they are published, and its judge adds {{load.judge_ratio}} times the cost of the answer it scores.

## Dependencies

| Package | Purpose |
|---|---|
| anthropic | Claude Sonnet 5 generation and the groundedness judge, through the Messages and Message Batches APIs |
| ollama | Local Qwen2.5 3B generation at a fixed seed |
| sentence-transformers | The three embedding models and the `bge-reranker-base` cross-encoder |
| faiss-cpu | The primary vector index, behind the project's store adapter |
| qdrant-client | The second vector backend, run in Docker |
| rank-bm25 | BM25 scoring for the lexical and hybrid arms |
| lxml | The HTML filing parser that separates layout tables from data tables |
| pymupdf | The PDF parsing path (unexercised: every filing in the corpus is HTML) |
| ragas | The external faithfulness metric compared with the project's judge |
| mlflow | Experiment tracking of every evaluated arm and the registered pipeline |
| dvc | The versioned corpus, caches and the pipeline that rebuilds every result |
| fastapi + uvicorn | The served API |
| prometheus-client | The service's request, latency and spend metrics |
| locust | The replay load test |
| matplotlib | Every plot |

## References

Cohen, J. (1960). *A Coefficient of Agreement for Nominal Scales*. Educational and Psychological Measurement.

Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation*. arXiv:2309.15217.

Islam, P., Kannappan, A., Kiela, D., Qian, R., Scherrer, N., & Vidgen, B. (2023). *FinanceBench: A New Benchmark for Financial Question Answering*. arXiv:2311.11944.

Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., & Yih, W. (2020). *Dense Passage Retrieval for Open-Domain Question Answering*. EMNLP.

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS.

Robertson, S., & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*. Foundations and Trends in Information Retrieval.

Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., & Stoica, I. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. NeurIPS Datasets and Benchmarks.
