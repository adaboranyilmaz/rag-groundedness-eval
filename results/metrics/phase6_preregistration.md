# Phase 6 pre-registration: analysis plan and predictions

Written on 2026-09-25, before any Phase 6 analysis was run. Part A fixes how each question
is measured; Part B records the author's predictions. Both are committed before
`scripts/07_reliability_analysis.py` is first run on the real traces, and the script refuses
to run while any prediction in Part B is unanswered. This file is not edited after that
commit; results are compared against it in `results/metrics/reliability_analysis.md`.

Inputs are the Phase 5 per-trace evaluations (`results/metrics/eval_per_trace.jsonl`, SHA-256
`0af8c897c12ec56a4ca566f2b48263222165b59a8ef8c4766cf09278a27dd30c`, single run) joined to the
Phase 4 traces (`results/traces/`) for retrieval metrics and scores.

---

## Part A: analysis plan

### Common definitions

- **Answered**: abstention status `answered` or `partial`. Declines are never scored for
  groundedness or correctness here (Phase 4 rule: confidence is analysed on answered
  questions only).
- **Scored**: answered, and the groundedness score is defined (at least one document claim).
- **Correct**: correctness label `correct`. Sensitivity: `correct` or `partially_correct`.
  `unit_error` counts as incorrect and is reported separately.
- **Grounded**: `fully_grounded` (every document claim supported by the context). This is the
  answer-level measure, the better-validated level of the judge (blind human vs judge kappa
  0.52 per answer, 0.23 per claim; `results/metrics/judge_agreement.json`). Sensitivity:
  groundedness score >= 0.5.
- **Cells**: condition (`retrieved`, `oracle`) x generator (`claude-sonnet-5`,
  `qwen2.5-3b`). The four prompts are pooled unless a question compares them.
- **Intervals**: 95% percentile bootstrap, 10,000 resamples, seed 0, resampling questions
  (all of a question's traces together), since each question appears in up to eight traces.
- **Human-label check**: the primary statistics of Q1-Q3 that use groundedness are
  recomputed on the 50 validation answers with the author's blind labels in place of the
  judge's (`results/labels/groundedness_human.json`), beside the judge's value on the same
  answers. With n = 50 this is a check of direction, not an estimate. Q5 cannot be checked
  this way: the sample holds one trace per question, so it has no v1/v2 pairs.
- **Primary vs exploratory**: the statistics marked *primary* below are the ones the phase
  is judged on. Everything else is reported and labelled exploratory.
- All results are single-run.

### Q1. Does retrieval quality predict groundedness?

- Population: `retrieved` condition, scored traces, questions with aligned gold evidence
  (answerability `evidence_present` or `evidence_absent`). The `oracle` condition cannot
  enter: its recall@5 is 1.0 by construction.
- Unit: question x generator, with the question's groundedness score averaged over its
  scored prompts.
- *Primary*: Spearman rho between recall@5 and mean groundedness, per generator; and the
  difference in mean groundedness between questions with recall@5 > 0 and recall@5 = 0.
- Exploratory predictors: span recall@5, nDCG@5, MRR, and the top-1 retrieval score. The
  last is the only one available at serving time, when there are no gold labels.
- Answer rate per recall group, per generator, is reported beside the correlation:
  groundedness exists only for answers, and whether a model answers depends on its context.
- Exploratory, paired: for questions with recall@5 = 0 under retrieval, the same question,
  generator and prompt under `oracle` vs `retrieved`, where both are scored: change in
  groundedness and in `fully_grounded`, and change in answer rate over all such pairs.
- Reading note: groundedness is judged against the context the model received, not against
  the gold evidence, so an answer can be fully grounded in chunks that hold no gold evidence.

### Q2. Where do correctness and groundedness diverge?

- Population: scored traces whose correctness label is `correct`, `partially_correct`,
  `incorrect` or `unit_error`.
- Output: quadrant counts per cell and per prompt: correct + grounded, correct + ungrounded,
  incorrect + grounded, incorrect + ungrounded.
- *Primary*: the share of correct answers that are not fully grounded ("right answer, wrong
  reasons"), per cell, with its interval.
- Examples: one per quadrant per cell (16), drawn at random with seed 0 from the quadrant,
  not chosen by hand.
- Exploratory, `retrieved` condition, correct answers only: gold evidence retrieved
  (recall@5 > 0) x fully grounded. A correct answer with no gold evidence retrieved that the
  judge also finds ungrounded is the case where two independent measures both say the answer
  did not come from the context.
- Exploratory, judge-free: among correct + ungrounded answers with checkable figures, the
  share whose figures appear nowhere in the context.
- Human check of the quadrant most exposed to judge error: the author reviews a seeded
  (seed 0) random sample of up to 15 correct + ungrounded answers per condition, generators
  pooled, seeing the question, context, answer and the judge's claim verdicts, and records
  for each: the judge is right (not grounded) / the judge is wrong (grounded) / unclear.
  Reported as the share of the sample the judge has right.

### Q3. Do independent reliability signals agree?

- Population: scored traces, per cell, prompts pooled.
- Signals, each in a continuous form and a "flag" form (flag = this signal says the answer is
  unreliable):

  | Signal | Continuous | Flag | Source |
  |---|---|---|---|
  | G | groundedness score | not fully grounded | groundedness judge |
  | CP | citation precision | < 1 | groundedness judge (shared with G) |
  | NS | share of the answer's checkable figures found in its cited excerpts | < 1 | judge-free |
  | C | stated confidence | < 80 | the generator |

  CP is undefined when nothing is cited; NS when nothing is cited or the answer has no
  checkable figure. Each pair uses the traces where both signals are defined.
- Statistics per pair: Spearman rho (continuous), Jaccard of the flagged sets, Cohen's kappa
  of the flags. Each is reported with its bootstrap interval and a random baseline: the mean
  and 2.5-97.5 percentiles of the statistic over 10,000 permutations of one signal within
  prompt strata (seed 0). A pair **agrees above chance** when the observed interval lies
  entirely above the baseline mean.
- *Primary*: G-NS (two independent instruments) and G-C. G-CP is reported with the caveat
  that both come from the same judge, so their agreement is not independent evidence.
- A signal whose most common value covers >= 90% of a cell's traces is reported as
  "no variance" in that cell, not as an agreement number.
- Sensitivity: confidence flag thresholds 70 and 90.

### Q4. Adversarial questions

- Set: 40 new questions over the existing corpus, 10 per category, each drafted with its
  evidence (or a check of its absence) and approved by the author
  (`data/adversarial/questions.jsonl`):
  - (a) answerable from one filing in the corpus; not a FinanceBench question
  - (b) answerable only by combining two filings in the corpus
  - (c) plausible but unanswerable from the corpus: 5 ask for information the filings do not
    disclose; 5 ask about filings not in the corpus (answerable from general knowledge, not
    from the documents)
  - (d) false premise: 5 whose premise a filing contradicts; 5 about a figure or entity that
    does not exist
- Generation: the Phase 4 pipeline unchanged, both generators, all four prompts; `retrieved`
  (Phase 3 winner, k = 5) for all 40, `oracle` additionally for (a) and (b).
- Outcomes:
  - (a), (b): accuracy (strict), groundedness, decline rate (over-abstention)
  - (c): decline rate (declining is correct); groundedness of the answers given
  - (d): premise handling, one of: rejects the premise / accepts it / declines without
    addressing it. Every (d) response is labelled by the author, blind to generator and
    prompt; the author's labels are the measurement. A judge prompt grades the same responses
    as a second rater, and kappa against the author is reported.
- Reporting: counts per category x generator x prompt. Pooled over prompts, intervals
  resample questions; per prompt (10 independent questions), Clopper-Pearson exact
  intervals. With 10 questions per category a proportion's interval is roughly +-30 pp, so no
  between-category test is run; differences are described only where intervals separate.

### Q5. Does citation-required prompting improve groundedness, or only its appearance?

- *Primary* contrast: `v2_citation_required` vs `v1_zero_shot`, same question, generator and
  condition, both scored. The two prompts differ by one instruction: quote the text used.
- Outcomes, paired: change in groundedness score (*primary*), in `fully_grounded`, in
  citation precision, in NS (figures found in cited excerpts), and in number of excerpts
  cited. Unpaired: answer rate under each prompt over all questions, because a prompt that
  changes which questions get answered changes the paired set.
- Reading fixed in advance, from the intervals of the paired changes:
  - **real improvement**: the change in groundedness lies above 0
  - **worse**: the change in groundedness lies below 0
  - **apparent only**: the change in groundedness includes 0 and the change in citation
    precision lies below 0 (under v2 the model cites excerpts that do not support its
    answer more often, so the added quotes make answers look grounded without making them
    so)
  - **no effect**: otherwise

  The change in citation precision and in NS is reported under every reading. Quote
  fidelity is not a criterion: only v2 quotes, so it has no v1 counterpart.
- Exploratory: `v3_chain_of_thought` and `v4_abstention` vs `v1_zero_shot`.

---

## Part B: predictions

Mark exactly one option per line with `[x]`. Written before any Phase 6 analysis was run.
How a result is judged **unexpected**:
- a range (e.g. "10-25%"): the result's 95% interval lies entirely outside the range;
- a direction (up / no change / down): "up" or "down" is contradicted when the interval
  lies wholly on the other side of zero or includes zero; "no change" when the interval
  excludes zero;
- "above chance" and the Q5 readings: decided by the rules in Part A;
- a choice between named options (largest quadrant, highest pair, yes/no): the observed
  option differs from the prediction. These have no interval and are reported as the weaker
  kind of surprise.

### Q1. Retrieval vs groundedness (retrieved condition)

- P1.1 Claude Sonnet 5, Spearman rho(recall@5, groundedness): [x] >= 0.5 · [ ] 0.2 to 0.5 · [ ] -0.2 to 0.2 · [ ] <= -0.2
- P1.2 Qwen2.5 3B, Spearman rho(recall@5, groundedness): [ ] >= 0.5 · [x] 0.2 to 0.5 · [ ] -0.2 to 0.2 · [ ] <= -0.2
- P1.3 The top-1 retrieval score predicts groundedness at least as well as recall@5: [ ] yes · [x] no
- P1.4 Guaranteeing the evidence (oracle vs retrieved, paired) changes groundedness for Claude Sonnet 5: [x] up · [ ] no change · [ ] down

### Q2. Correct vs grounded

- P2.1 Share of correct answers not fully grounded, oracle, Claude Sonnet 5: [ ] < 10% · [x] 10-25% · [ ] 25-50% · [ ] > 50%
- P2.2 Same, retrieved, Claude Sonnet 5: [ ] < 10% · [ ] 10-25% · [x] 25-50% · [ ] > 50%
- P2.3 Same, oracle, Qwen2.5 3B: [ ] < 10% · [ ] 10-25% · [ ] 25-50% · [x] > 50%
- P2.4 Largest quadrant for Qwen2.5 3B under retrieval: [ ] correct+grounded · [ ] correct+ungrounded · [ ] incorrect+grounded · [x] incorrect+ungrounded
- P2.5 In the human check, the judge is right about correct + ungrounded answers in: [ ] < 50% · [x] 50-80% · [ ] > 80%

### Q3. Signal agreement (Claude Sonnet 5, oracle)

- P3.1 G-C (groundedness vs stated confidence), Spearman: [ ] above chance · [x] not above chance
- P3.2 G-NS (groundedness vs figures in cited excerpts), Spearman: [x] above chance · [ ] not above chance
- P3.3 G-CP (both from the judge), Spearman: [x] above chance · [ ] not above chance
- P3.4 The pair with the highest Jaccard of flags: [ ] G-C · [x] G-NS · [ ] G-CP · [ ] C-NS

### Q4. Adversarial questions (pooled over prompts)

- P4.1 Claude Sonnet 5 rejects false premises (d) in: [] < 25% · [x] 25-50% · [ ] 50-75% · [ ] > 75%
- P4.2 Qwen2.5 3B rejects false premises (d) in: [x] < 25% · [ ] 25-50% · [ ] 50-75% · [ ] > 75%
- P4.3 The abstention prompt (v4) raises false-premise rejection over v1, Claude Sonnet 5: [x] yes · [ ] no
- P4.4 Claude Sonnet 5 answers the out-of-corpus questions in (c) from its own knowledge in: [ ] < 25% · [ ] 25-50% · [x] 50-75% · [ ] > 75%
- P4.5 Claude Sonnet 5 accuracy on two-filing questions (b) vs one-filing (a), retrieved: [ ] (b) clearly lower · [ ] about the same · [x] (b) higher

### Q5. Citation-required prompting (v2 vs v1)

- P5.1 Claude Sonnet 5, oracle, groundedness: [x] real improvement · [ ] apparent only · [ ] no effect · [ ] worse
- P5.2 Qwen2.5 3B, oracle, groundedness: [ ] real improvement · [x] apparent only · [ ] no effect · [ ] worse
- P5.3 Qwen2.5 3B, oracle, citation precision under v2: [x] up · [ ] no change · [ ] down

### Anything else expected (free text, optional)

-
