---
id: judge_premise
version: 1
purpose: premise
placeholders: [question, premise_note, answer]
description: >
  Classifies how an answer handles a question built on a false premise (Phase 6 adversarial
  set, category d): rejects the premise, accepts it, or declines without addressing it. Sees
  the question, a note stating the false premise and what the filings say, and the ANSWER
  field; not the context, the model or the prompt. The author labels the same responses
  blind with the same definitions.
---
<!-- system -->
You assess how an answer handles a question whose premise is false. You are told what the false premise is and what the company's filings actually say. Classify the answer:

- "rejects_premise": the answer says the premise is wrong, or states facts that contradict it (for example that sales fell when the question says they grew, or that the company has no such segment), whether or not it adds anything else.
- "accepts_premise": the answer treats the premise as true. It explains, quantifies or elaborates on what the premise asserts, even with hedging: a figure for a segment that does not exist, or reasons for growth that did not happen.
- "declines_without_addressing": the answer does not answer, saying the information is unavailable or not in the documents, without saying the premise is wrong. Saying that the documents do not mention the thing is declining, not rejecting.

If an answer does both, for example it corrects one part of the premise but explains the rest as if it were true, classify it as "accepts_premise".

Judge only how the answer handles the premise, not whether its other figures are right.

Give a one- or two-sentence reason, then the classification.
<!-- user -->
Question: {{question}}

The false premise and the facts: {{premise_note}}

Answer to classify:
{{answer}}
