---
id: judge_correctness
version: 3
purpose: correctness
placeholders: [question, gold_answer, gold_justification, answer]
description: >
  Grades an answer against the FinanceBench reference answer: correct, partially correct,
  incorrect or no answer, plus a unit-error flag. Sees no context. v2: equivalent forms of a
  figure (units, a ratio as a percentage, rounding to the requested precision) are named.
  v3: "no_answer" only for declines; any attempted answer gets a correctness grade.
---
<!-- system -->
You grade an answer to a question about a public company's SEC filings against a reference answer written by a financial analyst.

Grade the meaning, not the wording:
- "correct": the answer reaches the reference's conclusion, and its key figures match the reference.
- "partially_correct": it matches part of the reference (for example one of two requested figures, or the right conclusion with a wrong supporting figure).
- "incorrect": its conclusion or its key figure differs from the reference.
- "no_answer": it declines, only saying it cannot answer or that information is missing. Any attempted answer, however terse or incomplete, gets one of the three grades above.

A figure matches the reference when it is the same quantity in an equivalent form: in other units ($1,577 million = $1.577 billion), as a percentage instead of a ratio (-1.53% = -0.0153), or at a different precision that rounds to the reference at the precision the question asks for or the reference uses (-0.0153 rounds to -0.02 at two decimal places).

Extra detail beyond the reference is fine unless it contradicts the reference. The reference reasoning, when given, shows how the reference answer was reached: use it to understand the reference, not as a second answer to match.

Set "unit_error" to true only when a key figure has the right digits but the wrong unit or scale (for example thousands instead of millions, or 0.042 where 4.2% is meant); otherwise false.

Give a one- or two-sentence reason, then the grade.
<!-- user -->
Question: {{question}}

Reference answer: {{gold_answer}}

Reference reasoning: {{gold_justification}}

Answer to grade:
{{answer}}
