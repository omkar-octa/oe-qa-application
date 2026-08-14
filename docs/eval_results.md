# Eval Results

Snapshot of `python scripts/run_evals.py --model claude-haiku-4-5 --model claude-sonnet-5`
against the full 80-question bank in `src/tests/fixtures/eval_questions.json`, run
2026-08-14. Grading always used `claude-sonnet-5` as the judge (see
`src/utils/eval_judge.py`), so both rows below were scored by the same grader; only the
answering model differs. Raw per-model files: `src/data/eval_results_claude-haiku-4-5.json`,
`src/data/eval_results.json`. Reproduce the comparison table alone with
`python scripts/run_evals.py --summarize <file> <file>`.

## Summary

| Model | Pass | Avg input tokens | Avg output tokens | Avg seconds |
| --- | --- | --- | --- | --- |
| claude-haiku-4-5 | 45/80 | 36,739 | 596 | 9.5 |
| claude-sonnet-5 | 53/80 | 30,969 | 1,105 | 14.6 |

Sonnet passes 8 more questions than Haiku (53 vs 45 of 80) but takes about 1.5x the time
per question and writes almost twice the output tokens; input tokens are similar between
the two; since retrieval, not the answering model, decides how much gets sent to it.

Patterns that show up in both models' failures, not just one:

- **Cross-doc questions are the weakest category for both** (2/10 pass each), by a wide
  margin, worse than every other tag. This looks like a retrieval-strategy gap rather than
  a reasoning gap: several failures name the wrong paper entirely (X2, X7, X8, X9) or miss
  one of several papers the question spans (X9, X10, R5), which is the kind of miss that
  more capable reasoning alone will not fix if the search step never surfaces the right
  chunk.
- **Wrong-document retrieval recurs on the same questions for both models.** C7 and S1
  both models cite Sharmina where the reference expects Calvillo or Seger respectively,
  suggesting these two questions' search keywords systematically favor the wrong file
  regardless of which model is doing the reasoning.
- **Citation-page mismatches drive a large share of the "partial" verdicts** (B1, B6, E8,
  S3, S5, T8, X6, R4, and more): content is often right but the cited page differs from
  the reference's, in both models.
- **Table questions are weak for both, worst for Haiku** (0/3 pass for Haiku, 1/3 for
  Sonnet), consistent with B4/B5/B6 both misreading which table (Table 3 vs Table 5) or
  which wave the reference expects.
- **The judge itself truncated on a handful of questions** (S6 for Haiku; C4, M2, X10 for
  Sonnet) with `stop_reason == "max_tokens"` before writing a verdict, counted as an
  automatic fail per `EvalGrader`'s guard against a blank reasoning string reading as a
  genuine pass. Worth re-running those specific IDs rather than trusting the fail as
  reasoning-based.
- Sonnet recovers several of Haiku's failures on `multi-hop` (11/12 vs 9/12) and `figure`
  (6/14 vs 4/14) tags, but neither model clears half on `figure` or `cross-doc`.

## Results by tag

| Tag | claude-haiku-4-5 | claude-sonnet-5 |
| --- | --- | --- |
| (untagged) | 1/5 | 2/5 |
| compute | 6/14 | 8/14 |
| cross-doc | 2/10 | 2/10 |
| figure | 4/14 | 6/14 |
| footnote | 2/2 | 1/2 |
| multi-hop | 9/12 | 11/12 |
| single | 27/33 | 29/33 |
| table | 0/3 | 1/3 |
| trap | 5/10 | 6/10 |
| **Overall** | **45/80** (13 partial, 22 fail) | **53/80** (11 partial, 16 fail) |

## Not passing: claude-haiku-4-5

- **[B1]** (single) PARTIAL: The content correctly identifies solar PV, solar water
  heating, and (hybrid/electric) vehicles tracked via UKHLS Wave 4 vs Wave 10, matching
  the reference; however, the citation gives page 4 instead of the reference's pages 1
  and 8, a citation mismatch.
- **[B4]** (table, compute) PARTIAL: The system correctly identified HYBRIDEV as having
  the highest combined parental background contribution (66.40%) with correct citation to
  Table 5, p8, matching the reference. However, for the solar PV comparison it incorrectly
  used birth cohort from Table 3 (full sample, 41.89%) instead of the correct Table 5
  figure (49.24%) as specified in the reference, making the comparison factually wrong for
  the requested sub-sample.
- **[B5]** (table, compute) FAIL: The reference (Table 5, p8) shows SOLARHEAT (0.425) >
  HYBRIDEV (0.408) > SOLARPV (0.317) for dissimilarity, disagreeing with the
  parental-background ranking (HYBRIDEV top), this is the key trap. The system instead
  ranked HYBRIDEV highest for dissimilarity and concluded the rankings "match exactly,"
  directly contradicting the reference's core point and using different (likely wrong
  table/wave) figures.
- **[B6]** (table) PARTIAL: The system correctly identifies ethnicity as contributing
  least to EV adoption inequality and rightly contrasts it with solar water heating, but
  it cites different figures (0.02%/3.83% vs 3.03%/9.75%) from Table 3 (p.6-7) rather than
  the reference's Table 5, p.8 values (0.52% vs 9.04%), so both the specific numbers and
  citation are incorrect.
- **[C3]** (multi-hop) PARTIAL: The system correctly identifies real income gains
  outpacing job creation and cites the right source, but it incorrectly attributes
  construction and manufacturing as drivers of real income gains rather than job creation,
  which is a key attribution error relative to the reference.
- **[C4]** (figure) FAIL: The reference shows that changing the assumed HP efficiency
  ratio (1:1 vs 4.8:1) flips the "All Other" employment effect from positive to negative
  (Fig. 5, p9), but the system explicitly denies any sign change and misattributes the
  1:1/4.8:1 ratio to electricity-to-gas prices, citing an unrelated paper (Calvillo) about
  COP sensitivity rather than the correct source's Figure 5 data.
- **[C5]** (figure) FAIL: The reference identifies the South East as worst affected at
  -5,809 FTE (with London second at -3,748) from Fig. 5 panel (f) on p.9, but the system
  instead claims London is worst affected using percentage figures from a different table
  on p.11, misidentifying both the region and the metric/citation.
- **[C6]** (figure, compute) FAIL: The system correctly identifies South East as the
  largest construction gain (2,529, correctly cited), but incorrectly swaps the "All
  Other" loss figures/regions and concludes "No, the same region does not show both,"
  directly contradicting the reference's key point that South East also shows the largest
  "All Other" loss (-5,954), falling into the trap the reference warns against.
- **[C7]** (single) FAIL: The system answered based on the wrong source (Sharmina)
  instead of Calvillo, describing an entirely different modelling approach (LED-F
  framework) rather than the dynamic economy-wide CGE model combined with regional
  economic/workforce data as specified in the reference; both content and citation are
  incorrect.
- **[E4]** (figure, compute) PARTIAL: The system correctly identifies "Impact on
  recreational fishing" with the 12/3=4.0 ratio and cites the correct file/page, matching
  the reference's core fact, but its interpretation is generic ("salient and deeply
  considered concern") rather than reflecting the paper's specific explanation that the
  sample included participants with work/hobby ties to the channel's fish population,
  missing part of what was asked.
- **[E5]** (figure, trap) FAIL: The system fell into the trap by claiming the primary
  themes are simply "the most-referenced ones," when the reference explicitly notes
  Impact on recreational fishing has more references (12) than Impact on wildlife (11)
  despite far fewer participants, meaning "primary" tracks participant breadth, not raw
  reference count.
- **[E8]** (single) PARTIAL: The system correctly identifies reliability/predictability as
  the attractive factor and environmental impact as a caveat, with correct citation
  (Edwards-Jones p3-4), but it fails to capture the specific nuance that the caveat
  centered on scale (support conditional on minimizing impact, favoring small-scale
  lagoons), which the reference specifically highlights as the persistent caveat.
- **[T4]** (multi-hop, trap) PARTIAL: The system correctly reports that dispersed sites
  "now account for more than half" (Taylor p.3) and cites an "almost half" figure for
  dispersed sites from Poulter p.7, but it conflates these into a static "50-50" split
  rather than capturing the paper's key point that clusters were ~half at the time of the
  2021 strategy and dispersed sites have since overtaken them, missing the temporal shift
  the trap warns about, and citing a different source (Poulter) for the clusters figure
  instead of Taylor p.1.
- **[S1]** (single) FAIL: The system answered based on the wrong source document
  (Sharmina, s41560-025-01898-3.pdf) instead of the correct Seger paper on EV workplace
  charging, resulting in completely unrelated content and citation.
- **[S3]** (single) PARTIAL: The content correctly identifies all three objective
  functions (PM-VF, CCM, CEM) matching the reference, but the citation given (p. 8)
  differs from the reference's cited pages (p4, p5), constituting a citation mismatch.
- **[S4]** (figure, trap) FAIL: The system failed to provide any answer, whereas the
  reference clearly states peak demand rises 0.5% under cost-minimisation (Fig. 3e, p5),
  which the system did not report.
- **[S5]** (figure, compute) PARTIAL: The peak reduction figures (7.4% at 15%, 21.3% at
  50%) match the reference exactly and citations point to the correct paper (Seger), but
  the explanation given (valley-filling saturation forcing profile upward) differs from
  the reference's specific "second main insight" about PM-VF variability decreasing and
  predictability improving with higher EV rates, and the page numbers cited (p.3) don't
  match the reference's p4/p5.
- **[S6]** (figure, compute) FAIL: Judge response was truncated (max_tokens) before a
  verdict could be written; re-run this question.
- **[S7]** (figure, compute, trap) FAIL: The system declined to answer, whereas the
  reference shows the answer (CEM at -17.4% at S1) was determinable from Fig. 2, p4 of
  Seger's paper; this counts as a failure per the figure-question grading rule for
  declined answers.
- **[S9]** (single) FAIL: The reference specifies the baseline is the mean electricity
  consumption of February 2023 (Seger p.4), but the system's answer instead claims the
  baseline is "the mean from UCC (Uncontrolled Charging)," a different and confusing
  normalization claim, and cites p.2 rather than p.4.
- **[M2]** (figure) FAIL: The system correctly identifies the two axis labels and cites
  the right source (s41560-025-01898-3.pdf, p.3), but it swaps the horizontal/vertical
  orientation and, more importantly, misplaces two of the four scenarios (Atomized and
  Slow Lane are given opposite growth/cohesion values compared to the reference), which is
  a substantive content error on the exact quadrant mapping the question asks for.
- **[M5]** (multi-hop) FAIL: Although the system correctly identifies Atomized Society as
  having both the smallest energy reduction (18%) and least land conversion (5 kha/yr), it
  self-contradicts and ultimately concludes "the answer is no, they are not the same
  scenario," which is the opposite of the correct answer (they are the same scenario, per
  Sharmina p4).
- **[M6]** (compute) PARTIAL: The answer correctly cites Sharmina p4 and gives the correct
  8.5x factor (128÷15), matching the reference, but it also includes the invalid 80x
  comparison using the cumulative 1.6 kha energy crop figure as if it were a valid
  annual-rate comparison, exactly the trap the reference warns against.
- **[X1]** (cross-doc) FAIL: The reference identifies a specific cross-citation (Taylor
  citing Calvillo re: worker/skills shortages), but the system failed to provide any
  answer.
- **[X2]** (cross-doc) FAIL: The reference identifies the shared programme as Innovate
  UK's PFER connecting Peacock (Rugeley project) and Poulter (West Midlands governance),
  but the system instead named UKERC and cited Sharmina and Poulter, missing the actual
  programme and Peacock connection entirely.
- **[X3]** (cross-doc) PARTIAL: The system correctly identifies Taylor's gaps (data,
  evidence, governance capacity) and Poulter's governance/institutional deficiencies, but
  it concludes the diagnoses "substantially agree" on governance being the core issue,
  missing the reference's key point that Taylor frames it as an evidence/research gap
  while Poulter frames it as a governance/coordination failure. Additionally, citations
  for Poulter (p.7, p.10) diverge from the reference's cited page (p.1).
- **[X4]** (cross-doc) FAIL: The system correctly analyzed Peacock but paired it with
  Poulter et al. instead of Edwards-Jones, which is the wrong paper (Poulter is about
  cross-sectoral governance, not community responses to energy infrastructure with
  photo-elicitation methods and environmental/coastal trade-offs as the reference
  specifies); this is a fundamental mismatch in both content and citation for half the
  comparison.
- **[X6]** (cross-doc) FAIL: The system declined to answer entirely, whereas the reference
  shows the question was answerable from the sources (Edwards-Jones p1 2019 net-zero legal
  obligation; Taylor p1 2021 industrial decarbonisation strategy; no conflict).
- **[X7]** (cross-doc) FAIL: The Sharmina portion is correctly stated and cited, but the
  Calvillo portion is wrong: the reference specifies labour/skills shortages and
  wage-cost inflation (scaling with deployment pace) as the cost driver, whereas the
  system substitutes an unrelated finding about electricity-to-gas price differentials and
  GDP impacts, missing the key contrast the reference warns about.
- **[X9]** (cross-doc) PARTIAL: The system correctly identified Burlinson's and Seger's
  questions with proper citations, but failed to identify the third paper (Sharmina) and
  its role of embedding EVs in scenario variables, leaving a distinct part of the question
  unanswered.
- **[X10]** (cross-doc, trap) FAIL: The system failed to provide any answer, whereas the
  reference shows the question was answerable from multiple documents with specific facts
  and citations.
- **[R2]** () FAIL: The system failed to provide any answer, whereas the reference
  expected the system to report the absence of substantive content on this topic (with
  appropriate context from the corpus), which counts as declining to answer something that
  was addressable.
- **[R3]** () FAIL: The reference explicitly warns that Peacock does not give a
  participant count and that inventing a total is wrong behavior, but the system
  fabricated 14 participants (10+4) for Peacock and produced a false combined total of 33,
  exactly the trap the reference cautions against.
- **[R4]** () PARTIAL: Content correctly avoids the trap (0.277 main sample vs 0.317
  persistent, matching reference's ~0.28 and 0.317 figures), but the citation page (p.14)
  for the main sample dissimilarity value differs from the reference's cited location
  (Table 2/Fig.1, p.8), constituting a citation mismatch.
- **[R5]** () FAIL: The system only summarized 5 of the 8 corpus files, omitting
  Burlinson, Edwards-Jones, and Seger entirely, precisely the failure mode the reference
  warns about (retrieval not spanning all eight files), so it fails the core test despite
  reasonable synthesis of the papers it did find.

## Not passing: claude-sonnet-5

- **[B5]** (table, compute) FAIL: The reference's overall dissimilarity ranking (Table 5,
  Theta_I) is SOLARHEAT (0.425) > HYBRIDEV (0.408) > SOLARPV (0.317), and it concludes the
  rankings disagree at the top; the system instead reports HYBRIDEV > SOLARHEAT >
  SOLARPV using wave-specific figures and concludes they match (at least in Wave 4),
  directly contradicting the reference's key trap point.
- **[B6]** (table) PARTIAL: The system correctly identifies ethnicity as the least
  contributor to EV inequality and correctly notes its much larger role in solar water
  heating, matching the reference's qualitative conclusion, but the cited numbers
  (0.02%/3.83% vs 3.03%/9.75%) differ from the reference's specific figures (0.52% vs
  9.04%) and the citation is to Table 3, p.6 rather than the correct Table 5, p.8.
- **[C4]** (figure) FAIL: Judge response was truncated (max_tokens) before a verdict
  could be written; re-run this question.
- **[C5]** (figure) FAIL: The system misidentifies London as worst-affected with the
  -5,809 FTE figure, when the reference clearly states the South East holds that position
  (-5,809) with London second (-3,748), exactly the trap the reference warns about;
  citation page (p.10) is also off from the correct p.9.
- **[C6]** (figure, compute) FAIL: The reference identifies the South East (not London)
  as having both the largest construction gain (2,529) and largest "All Other" loss
  (-5,954); the system misattributes these exact figures to London, falling into the trap
  the reference warns about, and also cites the wrong page (p.13 vs p.9) and figure (Fig.
  10 vs Fig. 5).
- **[C7]** (single) FAIL: The reference expects the CGE (Computable General Equilibrium)
  model combined with regional economic/workforce data from Calvillo p1, but the system
  answer discusses an entirely different paper (Sharmina, s41560) about a "story and
  simulation" LED-F approach, which is unrelated content and citation.
- **[P8]** (footnote) PARTIAL: The content correctly captures that Northern Ireland is
  excluded because it has a separate energy market/regulation, matching the reference's
  substance, but the citation is given as p.4 rather than p.1 (footnote 1) where the
  reference places it.
- **[T4]** (multi-hop, trap) PARTIAL: The system correctly cites Taylor p.3 for "dispersed
  sites now account for more than half," but for the clusters ~half figure it cites
  Poulter p.7 instead of Taylor p.1, and it frames the two figures as a simultaneous split
  rather than the temporal shift (from clusters dominating to dispersed sites now
  dominating) that the reference emphasizes as the key point.
- **[T8]** (single) PARTIAL: The system correctly identifies metals and non-metallic
  minerals with proper citation (Taylor, p.3), but omits the petrochemicals exception
  (Geels 2022) mentioned in the reference.
- **[S1]** (single) FAIL: The system entirely missed the Seger paper's EV workplace
  charging results (28% peak load reduction, 9% cost/emissions decrease) and instead
  answered about two unrelated documents, failing to address the actual question and
  citing the wrong sources.
- **[S4]** (figure, trap) PARTIAL: The system correctly identifies the key trap, that the
  28% reduction belongs to the peak-minimising model, not CCM, and correctly states that
  under CCM peak demand rises rather than falls, matching the reference's direction.
  However, it fails to state the specific +0.5% figure, omits the accompanying cost
  (-19.6%) and emissions (-15.6%) figures, incorrectly claims the 50% value isn't stated in
  the text (when the reference says it's a bar label in Fig 3e), and cites p.3 rather than
  the correct Fig. 3e on p.5.
- **[S5]** (figure, compute) FAIL: While the peak-reduction percentages (7.4% at S1, 21.3%
  at S3) match the reference, the system's explanation contradicts the paper's actual
  stated insight, the reference says PM-VF variability decreases and the model becomes
  more predictable/effective as adoption rises, whereas the system claims a
  "capacity/flexibility threshold" causes diminishing marginal effectiveness; the citation
  is also wrong (p.3 for Firm-level paper) versus the correct pages (p4/p5, Fig. 2d/3d and
  the second main insight).
- **[S6]** (figure, compute) PARTIAL: The system correctly concludes "yes" each objective
  wins on its own metric and gives roughly correct magnitudes (PM-VF ~21.3% peak
  reduction, CCM ~19% cost savings, CEmissions ~17-19%), but it does not clearly lay out
  the required side-by-side comparison of all three models' values on each metric (e.g.,
  CEM -4.1%/CCM +0.5% for peak, CEM -15.1%/PM-VF -10.0% for cost, CCM -15.6%/PM-VF -10.7%
  for emissions) as the reference does. It also cites page 3 rather than the reference's
  page 5 in the same source file.
- **[S7]** (figure, compute, trap) PARTIAL: The system correctly identifies CEM as the
  best emissions performer at 15% (-17.4%) and gets PM-VF's -10.5% right, but gives CCM's
  figure imprecisely (~15% vs the reference's -13.5%) and entirely omits the key trap
  insight that CCM's cost saving (-18.5%) exceeds CEM's emissions saving, i.e., it fails
  to flag the cross-metric magnitude inconsistency the reference emphasizes.
- **[M2]** (figure) FAIL: Judge response was truncated (max_tokens) before a verdict
  could be written; re-run this question.
- **[M4]** (single, compute) PARTIAL: Content is correct (18% Atomized Society to 45% Slow
  Lane Society), matching the reference range and scenario endpoints, but the citation
  differs from the reference (p4 and p6 vs. the reference's p1 abstract and p4), so the
  citation only partially aligns.
- **[X1]** (cross-doc) FAIL: The reference identifies Taylor citing Calvillo for the
  labour/skills shortage and wage competition claim, but the system instead claims
  Calvillo cites Poulter for jobs/skills empirical research, which is a different,
  unsupported cross-citation not matching the reference's answer.
- **[X2]** (cross-doc) FAIL: The reference identifies the specific programme "Prospering
  from the Energy Revolution" (PFER) linking the Peacock and Poulter papers, but the
  system's answer instead discusses a different funding source (UKERC/UKRI grant
  EP/S029575/1) across various papers, never mentioning PFER or the Peacock-Poulter
  connection at all.
- **[X3]** (cross-doc) PARTIAL: The system correctly identifies and cites what each paper
  says is missing (Poulter's governance/user-intermediary gap at p.7/p.10, Taylor's
  evidence/research gap at p.3), matching the reference's specifics well. However, its
  final synthesis claims the diagnoses are essentially "consistent" and "reinforce rather
  than conflict," whereas the reference specifically notes they differ in diagnosis
  (evidence/research gap vs. governance/coordination failure) even while agreeing
  dispersed sites are under-served, a nuance the system's conclusion glosses over.
- **[X6]** (cross-doc) PARTIAL: The system correctly identifies Edwards-Jones (2019) and
  Taylor (2021) with correct citations and correctly concludes no conflict, but the third
  paper should be Burlinson (referring to the net-zero commitment without a date), instead
  the system substitutes Poulter with an unrelated 2019 Climate Change Act amendment
  claim, missing the actual third source.
- **[X7]** (cross-doc) FAIL: The reference's key contrast is that Calvillo attributes cost
  pressure to labour/skills shortages and wage-cost inflation (scaling with deployment
  pace), not to technology or price choices, whereas the system's answer claims Calvillo's
  driver is the electricity-to-gas price ratio, entirely omitting the labour-market/
  wage-inflation point and thus missing the specific contrast the reference emphasizes; it
  also cites Calvillo pp. 4/7/16 rather than the referenced p. 1.
- **[X8]** (cross-doc) FAIL: The reference identifies Peter Taylor/Imogen Rattle linking
  the Taylor and Poulter papers on dispersed industrial site decarbonisation, but the
  system instead cites Christian Brand linking Sharmina and Seger papers on energy
  demand/EV charging, which contradicts the reference's answer entirely.
- **[X9]** (cross-doc) FAIL: The system failed to provide any answer, whereas the
  reference clearly identifies three specific papers (Burlinson, Seger, Sharmina) and
  their distinct EV-related questions.
- **[X10]** (cross-doc, trap) FAIL: Judge response was truncated (max_tokens) before a
  verdict could be written; re-run this question.
- **[R3]** () FAIL: The reference expects the second qualitative study to be Peacock
  (gatekeeper interviews/cultural animation workshops), which does not give a participant
  count, but the system instead substituted Poulter's data and fabricated a combined total
  of 110, exactly the invented-total trap the reference warns against.
- **[R4]** () PARTIAL: The system's numeric values (0.277 main-Wave4, 0.203 main-Wave10,
  0.317 persistent sub-sample) match the reference exactly and correctly distinguish main
  vs. sub-sample, avoiding the trap. However, it cites Table A9, p.14, whereas the
  reference attributes this data to Table 5/Table 2, p.8, a significant page/table
  discrepancy that counts as a citation failure.
- **[R5]** () FAIL: The system only summarized six of the eight papers, omitting
  Burlinson (1-s2.0-S0140988325000672-main.pdf) and Taylor (industrial decarbonisation
  research agenda) entirely, which is exactly the failure mode the reference describes as
  the test's purpose (spanning all eight files, not just top-scoring subset).
