# Real closed-loop / binding-error incident search

Date: 2026-05-31

Purpose: identify published real-world incidents close to the manuscript's failure mode: a binding, sample identity, metadata, plate-map, data-join, or automated interpretation error causing a scientific false conclusion, preferably in an autonomous or closed-loop discovery setting.

## Bottom line

I did not find a clean public case of a deployed self-driving laboratory where a documented sample-label/provenance binding error caused the active-learning controller to spend budget on a false scientific axis.

However, several documented incidents are close enough to support the motivation and realism of the mechanism:

1. A real autonomous materials lab paper required post-publication correction after concerns about unambiguous phase identification, novelty claims, and a target mistakenly included in training data.
2. Duke genomic chemosensitivity predictors used high-throughput data-processing pipelines with simple errors such as row/column offsets and sensitive/resistant label reversals; the results were used to allocate patients to clinical trials before three trials were suspended.
3. A rice immunity discovery was retracted after a mislabeled bacterial strain helped misidentify the protein responsible for XA21-mediated immunity.
4. A materials-physics superconductivity claim caused two Physical Review B retractions after a sample was mislabeled.
5. Public omics studies contain confirmed sample mix-ups, including row/column/plate-layout errors, with measurable consequences for recovered genetic associations.

The honest paper-level use is not "real closed-loop false-science induction has occurred." The stronger and defensible use is:

> Real scientific workflows have repeatedly produced false mechanistic claims, retractions, and clinical or follow-up decisions from sample identity, row/column offset, plate-layout, and data-processing errors. Closed-loop AI discovery adds a new amplification channel: the learned false conditional relation is converted into experimental budget.

## Candidate incidents and relevance

### 1. A-Lab autonomous materials synthesis correction

Source:
- Szymanski et al., "An autonomous laboratory for the accelerated synthesis of inorganic materials", Nature 2023.
- Author Correction, Nature 2026, DOI 10.1038/s41586-025-09992-y.
- Summary page: https://experts.umn.edu/en/publications/author-correction-an-autonomous-laboratory-for-the-accelerated-sy/
- Original article: https://www.nature.com/articles/s41586-023-06734-w

Key documented points:
- The original system was an autonomous laboratory integrating robotics, ab initio databases, ML-driven data interpretation, text-mined synthesis knowledge and active learning.
- The correction says concerns were raised about unambiguous structure identification from diffraction and original material-novelty claims.
- The correction states the platform's successes were revised: 36 of 40 reported successes remained confirmed, four were inconclusive from XRD alone.
- One compound was removed from the discussion because it was mistakenly included in the training data.

Fit to our story:
- Strongest public autonomous-discovery relevance.
- Not a binding-error false-axis incident.
- Supports the editorial claim that autonomous discovery claims can depend on data interpretation, provenance, and training-data boundaries, and can require post-publication reanalysis.

Usable wording:
- "Recent autonomous materials-discovery work has already required post-publication correction when automated phase identification and novelty/provenance claims were challenged."

Do not say:
- Do not claim A-Lab had sample misbinding.
- Do not claim A-Lab's active-learning controller was misdirected by a binding error.

### 2. Duke genomic chemosensitivity predictors and suspended clinical trials

Sources:
- Baggerly and Coombes, "Deriving chemosensitivity from cell lines: Forensic bioinformatics and reproducible research in high-throughput biology", Annals of Applied Statistics 2009, DOI 10.1214/09-AOAS291.
- Nature Correspondence by Baggerly, "Disclose all data in publications", Nature 2010: https://www.nature.com/articles/467401b
- NCBI Bookshelf / IOM case summary: https://www.ncbi.nlm.nih.gov/books/NBK202159/

Key documented points:
- High-throughput microarray data were used to derive drug sensitivity signatures intended to predict patient response.
- Baggerly and Coombes reported simple but consequential errors, including row/column offsets.
- The IOM case summary notes sensitive/resistant label reversals among discrepancies and potential incorrect direction of therapy.
- Nature notes that three Duke clinical trials were suspended in late 2009 because of irreproducibility of genomic signatures used to select cancer therapies.

Fit to our story:
- Not autonomous lab closed-loop.
- Very strong "binding/data-processing errors can direct real scientific/clinical decisions" precedent.
- Directly resonates with our row/column-offset / binding-to-action mechanism.

Usable wording:
- "The translational omics literature already contains cases where simple row/column offsets and label reversals in high-throughput data pipelines propagated into clinical decision protocols."

Do not say:
- Do not call this an AI closed-loop discovery incident.
- Do not claim it proves our mechanism occurred naturally.

### 3. Ronald / XA21 / Ax21 rice immunity retraction

Sources:
- UC report: https://www.universityofcalifornia.edu/news/rice-disease-resistance-discovery-closes-loop-scientific-integrity
- Retraction Watch summary: https://retractionwatch.com/2013/10/10/ronald-science/
- PubMed retracted Science paper entry: https://pubmed.ncbi.nlm.nih.gov/19892983/

Key documented points:
- The lab announced Ax21 as the protein triggering XA21-mediated immunity in rice.
- In 2013, repeat work found that a bacterial strain was mislabeled and a test was variable.
- UC says these errors led to misidentification of Ax21 as the relevant immune-trigger protein.
- The group retracted two papers, including a Science paper.

Fit to our story:
- Strong direct example of mislabeled biological material inducing a false mechanistic scientific claim.
- Not closed-loop or AI-driven.

Usable wording:
- "In rice immunity, a mislabeled bacterial strain helped produce a false mechanistic assignment and subsequent retractions."

### 4. Mislabeled superconductivity sample causing chained retractions

Source:
- Retraction Watch: https://retractionwatch.com/2013/05/14/mislabeled-sample-leads-to-a-chain-reaction-of-physics-retractions/

Key documented points:
- Two Physical Review B papers were retracted after a sample used in the first paper and relied on in the second was found to be mislabeled.
- The reported Ba-doped phenanthrene sample was later identified as La-doped phenanthrene.

Fit to our story:
- Strong materials/physics sample-identity precedent.
- Shows that sample identity errors can propagate through follow-up science.
- Not closed-loop or AI-driven.

Usable wording:
- "In materials physics, a mislabeled sample propagated into follow-up superconductivity claims and chained retractions."

### 5. MixupMapper / public eQTL sample mix-ups

Source:
- Westra et al., "MixupMapper: correcting sample mix-ups in genome-wide datasets increases power to detect small genetic effects", Bioinformatics 2011: https://academic.oup.com/bioinformatics/article/27/15/2104/400933

Key documented points:
- Sample mix-ups were identified in four of five public human genetical-genomics datasets.
- In one in-house dataset, 28 mix-ups were confirmed by comparing against the RNA sample plate layout; columns had been swapped and rows inverted after hybridization.
- On average 3% of samples had wrong expression phenotypes; correction yielded about 15% more cis-eQTLs; one 23% error dataset yielded three times as many significant cis-eQTLs after correction.

Fit to our story:
- Very strong for realistic plate-layout / row-column binding errors.
- Demonstrates that small-to-moderate sample mix-up rates alter scientific association discovery.
- Not closed-loop budget allocation.

Usable wording:
- "Public high-throughput omics datasets have documented row/column sample-layout errors whose correction substantially changes discovered associations."

### 6. Transcriptomics misannotation prevalence

Source:
- Toker, Feng and Pavlidis, "Whose sample is it anyway? Widespread misannotation of samples in transcriptomics studies", F1000Research 2016: https://pmc.ncbi.nlm.nih.gov/articles/PMC5034794/

Key documented points:
- Apparent mislabeled samples were found in 46% of the datasets studied.
- A lower-bound estimate for all studies was 33%.
- In a single-cohort analysis, 2 of 4 datasets had mislabeled samples, indicating laboratory mix-ups rather than only metadata recording errors.

Fit to our story:
- Already used in B82 as a documented real metadata-error audit.
- Good prevalence anchor.
- Not closed-loop false-science event.

## How this changes the paper

Recommended addition:
- Add a short "Real incidents motivate the closed-loop question" paragraph in the Introduction or Discussion.
- Use A-Lab as the autonomous-discovery anchor.
- Use Duke + Ronald + MixupMapper as the concrete binding/sample-layout false-science anchors.
- State explicitly that these incidents motivate, but do not themselves prove, closed-loop binding-to-budget transduction.

Potential paragraph:

> This concern is not hypothetical in its components. High-throughput science has repeatedly produced consequential errors from sample identity and data-processing mismatches: row and column offsets and label reversals in translational omics contributed to suspended genomic-predictor trials; a mislabeled bacterial strain led to a retracted rice-immunity mechanism; and public eQTL datasets contain confirmed plate-layout swaps whose correction changes discovered associations. More recently, an autonomous materials-synthesis study required post-publication correction after challenges to automated phase identification, novelty and training-data boundaries. These cases do not show that false-science induction has already occurred in deployed closed-loop laboratories. They show that the required ingredients--binding errors, automated interpretation and scientific action--already coexist in modern discovery infrastructure.

## Next search directions

1. Search closed-loop / self-driving lab correction notices directly through Nature, Science, ACS, RSC and Retraction Watch.
2. Search supplementary data and issue trackers for public SDL repositories for "plate", "swap", "mislabeled", "metadata", "correction", "erratum".
3. Search high-throughput active-learning drug-screen papers for errata/corrections involving plate maps, barcode assignments or batch metadata.
4. Search materials autonomous platforms using XRD/Rietveld automation for post-publication corrections or disputed phase assignments.

