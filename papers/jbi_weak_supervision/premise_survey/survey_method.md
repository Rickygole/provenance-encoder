# Premise survey: how clinical and biomedical weak-supervision papers certify the labelling stage

Method record. Compiled 2026-10-15. All searches run 2026-10-15.

## 1. Question

The paper under preparation asserts, without citation, that in clinical and biomedical NLP a
weak-supervision labelling stage (an LLM or heuristic labeller whose output becomes training
labels, or is promoted into a pipeline) is "commonly" certified as adequate by showing that it
beats chance, rather than by comparison against a non-trivial alternative explanation of its
scores such as a metadata-only baseline. This survey was run to find out whether that assertion
is supportable as written, and if not, what wording the evidence does support.

## 2. Status of this sample

This is a **purposive, reportable sample, not an exhaustive systematic review**, and it was not
pre-registered. It is adequate for a motivating premise with a stated N and a stated sampling
frame. It would not be adequate as a headline empirical contribution. If the claim is promoted
to a contribution, the survey must be redone with a pre-registered protocol, a second coder and
a record-level screening log.

## 3. Databases and exact queries

### Search A, PubMed, programmatic weak supervision on clinical or biomedical text

Endpoint: NCBI E-utilities esearch, db=pubmed.

```
("weak supervision"[tiab] OR "weakly supervised"[tiab] OR "distant supervision"[tiab]
 OR "labeling functions"[tiab] OR "data programming"[tiab] OR "Snorkel"[tiab])
AND ("clinical notes"[tiab] OR "electronic health record"[tiab] OR "electronic health records"[tiab]
 OR "clinical text"[tiab] OR "biomedical text"[tiab] OR "radiology report"[tiab]
 OR "radiology reports"[tiab] OR "pathology report"[tiab] OR "clinical narrative"[tiab])
AND 2019:2026[dp]
```

Records identified: **62**. All 62 PMIDs were retrieved and title-screened via esummary.

### Search B, PubMed, LLM annotation of clinical text (run, then rejected as a frame)

```
("large language model"[tiab] OR "large language models"[tiab] OR GPT-4[tiab] OR ChatGPT[tiab])
AND ("annotation"[tiab] OR "annotations"[tiab] OR "labeling"[tiab] OR "labelling"[tiab] OR "labels"[tiab])
AND ("clinical notes"[tiab] OR "clinical text"[tiab] OR "radiology reports"[tiab]
 OR "pathology reports"[tiab] OR "electronic health records"[tiab])
AND 2022:2026[dp]
```

Records identified: **241**. This search was **not screened**. On inspection of the first
200 returned identifiers it was judged non-specific: the great majority are LLM-in-medicine
evaluation papers with no labelling stage whose output is reused, so title-screening 241 records
would have added cost without adding eligible studies. It was replaced by Search C. This is
recorded here because the search was run, and a reader is entitled to know that a large
unscreened block exists and that the LLM arm of this sample is therefore not exhaustively
sampled.

### Search C, PubMed, LLM-generated labels reused as supervision

```
("large language model"[tiab] OR "large language models"[tiab] OR "GPT-4"[tiab])
AND ("weak supervision"[tiab] OR "weakly supervised"[tiab] OR "silver standard"[tiab]
 OR "silver-standard"[tiab] OR "pseudo-labels"[tiab] OR "pseudo-labeling"[tiab]
 OR "training labels"[tiab] OR "automated annotation"[tiab])
AND (clinical[tiab] OR radiology[tiab] OR pathology[tiab] OR "electronic health record"[tiab]
 OR "electronic health records"[tiab] OR biomedical[tiab])
AND 2023:2026[dp]
```

Records identified: **22**. Nine overlapped Search A. The other 13 were title-screened.

### Search D, arXiv API

```
http://export.arxiv.org/api/query?search_query=all:"weak supervision" AND all:"clinical"
   &start=0&max_results=60&sortBy=relevance
```

The feed reported **67** total results. The response was truncated by the fetch tool at 100,000
characters, so **41** entries were readable and title-screened. The remaining 26 were not
screened. This is a known incompleteness in the NLP-venue arm.

### Search E, site-restricted search of the ACL Anthology

Query: `weak supervision labeling functions clinical notes` restricted to `aclanthology.org`.
Nine links returned, all title-screened. Clinical-notes coverage in the Anthology under these
terms is thin; the search surfaced mostly domain-general weak-supervision methods papers.

### Search F, known-item additions (4)

Four studies were added by known-item search rather than by database query, because they are
the canonical report-labeller and LLM-labeller references in this literature and their omission
would have made the sample unrepresentative in the direction favourable to the claim:

- Irvin et al. 2019 (CheXpert), verified at arXiv 1901.07031.
- Smit et al. 2020 (CheXbert), verified at ACL Anthology 2020.emnlp-main.117.
- Agrawal et al. 2022, verified at ACL Anthology 2022.emnlp-main.130.
- Guevara et al. 2024, found by a targeted PubMed author search (`Guevara M[au] AND social
  determinants of health AND large language models`, 1 record), verified at PMID 38200151.

Adding known items is a deviation from a protocol-driven search and is disclosed as such.
Three of the four (CheXpert, CheXbert, Agrawal) are among the most methodologically careful
records in the sample, so this deviation works against the claim under test, not for it.

## 4. Eligibility criteria

Included if all of:

1. Clinical or biomedical **text** is the input to a labelling stage: hand-written rules,
   dictionaries, ontologies, labelling functions, distant supervision from a knowledge base or
   from structured fields, a silver-label phenotyping algorithm, or an LLM prompt.
2. The labels produced are used as **training labels for a downstream model**, or the labeller's
   output is **promoted into a pipeline** as a phenotype, variable or extracted field.
3. Published 2019 to 2026.
4. The record is verifiable against a primary record (publisher page, PubMed, ACL Anthology, arXiv).
5. Full text was retrievable and machine-readable, so the coding fields could actually be read.

Excluded if any of:

- Weak supervision over images, video or physiological signals with no text labeller
  (this removed the largest block at title screening).
- Review, survey, overview, editorial or methodological commentary.
- No labelling stage: fully manual labels, or fully unsupervised methods with no emitted labels.
- Domain-general weak-supervision methods papers with no clinical or biomedical application.
- Preprint version of a study included at its journal version (journal version retained).

## 5. Flow and counts

Identification counts per search are exact (above). **A record-level de-duplication log was not
maintained at the title-screening stage, so a single "unique records screened" number is not
reported.** Reporting one would require arithmetic over an overlap set that was not logged, and
an invented number is worse than an absent one. The full-text stage is enumerated record by
record below.

- Candidate studies advanced to full text: **45**
- Excluded at full text, full text not retrievable or not machine-readable: **5**
  - Automated labeling of German chest X-ray radiology reports (PMID 38663428, Rofo; arXiv
    2306.05997): no PMC record, and the ar5iv HTML conversion failed with a fatal error.
  - Extraction of distant recurrence sites for breast cancer patients from free-text clinical
    notes using large language models (PMID 41911958, J Biomed Inform 2026, PMC13164447): PMC
    record not live until 2027-03-28.
  - A weakly supervised model for the automated detection of adverse events using clinical notes
    (PMID 34864210, J Biomed Inform 2022): not in PMC, no accessible full text.
  - J Am Med Inform Assoc 2026, doi 10.1093/jamia/ocag014 (PMID 41801982, PMC13127649): Europe
    PMC returned HTTP 500 and the PMC page returned HTTP 403.
  - Cross-task knowledge transfer for extremely weakly supervised text classification
    (aclanthology 2023.findings-acl.328): PDF text layer not extractable by the fetch tool.
- Excluded at full text, scope: **3**
  - Reduction of supervision for biomedical knowledge discovery (PMID 40890616, BMC
    Bioinformatics 2025): fully unsupervised, no labelling stage emitting labels.
  - Optimizing large language models for ontology-based annotation (PMID 41366817, BioData
    Mining 2025): training targets come from the manually curated CRAFT gold corpus, no weak
    labelling stage.
  - LADDER (aclanthology 2025.findings-acl.1177): pseudo-labels are attached to image slices
    from image-derived captions, not to clinical text.
- **Included and coded: 37**

## 6. Verification

Every included record carries, in `survey.json`, at least one primary identifier and the URL at
which it was verified. Bibliographic metadata (full title, author list, journal, year, volume,
pages) was taken from the NCBI PubMed record via esummary for the 33 PubMed-indexed records, and
from the arXiv abstract page or the ACL Anthology landing page for the other four. DOIs and
PMCIDs came from the NCBI PMC ID converter API. No metadata was taken from a secondary source,
and no encyclopaedic source was used.

## 7. Coding procedure and its limitations

Each included paper was coded from its full text (Europe PMC `fullTextXML`, the PMC article page,
ar5iv HTML, or the ACL Anthology landing page) with one fixed seven-item instrument:

1. What is the weak or automatic labelling stage?
2. Is the quality of that labeller itself evaluated against gold or manually annotated data?
3. Every comparator or baseline by exact name, and which of these classes are present:
   random or chance, majority class, structured data / metadata / diagnosis-code-only,
   simpler text model, human inter-annotator agreement, a different labeller.
4. Is a statistical test or confidence interval reported, and against what null?
5. Is positive-class prevalence of the gold or evaluation set reported, and what is it?
6. Is the number of gold positives reported, and how many?
7. Is any statistical power, sample-size justification or minimum detectable effect reported?

The instrument required a short verbatim supporting quote for every answer, or the token
UNCLEAR. Those quotes are retained in the `labeller_eval_evidence` and `notes` fields of
`survey.json` so that any single code can be rechecked against the source.

Limitations that must travel with any use of these numbers:

- **Single coder, no second rater, no coder agreement statistic.** The coding of this survey has
  exactly the defect the survey is about. It is a floor, not a proof.
- **Comparators were coded generously**, as present if they appear anywhere in the paper, at the
  labeller level or at the downstream-model level. This biases every comparator count upward and
  therefore makes the negative findings conservative.
- **Open-access bias.** Eligibility required machine-readable full text, which over-represents
  BMC, Nature Portfolio, Scientific Reports, JMIR, JAMIA Open, PMC author manuscripts, arXiv and
  the ACL Anthology, and under-represents paywalled Elsevier and Springer clinical journals.
  Three of the five full-text exclusions were paywalled or embargoed Elsevier or Thieme records.
- **The LLM-labeller arm is under-sampled**, because Search B (241 records) was not screened and
  26 of 67 arXiv records were not readable.
- Two pairs of records are companion papers from the same group on overlapping corpora
  (Banerjee 2019 JAMIA Open with Gori 2019 eGEMs; Agnikula Kshatriya 2020 IEEE BIBM with
  Agnikula Kshatriya 2021 BMC MIDM). A sensitivity analysis dropping one of each pair is reported
  in `survey_findings.md`.
