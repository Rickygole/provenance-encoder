# Literature findings: marginal-only certification, and the MCC margin ceiling

Every identifier below was resolved against Crossref content negotiation on the
DOI, on 2026-10-15. Nothing here is cited from memory.

## Q1. Is the ceiling result already published?

**Yes, and it is 67 years old.** The Matthews correlation computed on a
2x2 confusion matrix is numerically identical to Pearson's phi coefficient. This
was verified arithmetically on this paper's own numbers: the ceiling cells
12/211/0/30 give MCC 0.0818446503 and phi 0.0818446503, and the observed cells
11/212/1/29 give 0.0243260488 both ways.

The margin-constrained maximum of phi, phi_max, is classical:

- Cureton, "Note on phi/phi_max," Psychometrika 24(1):89-91, 1959,
  doi 10.1007/BF02289765. Title confirmed from the Crossref record.
- Guilford, "The Minimal Phi Coefficient and the Maximal Phi," Educational and
  Psychological Measurement 25(1):3-8, 1965, doi 10.1177/001316446502500101.

**Consequence for the manuscript.** Equation 1 is not a derivation. It is
phi_max restated in confusion-matrix notation. The paper must cite it at first
use and claim the transport into weak-labeller evaluation plus the class-level
consequence, not the bound. This has been done; see the Check 3 subsection.

## Q2. Does any published procedure certify on a marginal?

**Half one, the specific instrument: no.** No published procedure was found that
certifies a weak labeller by a hypergeometric tail test on its flag count.
Searched: Semantic Scholar Graph API, arXiv API, OpenReview notes API,
ACL-Anthology-framed searches, the crowdsourcing qualification-test literature,
and epidemiological screening-test methodology. The crowdsourcing and
LLM-annotator literature screens on fixed accuracy or agreement cutoffs, never on
a marginal tail test. The manuscript therefore states plainly that the gate is
the authors' own construction and that the result concerns the class of screens
that certify on a marginal.

**Half two, the class: yes, and this is the important finding.**

- van de Bijl, Klein, Pries, Bhulai, Hoogendoorn and van der Mei, "The Dutch
  Draw: constructing a universal baseline for binary classification problems,"
  Journal of Applied Probability 62(2):475-493, 2024, doi 10.1017/jpr.2024.52.
- Pries, van de Bijl, Klein, Bhulai and van der Mei, "The optimal
  input-independent baseline for binary classification: The Dutch Draw,"
  Statistica Neerlandica 77(4):543-554, 2023, doi 10.1111/stan.12297.

The Dutch Draw classifier is parameterised solely by the fraction predicted
positive, so its evaluation-measure values are functions of the item count, the
positive count and the flagged count alone. The authors recommend it explicitly
as a development sanity check. Its MCC instance has expectation zero for every
parameter value, which is an independently derived instance of the pathology this
paper argues about. This is the citation that answers the straw-man objection.

Caveat to record: the Dutch Draw paper gives no significance test, so it is
margin-only but not a tail test. Acceptance sampling is a tail test but consumes
item-level verified defectives, so it is not margin-only. The instrument analysed
here sits at the intersection, and nothing published was found there.

## Q3. Genealogy of the form

- Dodge and Romig, "A Method of Sampling Inspection," Bell System Technical
  Journal 8(4):613-631, 1929, doi 10.1002/j.1538-7305.1929.tb01240.x. For Type A
  finite-lot plans the operating-characteristic function is a hypergeometric
  tail, thresholded, with "in the tail" meaning accept. Establishes that
  certifying by a hypergeometric tail on a count is century-old standardised
  practice. ISO 2859-1 carries the published plans; ISO standards have no DOI and
  the internal clauses were not read, so they are not quoted.
- Lewis, Bromley, Sutton, McCray, Myers and Lancaster, "Determining sample size
  for progression criteria for pragmatic pilot RCTs: the hypothesis test strikes
  back!", Pilot and Feasibility Studies 7(1):40, 2021,
  doi 10.1186/s40814-021-00770-x. The closest published methodological analogue:
  a count-based pass/fail gate treated explicitly as a hypothesis test, with the
  sample size the decision requires.
- Ojala and Garriga, "Permutation Tests for Studying Classifier Performance,"
  ICDM 2009, pp. 908-913, doi 10.1109/ICDM.2009.108. The label-permutation null
  holds the class margins fixed, so on a 2x2 table with the three counts fixed
  that null is the hypergeometric null. The gate analysed here is the closed form
  of this published test.
- Combrisson and Jerbi, "Exceeding chance level by chance," Journal of
  Neuroscience Methods 250:126-136, 2015, doi 10.1016/j.jneumeth.2015.01.010.
  Certifies above-chance decoding by a binomial test on a correct count and shows
  the naive theoretical chance level misleads.

## What was not verified, and is therefore not cited

- Rivals et al., "Enrichment or depletion of a GO category: which test?" could
  not be resolved to a primary record and is not cited.
- No primary Fisher citation was pinned, so the standard identity between a
  one-sided hypergeometric tail test on a 2x2 overlap and Fisher's exact test is
  stated without a primary attribution or omitted.
- Non-English literature, textbooks, patent literature and the regulatory
  proficiency-testing corpus were not searched. Absence of evidence here is not
  proof of absence, and the manuscript says so.
