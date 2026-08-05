"""Medication-extraction dictionaries, reused VERBATIM from the reference notebook

    ACCORD_Phase2_Medication_Extraction (cells 7, 10, 12, 13 -- the final
    "v4" extractor that was actually run over the full cohort, producing
    hf_medication_mentions.csv / 219,683 mentions).

Do not edit ABBREV, NEGATION, ACTION_PATTERNS, HF_DRUGS, VARIANTS, ALL_DRUGS, or
resolve_action() -- the rebuild must stay comparable to the reference on this axis.
Everything below the "NEW" marker is added for this rebuild (dose parsing for
Rule 3, mostly) and was not in the reference.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# VERBATIM from ACCORD_Phase2_Medication_Extraction, cell 7
# ---------------------------------------------------------------------------

# Heart failure specific abbreviation expansion
ABBREV = {
    'metop': 'metoprolol', 'lopressor': 'metoprolol',
    'lasix': 'furosemide', 'dig': 'digoxin', 'digox': 'digoxin',
    'coumadin': 'warfarin', 'lisinop': 'lisinopril',
    'carv': 'carvedilol', 'coreg': 'carvedilol',
    'spiro': 'spironolactone', 'aldactone': 'spironolactone',
    'hydral': 'hydralazine', 'nitro': 'nitroglycerin',
    'amio': 'amiodarone', 'entresto': 'sacubitril-valsartan',
    'hctz': 'hydrochlorothiazide', 'asa': 'aspirin',
    'ntg': 'nitroglycerin', 'bblocker': 'beta-blocker',
    'acei': 'ace-inhibitor', 'arb': 'angiotensin-receptor-blocker'
}

# Negation patterns
NEGATION = re.compile(
    r'\b(no|not|without|discontinue|discontinued|stop|stopped|hold|held|'
    r'do not|do not give|avoid|contraindicated|never|off|off of|'
    r'not to take|not taking|not on|was not|is not|are not)\b',
    re.IGNORECASE
)

# Action context patterns
ACTION_PATTERNS = {
    'cessation': re.compile(
        r'\b(hold|held|stop|stopped|discontinue|discontinued|do not give|'
        r'avoid|contraindicated|off|not to take|no longer|cease|ceased|'
        r'withheld|not administered)\b', re.IGNORECASE),
    'continuation': re.compile(
        r'\b(continue|continued|maintain|maintained|keep|kept|resume|resumed|'
        r'ongoing|remain on|stay on|continue to take|to continue)\b', re.IGNORECASE),
    'initiation': re.compile(
        r'\b(start|started|initiate|initiated|begin|began|add|added|'
        r'new|newly|prescribed|started on|added to|commence|commenced)\b', re.IGNORECASE),
    'dose_modification': re.compile(
        r'\b(increase|increased|decrease|decreased|reduce|reduced|titrate|'
        r'titrated|taper|tapered|uptitrate|adjust|adjusted|dose change|'
        r'dose adjusted|dose increased|dose decreased|dose reduced)\b', re.IGNORECASE)
}

# ---------------------------------------------------------------------------
# VERBATIM from ACCORD_Phase2_Medication_Extraction, cell 10
# ---------------------------------------------------------------------------

# Known heart failure drugs to prioritize
HF_DRUGS = {
    'furosemide', 'lasix', 'metoprolol', 'carvedilol', 'lisinopril',
    'enalapril', 'ramipril', 'captopril', 'losartan', 'valsartan',
    'spironolactone', 'eplerenone', 'digoxin', 'warfarin', 'aspirin',
    'atorvastatin', 'simvastatin', 'amlodipine', 'hydralazine',
    'nitroglycerin', 'isosorbide', 'amiodarone', 'dobutamine',
    'dopamine', 'milrinone', 'sacubitril', 'entresto', 'ivabradine',
    'empagliflozin', 'dapagliflozin', 'heparin', 'enoxaparin',
    'apixaban', 'rivaroxaban', 'dabigatran', 'insulin', 'potassium',
    'magnesium', 'metolazone', 'torsemide', 'bumetanide', 'chlorthalidone',
    'hydrochlorothiazide', 'diltiazem', 'propranolol', 'atenolol',
    'bisoprolol', 'clonidine', 'nifedipine', 'nitroprusside',
    'morphine', 'hydromorphone', 'oxycodone', 'acetaminophen',
    'ibuprofen', 'prednisone', 'methylprednisolone', 'pantoprazole',
    'omeprazole', 'lansoprazole', 'metformin', 'glipizide', 'glargine',
    'lispro', 'novolog', 'humalog', 'lantus', 'colchicine', 'allopurinol',
    'trimethoprim', 'sulfamethoxazole', 'ciprofloxacin', 'vancomycin',
    'piperacillin', 'ceftriaxone', 'azithromycin', 'fluconazole'
}

# ---------------------------------------------------------------------------
# VERBATIM from ACCORD_Phase2_Medication_Extraction, cell 12
# ---------------------------------------------------------------------------

ALL_DRUGS = HF_DRUGS | set(ABBREV.keys()) | set(ABBREV.values())

# Add common spelling variants
VARIANTS = {
    'hydromorphnoe': 'hydromorphone',
    'metoprolol succinate': 'metoprolol',
    'metoprolol tartrate': 'metoprolol',
    'furosemide lasix': 'furosemide',
    'asa': 'aspirin',
    'acetylsalicylic': 'aspirin',
}

ALL_DRUGS.update(VARIANTS.keys())

# ---------------------------------------------------------------------------
# VERBATIM from ACCORD_Phase2_Medication_Extraction, cell 13
# ---------------------------------------------------------------------------


def resolve_action(action, negated):
    if not negated:
        return action
    # If negated, flip continuation to cessation and initiation to cessation
    if action == 'continuation':
        return 'cessation'
    if action == 'initiation':
        return 'cessation'
    # cessation negated means double negative, treat as continuation
    if action == 'cessation':
        return 'continuation'
    return action


# ---------------------------------------------------------------------------
# NEW for this rebuild. The reference never parsed a dose to a number -- it
# kept the raw matched string ("40mg") and threw away offsets, which is part
# of why Rule 3 (dosage conflict) has no source anywhere on disk: without a
# parsed numeric value there is nothing to compute a percent difference from.
# ---------------------------------------------------------------------------

# mg <-> mcg <-> g are mutually convertible (mass); mEq, units, ml are kept
# separate because converting them requires drug-specific molar mass / dilution
# information we do not have. A dose pair is only compared when both sides
# reduce to the same canonical unit.
_MASS_TO_MG = {'mg': 1.0, 'mcg': 0.001, 'g': 1000.0}

DOSE_RE = re.compile(
    r'(\d+(?:\.\d+)?)\s*(mg|mcg|mEq|units?|ml|g)\b', re.IGNORECASE
)


def parse_dose(dose_str: str):
    """Parse a raw dose string (e.g. '40 mg', '0.125mg', '5000units') into a
    (value, canonical_unit) pair, or None if it can't be parsed.

    canonical_unit is one of 'mg' (mass, already converted from mcg/g),
    'meq', 'unit', 'ml'. Two doses are only comparable (for Rule 3) if their
    canonical_unit matches.
    """
    if not dose_str or not isinstance(dose_str, str):
        return None
    m = DOSE_RE.search(dose_str)
    if not m:
        return None
    value = float(m.group(1))
    unit = m.group(2).lower()
    if unit in _MASS_TO_MG:
        return value * _MASS_TO_MG[unit], 'mg'
    if unit == 'meq':
        return value, 'meq'
    if unit in ('unit', 'units'):
        return value, 'unit'
    if unit == 'ml':
        return value, 'ml'
    return None


def dose_pct_diff(a: float, b: float) -> float:
    """Percent difference between two comparable dose values, relative to the
    larger of the two (symmetric, bounded in [0, 100])."""
    hi, lo = (a, b) if a >= b else (b, a)
    if hi == 0:
        return 0.0
    return (hi - lo) / hi * 100.0
