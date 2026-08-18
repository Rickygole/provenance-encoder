I have everything I need. Here is the spec.

---

# ACCORD Thesis — Visual System v3
## Buildable specification. Implement verbatim; nothing here is a suggestion.

---

## PROBLEM

The v2 system is typographically correct and visually absent. Its root cause is not execution — it is `/Users/rickygole/provenance-encoder/accord/thesis/figures/style.py`'s own stated philosophy: hairline spines, no fills, no frames, colour rationed to marks a few points across, and a hard 9pt type ceiling that leaves no figure with a headline. Every measured failure (10 of 14 figures under 10% ink, 1.5% mean saturated pixels, 0.24% on the pipeline schematic, four monochrome schematics, six float-damaged sheets) follows from those rules being obeyed. **v3 overturns them.** Done means: 12 figures, drawn as one designed set, each of which reads as a finished exhibit at arm's length and survives a hostile provenance audit unchanged.

**Assumption I had to make and am naming:** the beauty review's merge verdict takes the roster from 14 to **12**, not 13. The author authorised one cut (survey-audit); this spec cuts a second (`fig:disc-strata`) because it is a *literal duplicate* of Figure 8's middle family — the same two AUROC estimates, drawn twice, 60 pages apart. That is the one decision in this spec that needs the author's word before the build starts. Everything else follows from instructions already given.

---

## 1. THE v3 PALETTE

### 1.1 Structure of the colour model

Every hue is a **four-stop ramp**, and the stop determines the job. This is the whole colour system; there are no ad-hoc colours anywhere.

| Stop | Job | Carries entity identity? | Text on it |
|---|---|---|---|
| `DEEP` | text on paper; strokes on `TINT` fills; **solid fills carrying white text** | **yes** | `PAPER` |
| `CORE` | the mark: dots, lines, bars, arrows, strokes on `DEEP` | **yes** | n/a |
| `TINT` | area fills, ground bands, phase bands, region shading | **no — structural** | `INK` |
| `WASH` | very light grounds, second-order regions, band interiors | **no — structural** | `INK` |
| `BARE` | a point estimate with no committed interval (unchanged rule) | yes, weakly | n/a |

**The rule that makes this work, and it is load-bearing: identity lives in `DEEP` and `CORE` only. `TINT` and `WASH` are structural ground and carry no entity meaning.** This is what lets a schematic fill nine boxes with `COBALT_TINT` without any reader thinking those boxes are GPT-4o.

`DEEP` is defined as the stop with relative luminance ≈ 0.085. That single number buys both directions at once: ≥ 7.7:1 for the hue as text on white, **and** ≥ 7.7:1 for white text on the hue as a solid fill. One stop, two jobs, no second table.

### 1.2 Exact hexes — computed, not eyeballed

```
name       stop    hex        relL     on white   INK on it   grey(0.299/0.587/0.114)
COBALT     DEEP    #0A4DB1    0.0855   7.75:1     —            68.4
COBALT     CORE    #0B53C0    0.1006   6.97:1     —            73.9
COBALT     TINT    #BDD0EE    0.6210   —          11.35:1     205.7
COBALT     WASH    #DEE8F7    0.7996   —          14.37:1     230.7
COBALT     BARE    #98B6E4    0.4573   —           8.58:1     178.3

AZURE      DEEP    #20577D    0.0860   7.72:1     —            74.9
AZURE      CORE    #3FA9F5    0.3603   2.56:1     —           146.0
AZURE      TINT    #A1D5FA    0.6207   —          11.34:1     201.7
AZURE      WASH    #D1EBFD    0.8006   —          14.39:1     229.3
AZURE      BARE    #78C2F8    0.4935   —           9.19:1     178.0

EMERALD    DEEP    #005F44    0.0860   7.72:1     —            63.5
EMERALD    CORE    #00A97A    0.2978   3.02:1     —           113.1
EMERALD    TINT    #98DCC9    0.6208   —          11.35:1     197.5
EMERALD    WASH    #CEEFE5    0.8051   —          14.46:1     228.0
EMERALD    BARE    #75D0B7    0.5231   —           9.69:1     177.9

AMBER      DEEP    #714B00    0.0854   7.75:1     —            77.8
AMBER      CORE    #F5A300    0.4561   2.07:1     —           168.9
AMBER      TINT    #F9C765    0.6193   —          11.32:1     202.8
AMBER      WASH    #FCE4B6    0.7956   —          14.30:1     229.9
AMBER      BARE    #F6AC1A    0.4917   —           9.16:1     177.5

PLUM       DEEP    #971D6B    0.0852   7.77:1     —            74.4
PLUM       CORE    #C3268A    0.1482   5.30:1     —            96.3
PLUM       TINT    #EEC2DE    0.6203   —          11.34:1     210.3
PLUM       WASH    #F7E1EF    0.7986   —          14.35:1     233.2
PLUM       BARE    #E296C6    0.4206   —           7.96:1     178.2

SCARLET    DEEP    #A41704    0.0851   7.77:1     —            63.0
SCARLET    CORE    #D81E05    0.1554   5.11:1     —            82.8
SCARLET    TINT    #F5C2BC    0.6163   —          11.27:1     208.6
SCARLET    WASH    #FAE2DF    0.8004   —          14.38:1     232.8
SCARLET    BARE    #EE9A8F    0.4327   —           8.16:1     177.9
```

**Structure tones** (no identity, ever):

```
PAPER      #FFFFFF   1.0000    —          —
INK        #16181D   0.0091   17.76:1     text, spines, ticks
GRAPHITE   #2E3440   0.0341   12.49:1     connectors; a MEASURED quantity that is not an entity
MUTE       #5A6072   0.1175    6.27:1     secondary annotation text
SLATE      #7C8AA5   0.2518    3.48:1     reference lines, seed dots, accounting stubs — never text
MIST       #DCE3EE   0.7633    —          neutral structural ground
HAIR       #EAEDF3   0.8453    —          the one optional value grid
PLATE_EDGE #C4CCDA   —         —          schematic enclosure stroke only
```

`SCARLET_CORE` is unchanged from v2 (`#D81E05`) — the author's accent survives the re-base. `COBALT` is the one hue re-based for a hard reason: at v2's `#0072B2` the blue and the scarlet printed at greyscale 87 and 83, indistinguishable in black-and-white on the same canvas (fig-power draws both). `#0B53C0` moves it to 73.9 and opens the gap to 8.9.

### 1.3 Colour-blind safety — measured, not asserted

Machado (2009) simulation at severity 1.0, minimum pairwise ΔE76 across the six hues **plus** `SLATE`:

```
normal   ΔE = 32.6   (AZURE vs SLATE)
protan   ΔE = 21.2   (COBALT vs PLUM)
deutan   ΔE = 14.7   (EMERALD vs SLATE)
tritan   ΔE = 15.0   (AZURE vs EMERALD)
```

Greyscale ladder of `CORE` stops, minimum adjacent gap **8.9**:
`COBALT 73.9 · SCARLET 82.8 · PLUM 96.3 · EMERALD 113.1 · SLATE 136.9 · AZURE 146.0 · AMBER 168.9`

The deutan `EMERALD/SLATE` pair at 14.7 is the weakest link and is handled structurally, not chromatically: **`SLATE` is only ever a dashed hairline or a small open seed dot; it never takes the same mark type as an entity hue on the same axes.** That constraint is enforced in verification (§11.6).

The v2 guarantee stands and is strengthened: hue is the fast read; every series is *also* directly labelled at its own mark in its own `DEEP` tone; where the grammar permits a marker other than a dot, marker shape separates as well.

### 1.4 Entity map, re-based

```python
ENTITY_HUE = {
    "gpt4o_zeroshot":       COBALT,
    "gpt4o_fewshot":        AZURE,     # same model, different prompt -> same family, lighter value
    "accord":               AMBER,
    "llama3_8b":            PLUM,
    "weak_labeller":        PLUM,      # the production weak labeller IS Llama 3 8B
    "pharmacist_truth":     EMERALD,
    "pretrained_reference": EMERALD,
    "qwen3_8b":             COBALT,    # no GPT-4o series on that canvas
    "granite33_8b":         AMBER,     # NEW. no ACCORD series on that canvas
    "control_model":        EMERALD,
}
NON_ENTITY_MEASUREMENT = GRAPHITE      # alpha params, MDE curve, coded-corpus counts
```

`granite3.3:8b` is a new binding and it fixes a live bug: today it draws in `GREEN`, which in the entity map means *pharmacist truth*. A weak labeller wearing the truth colour is exactly the kind of thing this thesis is about, and it must not appear in this thesis.

### 1.5 Fill / stroke pairing — three legal combinations, no others

| # | Fill | Stroke | Text | Where |
|---|---|---|---|---|
| **F1 TINT PANEL** | `X_TINT` | `X_DEEP`, 1.5pt | `INK` | stage boxes, phase bands, region grounds |
| **F2 SOLID CHIP** | `X_DEEP` | `X_CORE`, 1.4pt (a bright rim) | `PAPER`, bold permitted | verdict chips, terminal boxes, the one loud slot |
| **F3 WASH GROUND** | `X_WASH` | none | `INK` or `X_DEEP` | second-order regions, band interiors, over/under wedges |

Any fourth combination is a bug. A `CORE` stop is **never** an area fill (it is a mark colour and at area scale it screams). A `TINT` never carries white text.

### 1.6 The one scarlet semantic

Scarlet's meaning currently shifts across five figures (this study / the defect / the clinical point / ACCORD / the operating point). One semantic from here:

> **SCARLET is the failure this thesis is reporting: the exact mark where the instrument or the pipeline is broken.**

All seven current uses survive that reading unchanged — the failed stages, the 22× multiplier, the underpowered evaluation core, the wrong exemplar slot, the 4%-read-as-85% operating point, the underpowered E9 audit point, and ACCORD sitting on the indifference line buying nothing. Nothing needs reassigning. What *does* change is that **cohort-flow's exclusion stubs lose scarlet** — an exclusion is accounting, not failure — and take `SLATE`.

Budget: **at most one saturated scarlet element (`CORE` or `DEEP`) per axes.** Per *axes*, not per canvas, so a two-panel figure may carry two. `SCARLET_TINT` and `SCARLET_WASH` are grounds and are outside the budget.

**Schematic exception, stated so it cannot be read as drift:** on a schematic, scarlet encodes a *state*, and a state is data, so it repeats (four failing stages on Figure 1 are four scarlet stages). On a plot, scarlet is a single claim mark. This is the only bifurcation in the system and it is what requirement 3 — "coloured verdict states" — actually asks for.

**Figures 8 and 9 carry no scarlet at all.** Figure 8's claim is a shape (everything crosses zero) and Figure 9's claim is a success. That is the correct outcome, not a missing accent, and it is stated here so the critic reads it as a decision.

### 1.7 No alpha. Anywhere.

Every translucent effect in the v2 set becomes a precomputed tint hex. Transparency is unpredictable across the PDF backend, the notebook's inline PNG, and the printer, and it is the direct cause of failure 8's washed-viridis collision on Figure 10. `alpha=` may not appear in any drawing script. Verification greps for it.

---

## 2. TYPOGRAPHY

### 2.1 Face — STIX, and the reason

Figures are currently DejaVu Serif against a Times body (`mathptmx`). The mismatch is visible on every sheet. **`STIXGeneral` is the fix**: it is a Times-metric-compatible face and it ships inside matplotlib's own `mpl-data/fonts/ttf` (confirmed present in the installed 3.9.2: `STIXGeneral.ttf`, `-Bol`, `-Italic`, `-BolIta`, plus the STIX sizing fonts). It therefore renders identically on this machine and in Colab, which was the entire reason DejaVu was pinned.

```python
"font.family": "serif",
"font.serif": ["STIXGeneral"],
"mathtext.fontset": "stix",
"pdf.fonttype": 42,
"axes.unicode_minus": False,
```

STIX is *narrower* than DejaVu Serif, so every spacing decision already checked under DejaVu has more room, not less. That is the safe direction, but §11.3 still re-checks overlap on every canvas.

### 2.2 Scale — the figures get a hierarchy for the first time

Every figure is drawn at its printed size (scale 1.0), so drawn points are page points.

| Role | Size | Colour | Weight |
|---|---|---|---|
| **HEADLINE** | **13.0pt** | `INK` | regular |
| Big claim numeral (only where specified) | **22.0pt** | claim `DEEP` | **bold** |
| Deck / sub-line under headline | 9.5pt | `MUTE` | regular |
| Axis label | 9.5pt | `INK` | regular |
| Tick label | 9.0pt | `INK` | regular |
| Direct series label at its mark | 9.0pt | that series' `DEEP` | regular |
| Secondary annotation | 8.5pt | `MUTE` | italic |
| Panel letter | 10.0pt | `INK` | italic |
| Text inside a SOLID chip | 9.0pt | `PAPER` | **bold** |
| Floor | **8.5pt** | — | — |

The headline is 13pt against a 12pt body, so it sits *above* the body in the hierarchy and the figure stops reading as an inclusion. **Bold is permitted in exactly two places: inside SOLID chips, and the big claim numeral.** Nowhere else. This deliberately overturns v2's "no bold anywhere."

### 2.3 The headline is a finding, not a title

Every figure gets one headline line stating **what the figure found**, never what it plots. The caption never repeats it. The chapter's `\caption[...]` short form stays as-is for the List of Figures. Headlines are specified per figure in §5.

`SHOW_INFIGURE_TITLES` in `make_figures.py` becomes permanently `True` and stops being a flag.

### 2.4 Panel letters — one rule

Italic 10pt `INK`, drawn in **figure coordinates**, x = the axes box's left edge (`ax.get_position().x0`), y = 5pt above the axes box top edge. Always outside the frame, never inside, never in data coordinates. Where a headline is present the panel letters sit *below* it. One helper, `style.panel_label(fig, ax, "(a)")`; no figure computes this itself.

### 2.5 Casing

Any text or line label sitting on a filled field is cased: `path_effects.withStroke(linewidth=2.8, foreground="#FFFFFF")`. Casing applies over `TINT`, `WASH`, and heatmap fields only. Text on a SOLID chip is white and is never cased. The v2 cased-interval rule on bars survives unchanged and is now a special case of this one.

### 2.6 Bans that hold

No en dash, em dash, or U+2212 on any canvas. No file name, register key, or repository path on any canvas. `.iloc[0]` on Figure 4 is neither of those, but it reads as pasted code in a bound thesis: **on-canvas it becomes "selected by position (row 1)"; the literal `.iloc[0]` moves to body text**, where it already appears.

---

## 3. THE FRAME SYSTEM

Nothing in the v2 set is bounded. Two enclosures, and **exactly one per figure**.

### 3.1 Plots — full boxed axes

```python
"axes.spines.top": True, "axes.spines.right": True,
"axes.linewidth": 0.9, "axes.edgecolor": INK,
"xtick.major.width": 0.9, "ytick.major.width": 0.9,
"xtick.major.size": 3.2, "ytick.major.size": 3.2,
"xtick.direction": "out", "ytick.direction": "out",
```

All four spines, 0.9pt `INK`, ticks outward. **`set_bounds()` clipping is removed everywhere** — a closed box cannot be clipped and the two ideas fight.

This retires v2's rule that "a spine is drawn only on an axis that measures something," and the retirement is principled, not a concession: that rule existed because a lone vertical rule beside four unrelated row labels reads as a *bracket* grouping them. A closed rectangle does not read as a bracket. **Closure, not extent, now carries the meaning**, so a categorical axis inside a box is safe. `style.lean()` is deleted; `style.frame(ax, grid=False)` replaces it.

Optional value grid: `HAIR #EAEDF3` at 0.6pt, `axes.axisbelow=True`. Rule: **a plot read for values gets the grid; a plot read for position or shape does not.** Specified per figure in §5; the builder never decides.

Multi-panel: every panel gets its own box. No outer plate on a plot.

### 3.2 Schematics — the plate

A rounded rectangle covering the full canvas, inset 1.5pt from every edge, corner radius 4pt, fill `PAPER`, stroke `PLATE_EDGE #C4CCDA` at 1.0pt. No inner axes box.

### 3.3 The signature rule — WITHDRAWN BY THE AUTHOR

This section specified a **3.0pt-tall solid bar flush to the top inside edge of every canvas**, in that figure's dominant `DEEP` hue, and claimed it was "the cheapest possible unification and the single thing that will make the bound set read as designed rather than assembled."

**It is withdrawn. There is no bar on any figure.** The author read the bound draft and asked for it off figures 2, 4, 7, 9, 10, 11 and 12 by name — *"the line above which shouldn't be there"*, *"the red line from the top border"* — and it comes off all twelve, because a set in which six figures carry a coloured rule and six do not is worse than either.

The claim above was wrong in a specific way worth recording, because it is the kind of claim this spec makes often. On screen, at 100%, the bar reads as a masthead. On a bound page at 0.98 of the measure it is a 3pt coloured line hanging under the top of a float with nothing beneath it but the white above the headline, and it reads as printer's registration rather than as design. The unification is done by what was already doing it: one headline at 13pt in `INK`, one type scale, one palette, one frame system.

`style.signature(fig, hue_deep, headline, deck)` survives with its whole signature intact and still sets the headline and the deck. `hue_deep` is now unused and is deliberately kept: every call site names the figure's dominant hue, which is the one place each figure declares what it is built on. The 3.0pt the bar occupied is given back — the headline now sits `PAD_HEADLINE` = 7.0pt from the top edge rather than 10.0pt, and each figure's top axes margin is 3pt smaller than it was.

### 3.4 Line weight table — the whole set, no exceptions

```
plate stroke          1.0pt   PLATE_EDGE
axes spine / tick     0.9pt   INK
value grid            0.6pt   HAIR
data line             1.8pt
reference line        1.0pt   SLATE, dash (0,(4,2.4))
interval line         1.5pt,  cap half-width 3.2pt, capthick 1.5pt
marker, dot           6.6pt
marker, non-dot       6.2pt
bare marker           6.6pt open, 1.2pt edge in X_BARE
seed dot              3.2pt   SLATE, solid (no alpha)
schematic node stroke 1.5pt   X_DEEP  (F1)  /  1.4pt X_CORE (F2)
connector             1.4pt   GRAPHITE
bar outline           1.0pt   X_DEEP        <- NEW; v2 bars had no outline
```

### 3.5 Arrowheads

matplotlib's default micro-heads are replaced everywhere:

```python
FancyArrowPatch(a, b,
    arrowstyle=ArrowStyle("-|>", head_length=0.62, head_width=0.30),
    mutation_scale=12, lw=1.4, shrinkA=0, shrinkB=0,
    joinstyle="miter", capstyle="butt", color=GRAPHITE, zorder=3)
```

Target: head ≈ 7.4pt long, ≈ 7.2pt wide, solid-filled in the stroke colour. Wrapped as `style.arrow(ax, xy_from, xy_to, color=GRAPHITE)`; no script builds one by hand. §11.5 verifies head size on a render.

### 3.6 No leader lines. Anywhere.

Labels sit adjacent to their marks. Where adjacency collides, **the axis range expands or the mark moves** — the label never grows a leader. This one rule resolves four of the eight overlap residues at once (power's scarlet leader, met-exchange's two long leaders, threshold-surface's contour callouts, exemplar's two annotation leaders). The two short horizontal row-connectors on Figure 4 are not leaders: they are 1.1pt rules on the row's own baseline, which is a different object.

---

## 4. RULES OVERTURNED — the explicit list

The builder must know which v2 docstring paragraphs are now dead, because they are written into `style.py` as prose and will otherwise be honoured by reflex.

| v2 rule | v3 |
|---|---|
| Left and bottom spines only | **All four. Boxed.** |
| Spine only on an axis that measures | **Dead.** Closure replaces extent. |
| No bold anywhere | **Bold in SOLID chips and the claim numeral only.** |
| Nothing above 9pt | **13pt headline; 22pt claim numeral.** |
| Bars have no outlines | **Bars take a 1.0pt `X_DEEP` outline.** |
| No legends at all | **Holds.** Direct labelling only. |
| Colour reserved for marks | **Dead.** Colour is a carrying channel: fills, grounds, phases, states. |
| No figure carries a title | **Dead.** Every figure carries a finding headline. |
| DejaVu Serif pinned | **STIXGeneral pinned.** |
| `\apanote` under three captions | **Killed globally.** |
| Bare estimates lightened | **Holds**, re-based to `X_BARE`. |
| Wilson-only intervals, committed data only | **Holds. Absolutely.** |
| Scarlet is one claim element | **Holds, per axes**, with the stated schematic-state exception. |

---

## 5. PER-FIGURE DIRECTIVES

Ink targets use two measured quantities, defined once so builder and critic measure the same thing:

- **coverage** = fraction of pixels in the saved bounding box differing from `#FFFFFF` by > 2/255 in any channel, at 300 dpi.
- **chroma** = fraction of pixels with HSV saturation ≥ 0.25 and value ≥ 0.20.

Set-wide floors, gating: **coverage ≥ 0.28 and chroma ≥ 0.10 on every figure; set-mean chroma ≥ 0.18.**

---

### Fig 1 — `fig-pipeline-status` · ch1 · FULL REBUILD
`make_figures.py:265`

**Diagnosis.** The diagram occupies 40% of the canvas with an 8pt grey table beside it; the traffic-light grammar dies after two rows; coloured ink is 0.24%.

**Rebuild.** Kill the box column entirely. Each stage becomes a **full-canvas-width row band**, 0.42in tall, 3pt paper gutters between:

- far left: stage index `1`–`6`, 10pt, in the row's state `DEEP`
- left third: stage name, **10.5pt `INK`**
- middle: the **verdict chip** (F2 SOLID), rounded r=3pt, 9pt bold `PAPER` text, carrying only the verdict *word* (`Works` / `Unvalidated` / `Fails` / `Falsified`)
- the numeric detail (`47.4%`, `22x`, `AUROC 0.515, 95% CI [0.32, 0.70]`) sits right of the chip in 9pt `INK`
- right third: diagnostic, 8.5pt `MUTE` italic

**State ramp** — band fill F1, chip fill F2:

| verdict | band | chip |
|---|---|---|
| passed a diagnostic | `EMERALD_TINT` / `EMERALD_DEEP` stroke | `EMERALD_DEEP` |
| no diagnostic ever ran | `MIST` / `SLATE` stroke | `SLATE` |
| a diagnostic faulted it | `SCARLET_TINT` / `SCARLET_DEEP` stroke | `SCARLET_DEEP` |

No connectors — top-to-bottom order plus the index column carries sequence, and six arrows on a full-width band chart is noise.

Plate. (No signature bar: §3.3 is withdrawn.)
**Headline:** *Five of six stages failed the diagnostic that was run on them, or had none.*
**Draw** 5.854 × 3.39, `width=0.98\linewidth`. **Class B** (block 4.83in).
**Targets:** coverage ≥ 0.72, chroma ≥ 0.40.

---

### Fig 2 — `fig-intro-headline` · ch1 · PANEL (b) DELETED, (a) PROMOTED
`make_figures.py:411`

**Panel (b) is deleted.** Its four MCC rows are Figure 8's first family, drawn identically 85 pages apart. Figure 8 is now the one place those estimates live.

**Panel (a)** becomes a full-width single panel, boxed axes, no grid (read for shape, not values):

- three bars, `PLUM` / `PLUM` / `EMERALD` per the entity map, `CORE` fill + 1.0pt `DEEP` outline, 0.62 of the slot
- cased Wilson intervals, unchanged rule
- counts and rates at each bar's right end, 9pt in the bar's `DEEP`
- **the claim numeral:** `22x` set at **22pt bold `SCARLET_DEEP`** at the right, with a `SCARLET_CORE` bracket spanning only the two gold-set bars, and a 9pt `SCARLET_DEEP` sub-line *the weak labeller's marginal over the pharmacist's, same 299 pairs*
- ground: a `SCARLET_WASH` band spanning the full plot height from x = the pharmacist rate to x = the weak-labeller gold rate — the gap *is* the finding, so draw it

Dominant hue `PLUM_DEEP`, named at the `signature()` call. (No bar: §3.3 is withdrawn.)
**Headline:** *The weak labeller calls 87% of gold-set pairs contradictions. The pharmacist calls 4%.*
**Draw** 5.854 × 2.78, `0.98`. **Class A** (block 4.22in). Panel letters removed (single panel). `xlim` closes from 1.86 to 1.56: the old right third held nothing and the bars take it.
**Targets:** coverage ≥ 0.34, chroma ≥ 0.16.

---

### Fig 3 — `fig-cohort-flow` · ch3 · FULL REBUILD
`make_figures.py:1298`

**Diagnosis.** Nine white boxes, black strokes, one red box; marooned alone on p.54 with a 187-word caption.

**Rebuild** — a true branching flow with three phase groups:

| rows | phase | band |
|---|---|---|
| 1–4 | cohort construction | `MIST` band, `SLATE` phase rail at left |
| 5–7 | extraction and mining | `COBALT_TINT` band, `COBALT_DEEP` rail |
| 8 | pharmacist annotation | `EMERALD_TINT` band, `EMERALD_DEEP` rail |
| 9 | the evaluated core | **F2 SOLID `SCARLET_DEEP`**, `SCARLET_CORE` rim, white bold text |

- phase name set once in the left margin, 8.5pt `MUTE`, rotated 90°, beside its rail
- stage boxes F1 in their phase tint, `DEEP` stroke 1.5pt, `INK` text, count on its own line at 9.5pt
- trunk connectors: `style.arrow`, `GRAPHITE`
- **exclusion stubs** leave the trunk in the gap between stages and run right into a 34%-width gutter: `SLATE` arrow, and a `MIST` chip (F1, `SLATE` stroke) carrying the count in `GRAPHITE`. **Not scarlet** — an exclusion is accounting, not failure, and scarlet now means one thing.

Plate. (No signature bar: §3.3 is withdrawn.)
**Headline:** *13,608 admissions in; 253 pairs and 12 confirmed contradictions out.*
**Draw** 5.252 × 3.86, `width=0.88\linewidth`. **Class B** (block 5.30in). Height REDUCED from the 4.18 a proportional growth gives: 4.18 blocks at 5.62in, inside the forbidden band. The narrower class is still what un-maroons p.54.
**Targets:** coverage ≥ 0.60, chroma ≥ 0.32.

---

### Fig 4 — `fig-exemplar` · ch4 · FULL REBUILD
`make_figures.py:610`

**No synthetic clinical sentences.** A thesis about instruments that mislead, built on a MIMIC-III data agreement, does not put invented clinical text on a canvas where a reader could take it for a record. Abstract slots, coloured loudly. That is the decision; do not revisit it.

**Rebuild:**

- two labelled columns, five slot rows
- nursing slots M1–M5: F1 `COBALT_TINT` / `COBALT_DEEP`
- discharge slots CONT ×4: F1 `MIST` / `SLATE`
- **the CESS slot: F2 SOLID `SCARLET_DEEP`, `SCARLET_CORE` rim, 9pt bold `PAPER`** — the one loud element
- a `SCARLET_WASH` band spanning both columns across the CESS row, so the mismatch reads as a *row*
- the trigger bracket over the two column heads: `COBALT_CORE` 1.8pt with a `COBALT_DEEP` 9pt label *the rule quantifies over the whole set*
- right-hand annotations align to their rows, joined by a 1.1pt horizontal rule on the row baseline. No diagonals, no leaders.
- the position-selection annotation reads **"selected by position (row 1)"**

Plate. (No signature bar: §3.3 is withdrawn.)
**Headline:** *The rule fires on the set. The exemplar shown to everyone is row 1.*
**Draw** 5.252 × 2.70, `width=0.88\linewidth`. **Class A** (block 4.14in).
**Targets:** coverage ≥ 0.48, chroma ≥ 0.26.

---

### Fig 5 — `fig-prevalence-response` · ch4 · RESKIN + DECLUTTER
`fig_batch2.py:1–108`

- 2 × 3 grid, every panel boxed, no grid lines
- **fill the calibration wedge:** the region above the diagonal in every panel takes `MIST` (over-prediction), labelled once in the top-left panel only, 8.5pt `MUTE`, cased. Six panels of empty white become six panels of read structure.
- ~~ylim to **0 – 1.0** in all six~~ — **OVERTURNED BY THE AUTHOR.** *"needs to be more clear use some other vibrant colors."* The 0–1.0 axis is the root cause of the illegibility and it goes. The tallest point on the sweep is 0.17 and the highest Wilson bound anywhere on the canvas is 0.256, so nine tenths of the plot height was holding one off-sweep annotation and every measured series lay flat in the bottom quarter. **ylim is now 0 – 0.38 against an x of 0 – 0.32**, which puts perfect calibration on a diagonal at roughly 45° and makes "does not track prevalence" a shape rather than an assertion. No point was dropped and no interval clipped.
- column headings in their entity `DEEP`: `llama3:8b` `PLUM_DEEP`, `granite3.3:8b` `AMBER_DEEP`, `qwen3:8b` `COBALT_DEEP`
- series lines **2.4pt** `CORE`, markers **7.4pt** faced `CORE` and rimmed `DEEP` at 1.0pt, intervals **1.6pt**, caps 3.6pt. This is the one place in the set that departs from the §3.4 weight table, and it is the author's "vibrant" answered in weight. **No hue was re-based**, and the arithmetic is recorded in `style.py`: `COBALT_CORE` already sits at the greyscale ladder's minimum adjacent gap of 8.9 against `SCARLET`, so brightening it costs the black-and-white guarantee Figure 7 depends on; `AMBER_CORE` is already the lightest stop on the ladder at 2.07:1, so brightening it makes granite *less* legible; and PLUM is one of three, so moving it alone unbalances the row. The palette is colour-blind verified and it stays.
- *perfect calibration* sits on the diagonal itself, rotated to the line's angle, `MUTE` 8.5pt, cased. It moves from panel (1,3) to panel (2,3), and *over-prediction* moves from panel (1,1) to panel (1,3): panel (1,1) now carries the clinical anchor and nothing else, and the two reference labels take the emptiest panels in the grid
- **the clinical anchor** stays in panel (1,1) and its label stays one cased `SCARLET_DEEP` line, *clinical: 4% read as 85%*. Its true position is (0.04, 0.85), which the new axis does not reach, so it is drawn as what it is: **a `SCARLET_CORE` 2.4pt arrow at its own x = 0.04, starting above the highest bound in the panel and leaving through the top**, with both of its coordinates in the label beside it. **No marker sits under the arrow.** A square drawn at a y the point does not have would be a false position however it is labelled, and this figure is in a thesis about instruments that mislead. The caption changes to match: *"The scarlet arrow marks this thesis's clinical operating point, which lies above the top of its panel."*

Dominant hue `SCARLET_DEEP`, named at the `signature()` call. (No bar: §3.3 is withdrawn.)
**Headline:** *No labeller tracks prevalence. Stating the base rate does not fix it.*
**Draw** 5.854 × 3.78, `0.98`. **Class B** (block 5.22in). Taller than proportional, to square the panels up under the new y axis.
**Targets:** coverage ≥ 0.40, chroma ≥ 0.14.

---

### Fig 6 — `fig-falsification` · ch4 · FULL REBUILD
`make_figures_extra.py:~85–181`

The most dramatizable result in the thesis, currently four dots on white.

**Panel (a) — dumbbell.** For each of α_D and α_N: an open 6.6pt `GRAPHITE` ring at the initialisation, a solid 6.6pt `GRAPHITE` dot at the final value, a 2.4pt `GRAPHITE` connector between them, and the three seeds as 3.2pt `SLATE` dots on their own sub-row. Behind both rows, a vertical tolerance band. Row labels carry the initialisation.

**DECLUTTERED ON THE AUTHOR'S READING:** *"looks very busy make it clear."* Panel (a) carried six kinds of thing on two rows of data, and two of them said one sentence twice — the band label *"neither left a 2% band"* and the row labels *"+0.4%"* and *"+1.5%"*. **The prose went and the measurements stayed.** The band is named on the x axis instead, where a tolerance belongs and where the two tick labels it runs between already read 0.98 and 1.02. The movement labels move from 8.5pt `MUTE` to **9pt `GRAPHITE`**, the tone of the dumbbell they label. The band moves from `SCARLET_TINT` to **`SCARLET_WASH`**: it covers three quarters of the panel, and at `TINT` the ground was louder than the two grey dumbbells it exists to locate. The seed dots stay — three seeds landing on each other is evidence, not decoration.

**Panel (b) — cliff.** Three filled columns, F1→F3 gradient of drama: HF in-domain at 0.96 in `COBALT_TINT` fill / `COBALT_DEEP` stroke; sepsis 0.07 and AKI 0.06 in `COBALT_BARE` fill / `COBALT_DEEP` stroke (bare = no committed interval, per the surviving rule). Behind the two transfer columns, a `SCARLET_WASH` ground labelled *not fitted here*. The drop is a **`SCARLET_CORE` 2.2pt arrow patch** from the fitted column's top right corner to just above the first column it was not fitted on — a patch, **never an en dash and never a `->` on canvas**. Candidate counts in each column's foot, 8.5pt `MUTE`.

**~~`0.96 → 0.07` set at 22pt bold `SCARLET_DEEP`~~ — DELETED ON THE AUTHOR'S READING.** The y axis beside it already runs 0 to 1.00 in quarters with the two columns sitting on exactly those values: the same two numbers, twice, a centimetre apart, and it is most of what made the panel busy. The drop survives as pure geometry and still reads at thumbnail size — a column at the top of the frame, a scarlet line falling almost to the floor, two columns that are barely there. The ground label drops from `SCARLET_DEEP` to `MUTE`, which leaves the arrow as this panel's one saturated scarlet element and brings it back inside the §1.6 budget it had been over.

Cohorts are not entities. **All three columns take one hue.** The current SKY/GREEN/PURPLE assignment applies the entity map to things that are not entities and must go.

Both panels boxed; panel letters per §2.4. Dominant hue `SCARLET_DEEP`, named at the `signature()` call. (No bar: §3.3 is withdrawn.)
**Headline:** *Two parameters reported as learned did not move; the rule head does not transfer.*
**Draw** 5.854 × 2.68, `0.98`, panels side by side. **Class A** (block 4.12in).
**Targets:** coverage ≥ 0.42, chroma ≥ 0.20.

---

### Fig 7 — `fig-power` · ch4 · RESKIN, FILL MUST READ
`fig_batch2.py:~334–395`

- boxed axes, no grid
- **detectable region** `EMERALD_WASH`; **not-detectable region** `SCARLET_WASH`. Two grounds, hard boundary at the curve. The current `PALE` fill does not read.
- MDE curve 1.8pt `GRAPHITE` (a non-entity measurement)
- **this study:** 6.6pt `SCARLET_CORE` dot, **enlarged**, with its label sitting directly above-left, cased, 9pt `SCARLET_DEEP`. **No leader.**
- mortality control: 6.2pt `SLATE` square, label directly above, cased, `MUTE`. **No leader.**
- region labels *detectable* / *not detectable at this sample size* set 9pt cased in `EMERALD_DEEP` / `SCARLET_DEEP` at the far corners of their own regions

Dominant hue `SCARLET_DEEP`, named at the `signature()` call. (No bar: §3.3 is withdrawn.)
**Headline:** *At 12 positives, nothing below AUROC 0.735 was ever detectable.*
**Draw** 5.854 × 2.85, `0.98`. **Class A** (block 4.29in). Height REDUCED from the 3.00 a proportional growth gives, which would have made this a Class B float.
**Targets:** coverage ≥ 0.62, chroma ≥ 0.18.

---

### Fig 8 — `fig-forest` · ch4 · PROMOTED TO MASTER SUMMARY
`fig_batch2.py:~216–332`

This is now the single place every headline estimate in the thesis appears. Figures 2(b), 10 and 11(b) die into it.

- **Family bands:** each of the three families gets a full-plot-width ground band — family 1 `MIST`, family 2 `EMERALD_WASH`, family 3 `AMBER_WASH`. Family headings 9pt `MUTE` italic on the band. **Headings lose their fake hues** — a family is not an entity.
- **Rows take their entity hues**, which today they do not (family 1's four rows all draw `BLUE`, contradicting Figure 2(b) 85 pages earlier):
  - GPT-4o zero-shot `COBALT` · GPT-4o few-shot `AZURE` · ACCORD `AMBER` · Llama 3 8B `PLUM`
  - defect-free stratum `EMERALD` · decontaminated ensemble `AMBER`
  - both pretraining-gap rows `EMERALD`
- zero line: **1.2pt solid `GRAPHITE`** (not dashed grey). It is the spine of the figure, not a reference to be looked past.
- markers 6.6pt, intervals 1.5pt, caps 3.2pt, each in its row's `CORE`
- **NEW: a right-hand numeric column**, 9pt, `estimate [lo, hi]` per row, right-aligned inside the box, in the row's `DEEP`. This is what makes it a master summary and it is also the fix for the sliver problem.
- **Sliver rule, mandatory:** any interval whose drawn half-width is under 1.5pt is additionally drawn as a 1.5pt-wide `X_DEEP` vertical tick pair exactly at its bounds, and its numeric column entry is prefixed with a 6pt `MUTE` marker glyph `‡` plus a single 8.5pt `MUTE` footline inside the box: *‡ interval narrower than the marker*. The AG News row triggers this. An invisible interval sitting beside visible ones, with no cue, is exactly the failure mode this thesis's first rule exists to prevent.
- boxed axes, **value grid ON** (this plot is read for values)
- **no scarlet.** Stated in §1.6.

Dominant hue `GRAPHITE`, named at the `signature()` call. (No bar: §3.3 is withdrawn.)
**Headline:** *Every headline estimate in this thesis crosses zero.*
**Draw** 5.854 × 3.71, `0.98`. **Class B** (block 5.15in).
**Targets:** coverage ≥ 0.46, chroma ≥ 0.15.

---

### Fig 9 — `fig-scaling` · ch5 · PANEL (b) DELETED
`fig_batch2.py:~110–214`

**Panel (b) is deleted.** It is two rows on a full-width canvas, and its content — the mortality-control gap of +0.0031 with its interval — is Figure 8's last row, now printed as a number in the new numeric column. The deletion also removes both of the residues logged against this figure: the floating *gap +0.0031* and the band-label touch.

Single panel, boxed, value grid ON:

- from-scratch arm `COBALT`, 1.8pt line, 6.6pt markers, 1.5pt caps
- pretrained band: `EMERALD_WASH` fill, `EMERALD_CORE` 1.8pt centre line
- **the pretrained label moves inside the box**, right-aligned at the line's right end, one line, `EMERALD_DEEP`, cased; xlim extends 8% right to buy the room
- gap labels under each point stay, 8.5pt `MUTE`
- **no scarlet** (the claim is that the harness works)
- panel letters removed (single panel)

Dominant hue `EMERALD_DEEP`, named at the `signature()` call. (No bar: §3.3 is withdrawn.)
**Headline:** *The from-scratch arm rises with size and never reaches the pretrained band.*
**Draw** 5.854 × 2.84, `0.98`. **Class A** (block 4.28in).
**Targets:** coverage ≥ 0.36, chroma ≥ 0.14.

---

### Fig 10 — `fig-threshold-surface` · ch5 · ENCODING FIX
`fig_threshold_surface.py`

**The real bug:** the "washed" viridis in the underpowered region is *the same channel as the data*, so a washed dark value is indistinguishable from an unwashed pale one. The region must move to a different channel.

- **underpowered region: flat `MIST` overlay + 45° hatching in `SLATE` at 0.7pt, with a hard 1.2pt `GRAPHITE` boundary.** No alpha, no desaturation. Values under it are deliberately unreadable, which is the honest statement: that region cannot answer.
- **all three on-field labels move out of the field.** The two contour labels (`0.05 tolerance`, `power limit`) go to the right margin at the y where their contour exits the plot, 9pt, horizontal, in `GRAPHITE`. The underpowered note goes to a single 9pt `MUTE` line immediately beneath the x-axis label. Nothing rotated, nothing over data.
- **the stray caret glyph at the lower-left is a contour-label artifact and must be eliminated** — suppress inline contour labelling entirely (`clabel` off) now that labels are marginal.
- E9 operating point: `SCARLET_CORE` 1.8pt line, label above it, cased, `SCARLET_DEEP`. One scarlet element.
- boxed axes; colourbar gets a 0.9pt `INK` frame and 9pt ticks
- viridis is retained: it is sequential and CB-safe, and it is the only heatmap in the set

Dominant hue `SCARLET_DEEP`, named at the `signature()` call. (No bar: §3.3 is withdrawn.)
**Headline:** *The audit passed at an operating point where nothing was detectable.*
**Draw** 5.854 × 2.85, `0.98`. **Class A** (block 4.29in). Height REDUCED from proportional for the same reason as Figure 7; the width takes the growth and the heatmap field goes from 3.28in to 3.77in.
**Targets:** coverage ≥ 0.90, chroma ≥ 0.55 (heatmap; already the densest figure).

---

### Fig 11 — `fig-disc-actionability` · ch5 · FULL REBUILD
`make_figures.py:1214`

The hairline axis becomes a **fat timeline**: a 16pt-tall horizontal band split into three coloured phases, F1 fills, `INK` phase labels *inside* the band, cased.

| phase | fill / stroke | label inside |
|---|---|---|
| up to the discharge order | `EMERALD_TINT` / `EMERALD_DEEP` | *intervention possible* |
| discharge order → DISCHTIME | `AMBER_TINT` / `AMBER_DEEP` | *window closing* |
| after DISCHTIME | `SCARLET_TINT` / `SCARLET_DEEP` | *no intervention possible* |

- event ticks pierce the band, 1.4pt, in the event's own tone; event names above the band, 9pt
- proposed detection point: `COBALT_CORE` 6.2pt marker + `COBALT_DEEP` label below
- **the system's trigger point: F2 SOLID `SCARLET_DEEP` diamond** with a white-bold `!` and a `SCARLET_DEEP` 9pt label below. One scarlet mark.
- plate; dominant hue `SCARLET_DEEP`. (No bar: §3.3 is withdrawn.)

**Headline:** *The document the system triggers on is finalised at or after the patient leaves.*
**Draw** 5.854 × 2.13, `0.98`. **Class A** (block 3.57in).
**Targets:** coverage ≥ 0.44, chroma ≥ 0.24.

---

### Fig 12 — `fig-met-exchange` · ch5 · FULL REBUILD
`make_figures.py:911`

**Diagnosis.** The balanced-accuracy line and the MCC band lie on top of each other; two markers stack at the origin against the y-axis rule; two long leaders cross the field; the whole upper-left is empty.

**Rebuild:**

1. **Zoom to the occupied wedge.** xlim `(-1.2, 7.0)`, ylim `(-30, 200)`. The current range to x=12 exists only to give the leaders somewhere to go, and the leaders are gone. The negative x start is what pulls the boxed left spine off the origin cluster.
2. **Indifference lines become shaded trade-off regions.** The wedge between the balanced-accuracy slope and the PABAK slope is filled `MIST` (F3); the narrow band between balanced accuracy and MCC-through-ACCORD is filled `SLATE`-tinted at `#DCE3EE`-over-`MIST` — i.e. draw the wider region first, the narrower over it. The *span* of what different statistics will trade is the content, and it fills the dead upper-left.
3. **Line labels at the margins.** Each of the three slopes is labelled at the **right edge of the plot at the line's own exit y**, horizontal, 9pt, `GRAPHITE`, cased. No rotation, no leaders. Where two exit y's are within 11pt, the lower label drops to the next 11pt slot and its line gets a 3pt `GRAPHITE` connector tick at the frame — a tick at the frame is not a leader.
4. **Origin congestion.** ACCORD is a **F2 SOLID `SCARLET_DEEP` diamond, 7.0pt**, labelled above-left; Llama 3 8B is a 6.2pt `PLUM_CORE` open diamond labelled below-left. Both cased. The x=0 reference becomes a 1.0pt dashed `SLATE` interior rule, now clearly separate from the boxed left spine at x = −1.2.
5. GPT-4o zero-shot `COBALT` square, few-shot `AZURE` triangle, both 6.2pt with adjacent cased labels in their `DEEP`.
6. Boxed axes, no grid.

Dominant hue `SCARLET_DEEP`, named at the `signature()` call. (No bar: §3.3 is withdrawn.)
**Headline:** *ACCORD sits on the indifference line: it buys no trade at all.*
**Draw** 5.854 × 3.39, `0.98`. **Class B** (block 4.83in). The right margin widens to 1.78in so the three slope labels sit INSIDE the canvas: at the old margin they overran it, savefig's tight box grew, and `\includegraphics` then scaled the figure down 8.3% and set its 9pt type at 8.3pt.
**Targets:** coverage ≥ 0.50, chroma ≥ 0.16.

---

### CUT — `fig-survey-audit`
By the author's order. `msu_ch4_results.tex:957` currently reads *"Four of 75 strictly, 14 of 75 permissively. Figure~\ref{fig:survey-audit} puts the three groups on one denominator."*

**Replace with exactly one sentence, and delete the figure environment at lines ~969–985:**

> Four of 75 papers report the check under the strict reading and 14 under the permissive one, so 61 report it under neither.

Nothing else in that subsection changes; the surrounding text already carries the 10-paper gap, the 23/9/32 breakdown, and the drop-versus-impute counts in words.

*(If the author reverses and wants a figure there, the correct object is a 75-square unit/waffle chart, not a stacked bar — 75 squares in a 15 × 5 grid, `SCARLET_CORE` ×4, `AMBER_CORE` ×10, `MIST` ×61, 8pt squares, 1.5pt gutters. That is the one form that makes 4-of-75 legible. It is specified here and not built.)*

### CUT — `fig-disc-strata`
Its two rows are Figure 8's second family, identically. `msu_ch5_discussion.tex:204–206` currently reads *"Figure~\ref{fig:disc-strata} puts the full set and the defect-free stratum side by side. Both intervals straddle chance, and the stratum's is wider because it holds half the positives."*

**Replace with, and delete the figure environment at lines ~208–222:**

> Both intervals straddle chance, and the stratum's is wider because it holds half the positives (Figure~\ref{fig:forest}, second family).

---

## 6. CAPTION POLICY

**Hard cap: 45 words of caption text, which builds to at most 4 single-spaced lines at 12pt across 433.62pt.** Verification counts words mechanically (§11.7).

What a caption may contain: what the marks are, the interval method, and any *inability* the reader must know about (a missing interval, a categorical verdict, an omitted exclusion). What a caption may **not** contain: the finding (that is the headline), cross-figure reconciliation, methodological justification, or an argument about why an alternative encoding was rejected. All of that moves to body text in the section that already discusses it.

**`\apanote` is deleted globally** — the command definition at `main_msu.tex:168` stays for tables, but no figure uses it. It appears on only 3 of 14 figures today, which makes it read as an accident.

### The twelve captions, written

```
Fig 1  The six pipeline stages with the verdict each received and the diagnostic
       that produced it. Two verdicts are categorical because no committed result
       file carries the underlying values. The prevalence multiplier compares the
       weak labeller and the pharmacist on the same 299 pairs.                (43)

Fig 2  The weak labeller's positive rate on the 1,679-row pool and on the 299-pair
       gold set, against the pharmacist's measured base rate on those same 299
       pairs. Wilson intervals from the counts shown. The bracket compares only
       the two gold-set bars.                                                 (43)

Fig 3  The path from the source database to the evaluated sets, with the count at
       each stage and the exclusions recorded as counts. Two intermediate
       exclusions are not drawn because no committed file records either as a
       count.                                                                 (38)

Fig 4  The check that was not run. The rule fires on a conflict between two sets of
       medication mentions; the exemplar shown to annotator, weak labeller and
       model is chosen by position instead. Slots are abstract and carry no
       clinical text.                                                         (41)

Fig 5  Weak-labeller positive rate against true prevalence for three open-weight
       models under two prompt conditions. The diagonal is perfect calibration;
       the tinted wedge is over-prediction. Wilson intervals from the sweep's
       decision counts. The scarlet square is this thesis's clinical operating
       point.                                                                 (42)

Fig 6  Panel (a): each authority scale parameter from its initialisation to its
       final value, with the three seeds. Panel (b): the rule-type head's macro F1
       in the domain it was fitted on and on two it was not. Transfer scores carry
       no committed interval.                                                 (45)

Fig 7  Minimum detectable AUROC at 80% power and two-sided alpha 0.05, by Hanley
       and McNeil [ref], against the number of positives in the evaluation set.
       The circle is this thesis's contradiction evaluation; the square is the
       mortality positive control.                                            (39)

Fig 8  Every headline estimate in this thesis with its 95% interval, grouped into
       three unit families and read against zero. The unit is named once per
       family; position ranks rows within a family only. Values are printed at
       right because one interval is narrower than its marker.                (45)

Fig 9  From-scratch encoders at three parameter scales against a pretrained
       encoder on the public classification task, in accuracy. Caps and the band
       are seed standard deviations; every seed is drawn with its mean.       (34)

Fig 10 The audited demographic-parity gap across operating points and selection
       strengths for one selection experiment. The solid contour is the tolerance
       the audit was written against. The hatched region is where the minimum
       detectable gap exceeds that tolerance.                                 (39)

Fig 11 When the system can fire, relative to when the patient leaves. Nursing
       notes carry a charting time; discharge summaries carry only a charting
       date, so the document the system triggers on is finalised at or after
       discharge.                                                             (39)

Fig 12 What each summary statistic trades on this evaluation set: forgone true
       positives against gained true negatives, with four systems in the same
       coordinates. The indifference slopes follow from the class counts alone
       and are reference structure.                                           (39)
```

### What moves to body text — per figure, explicitly

- **1:** the multiplier-denominator reconciliation with Fig 2 → `msu_ch1_introduction.tex` near line 125.
- **2:** the pool-versus-training-pool row distinction and the forward pointer to Fig 8 for the MCC estimates → `msu_ch1_introduction.tex:237–242`, rewritten (see §8).
- **3:** the full stage-by-stage reading-down narrative (currently ~90 words of caption) → `msu_ch3_methodology.tex` near line 144, where the sections that specify those stages already begin.
- **4:** the 0-to-0.4745 disagreement-rate range and why a population rate does not belong on a single-case picture → `msu_ch4_results.tex` near line 156.
- **5:** the "drawn once rather than repeated into all six panels" justification → deleted entirely; the figure now shows it.
- **6:** the seeds-coincide-at-this-scale note and the underpowered-transfer argument → `msu_ch4_results.tex`, the falsification subsection.
- **7:** the 253-pairs and 6,776-admissions context → already in body at `msu_ch4_results.tex:684`.
- **8:** the whole "colour does not encode zero-crossing" paragraph → deleted; it defends a decision no reader questioned.
- **9:** the panel-(b) paragraph → replaced by one sentence pointing at Fig 8's last row (see §8).
- **10:** the strength-fraction and dashed-boundary description → deleted; the hatching is self-evident.
- **12:** the "ACCORD accented because" justification → deleted; the headline says it.

---

## 7. FLOAT POLICY

**The v3 set contains zero page floats.** Every figure shares a page with body text. Six near-empty sheets cannot be produced by a document with no `[p]` floats in it, so this is a structural fix rather than a tuning exercise.

Geometry: `\textwidth` 433.62pt (6.014in), `\textheight` 620.43pt (8.617in), body double-spaced at ~28.8pt per line.

Every figure is **drawn at its printed size**, so drawn points are page points and 9pt is 9pt. `style.py` gains:

```python
def figw(frac):
    """(drawn_width, printed_width) for a given \\linewidth fraction.
    savefig pad is 0.02in per side and canvas() pins the bbox, so
    saved width = drawn + 0.04 = printed exactly."""
    printed = frac * 6.014
    return printed - 0.04, printed
```

Two width classes only — `0.98` and `0.88`. No figure invents a third.

**They were `0.90` and `0.80`.** The author asked for every figure to be bigger, on ten of the twelve by name, and both classes move up one step: about a 9 percent linear increase, 19 percent more area. `0.98` rather than `1.00` because a graphic set flush to both margins reads as a bleed rather than as a figure. `FIGW` and the schematic width constant move with them, so every figure is still **drawn at its printed size** and 9pt type is still 9pt on the page — that invariant is the whole reason the drawn width is not simply left alone and scaled by `\includegraphics`.

**The invariant has to be checked on the saved file, not assumed.** `savefig` crops to a tight box, so any artist that overruns the canvas widens the PDF, and `\includegraphics[width=...]` then scales the whole figure DOWN to fit the measure, taking the type with it. Four figures were already overrunning at v3's original sizes; `fig-met-exchange` was 8.8 percent over and printing its "9pt" labels at 8.3pt, under the 8.5pt floor this spec sets in §2.2. Every one of the twelve now comes out at exactly `frac × 433.62pt`, verified with `pdfinfo`.

**Block height** = printed graphic height + **1.40in** (≤4 caption lines at 14.4pt, the 0.6-baselineskip caption skip, and float separations).

| Class | Block height | ratio of `\textheight` | Placement | Leaves |
|---|---|---|---|---|
| **A** | ≤ 4.31in | ≤ 0.50 | `[!ht]` | ≥ 10 body lines |
| **B** | 4.31 – 5.34in | 0.50 – 0.62 | `[!t]` | ≥ 8 body lines |
| **forbidden** | 5.34 – 6.89in | 0.62 – 0.80 | — | the gap that makes bottom-gapped sheets |
| **C** | ≥ 6.89in | ≥ 0.80 | `[p]`, paired | — |

**No figure in v3 is Class C.** A figure computing into the forbidden band must be resized to one side of it by changing its drawn aspect ratio — never by changing its type size.

**Roster with computed placement:**

| Fig | drawn W × H | width | printed H | block | ratio | class | placement |
|---|---|---|---|---|---|---|---|
| 1 pipeline | 5.854 × 3.39 | 0.98 | 3.43 | 4.83 | 0.56 | B | `[!t]` |
| 2 headline | 5.854 × 2.78 | 0.98 | 2.82 | 4.22 | 0.49 | A | `[!ht]` |
| 3 cohort | 5.252 × 3.86 | 0.88 | 3.90 | 5.30 | 0.62 | B | `[!t]` |
| 4 exemplar | 5.252 × 2.70 | 0.88 | 2.74 | 4.14 | 0.48 | A | `[!ht]` |
| 5 prevalence | 5.854 × 3.78 | 0.98 | 3.82 | 5.22 | 0.61 | B | `[!t]` |
| 6 falsification | 5.854 × 2.68 | 0.98 | 2.72 | 4.12 | 0.48 | A | `[!ht]` |
| 7 power | 5.854 × 2.85 | 0.98 | 2.89 | 4.29 | 0.50 | A | `[!ht]` |
| 8 forest | 5.854 × 3.71 | 0.98 | 3.75 | 5.15 | 0.60 | B | `[!t]` |
| 9 scaling | 5.854 × 2.84 | 0.98 | 2.88 | 4.28 | 0.50 | A | `[!ht]` |
| 10 threshold | 5.854 × 2.85 | 0.98 | 2.89 | 4.29 | 0.50 | A | `[!ht]` |
| 11 timeline | 5.854 × 2.13 | 0.98 | 2.17 | 3.57 | 0.41 | A | `[!ht]` |
| 12 exchange | 5.854 × 3.39 | 0.98 | 3.43 | 4.83 | 0.56 | B | `[!t]` |

**No figure changed class and nothing lands in the forbidden 5.34–6.89in band.** Three heights are deliberately BELOW a proportional growth, and each is named where a reader will look for it: **3 cohort-flow** (3.86 not 4.18; 4.18 blocks at 5.62in, inside the forbidden band), **7 power** and **10 threshold-surface** (2.85 not 3.00 and 3.06; either would have pushed a Class A float into Class B and changed its placement flag). All three still grow, and all three take the full width increase. The rule that decided them: **the height gives, never the type.**

Two heights are deliberately ABOVE proportional. **5 prevalence-response** goes to 3.78 rather than 3.60 because its redesigned y axis wants near-square panels, and **11 timeline** and the schematics carry their point-grid constants scaled with the plate rather than left at their old size inside a bigger frame.

**Preamble** (`main_msu.tex:132–140`) — replace the float-tuning block with:

```latex
\renewcommand{\topfraction}{0.72}      % was 0.92
\renewcommand{\bottomfraction}{0.45}   % was 0.6
\renewcommand{\textfraction}{0.22}     % was 0.07 -- forces >=8 body lines
\renewcommand{\floatpagefraction}{0.85}
\setcounter{topnumber}{2}
\setcounter{totalnumber}{2}
```

`textfraction` at 0.07 is the direct cause of the bottom-gapped sheets: it permits a page that is 93% float. At 0.22 LaTeX will not build such a page.

**The two named damage sites:**

- **pp.87–90 run.** p.87 carries 19 lines against a 26-line median because Figure 4's 31-line block on p.86 starved it, and pp.88–89 are back-to-back tables. Fix: Figure 4 drops from a 31-line block to Class A (`3.89in`, ~14 lines) with a 41-word caption, and `textfraction` 0.22 stops the tables from top-stacking. The run repacks to full pages.
- **p.54, marooned Fig 3.** The figure plus a 187-word caption fills the sheet. Fix: the narrow class (now `width=0.88\linewidth`, a 5.30in block) plus a 38-word caption releases ~9 body lines onto the same page. It is now a Class B float with text under it, and it still is at the larger size: verified on the current build, where it takes the top of the page with 10 lines of body under the caption.

**Page budget.** The bound document currently runs 145 numbered pages (pdf 160, offset 15) against a 150 ceiling. Twelve caption rewrites drop ~900 words of single-spaced 12pt caption (≈ 2.5 pages), two deleted figure blocks return ≈ 1.5 pages, and the repack returns 1–2 more. Landing zone **139–141**. Framing and fills cost nothing.

**Measured after the enlargement: 143 numbered pages, pdf 158, offset 15**, against the 150 ceiling. Growing all twelve figures by 9 percent cost **zero** pages, which is what the class table above predicts: no figure changed class, so no float changed the page it lands on. The thin-page profile is byte-for-byte the same as before the enlargement on eight of nine sheets and improves on the ninth (p.116 goes from 9 body lines to 14). Measure it with a temporary `\label{ENDOFDOC}` before `\end{document}` and read the page off `main_msu.aux`; do not count PDF pages and subtract.

---

## 8. TeX REWIRING — exact sites

1. `msu_ch1_introduction.tex:168` — caption of Fig 1, replace whole `\caption{...}` body with §6 text; keep the `\caption[...]` short form.
2. `msu_ch1_introduction.tex:237–242` — rewrite. Panel (b) no longer exists. New text: *"Figure~\ref{fig:intro-headline} carries the first half of that failure: the weak labeller's positive rate against the pharmacist's, which is the number the rest of the thesis follows. The second half, Matthews correlation with bootstrap intervals for the four systems on the same 253 pairs, is reported with every other headline estimate in Figure~\ref{fig:forest}; every interval contains zero, so the ordering those numbers appear to give is an ordering over noise."*
3. `msu_ch1_introduction.tex:255` — drop *"The ACCORD row in panel (b) is..."* (no panel b). The contamination flag moves to the body sentence above.
4. `msu_ch1_introduction.tex` figure env — `[!htbp]` → `[!t]` (Fig 1), `[!ht]` (Fig 2).
5. `msu_ch3_methodology.tex:144` — absorb the cohort-flow reading-down narrative from the old caption.
6. `msu_ch3_methodology.tex` Fig 3 env — `width=0.9` → `width=0.88\linewidth` (was `0.80`; §7), `[!htbp]` → `[!t]`.
7. `msu_ch3_methodology.tex:895` — `\apanote` referencing Fig 7 is on a *table*; leave it.
8. `msu_ch4_results.tex:156` — absorb the exemplar disagreement-rate range.
9. `msu_ch4_results.tex` Fig 4 env — `width=0.88\linewidth` (was `0.80`; §7), `[!ht]`; caption replaced.
10. `msu_ch4_results.tex` Fig 5, 6, 7, 8 envs — captions replaced; **all `\apanote` lines deleted**; placements per §7.
11. `msu_ch4_results.tex:957` — replace the survey sentence per §5; **delete the figure env at ~969–985**.
12. `msu_ch5_discussion.tex:204–206` — replace per §5; **delete the disc-strata figure env at ~208–222**.
13. `msu_ch5_discussion.tex:268–273` — rewrite the scaling paragraph: drop *"and its second panel carries the mortality-control comparison on that task's own scale"*; the closing clause becomes *"...and the mortality control returns almost nothing (Figure~\ref{fig:forest}, last row)."*
14. `msu_ch5_discussion.tex` Figs 9, 10, 11, 12 envs — captions replaced; placements per §7.
15. `main_msu.tex:132–140` — float block per §7.
16. `main_msu.tex:158–167` — leave `\captionsetup` as-is; it already produces the required "Figure N: text" below-figure single-spaced block.
17. `accord/stats/verify_thesis_deep.py:42` — `THESIS_FIGS` drops `"fig-survey-audit"` and `"fig-disc-strata"`; list length 14 → 12.
18. `accord/stats/audit_figure_data.py:3` — docstring "thirteen thesis figures" → "twelve".
19. `figures/make_figures.py:1586` and `main()` at :1570 — remove the `make_fig_survey_audit()` and `make_fig_disc_strata()` calls; **leave the functions in place, unreferenced**, exactly as `fig-rule-precision` was handled at the committee review (`verify_thesis_deep.py:38–41` documents that precedent).

---

## 9. MODULES AND BOUNDARIES

```
figures/style.py                  OWNS: the palette ramps, the entity map, rcParams,
                                        figw(), frame(), plate(), signature(),
                                        panel_label(), arrow(), chip(), band(),
                                        case(), wilson_ci(), canvas()
                                  EXPORTS: PALETTE (dict of dicts) as the single
                                        machine-readable source of every hex
                                  FORBIDS: any drawing script defining a hex literal

figures/make_figures.py           OWNS: Figs 1, 2, 3, 4, 12  (schematics + ch1)
figures/fig_batch2.py             OWNS: Figs 5, 7, 8, 9
figures/make_figures_extra.py     OWNS: Fig 6
figures/fig_threshold_surface.py  OWNS: Fig 10
figures/make_figures.py           OWNS: Fig 11
                                  (existing ownership; do not reshuffle)

figures/tools/check_parity.py     NEW. Extracts every #RRGGBB literal and every
                                  rcParam key/value from the notebook's setup and
                                  palette cells; asserts set-equality with
                                  style.PALETTE and style.RCPARAMS. Non-zero exit
                                  on any drift.

figures/tools/measure_ink.py      NEW. Renders each PDF at 300 dpi, reports
                                  coverage and chroma per §5, and exits non-zero
                                  on any figure under its floor.

figures/Thesis_Figure_Material.ipynb
                                  Cells 1-3 (setup, palette, helpers) are REPLACED
                                  from style.py. Two figure cells deleted, twelve
                                  rewritten. check_parity.py gates the pair.
```

The boundary that matters: **`style.py` is the only file in the repository allowed to contain a colour literal.** Today the notebook's palette cell duplicates them by hand, which is how a hue drifts between the script path and the shipping path. `check_parity.py` makes that drift a build failure rather than a proofreading job.

---

## 10. BUILD ORDER

Fifteen steps, each independently verifiable. Do not reorder — steps 1–4 are the foundation and every later step depends on them being right.

1. **`figures/style.py` — rewrite.** Palette ramps (§1.2), `ENTITY_HUE` (§1.4), `PALETTE` export, `RCPARAMS` with STIX (§2.1) and the boxed-axes defaults (§3.1), the type scale as named constants (§2.2), `figw()` (§7). Delete `lean()`, `BLUE`/`TEAL`/`ACCENT`/`MUTED`/`LIGHT`/`DEAD`/`ALERT`, `SERIES_BARE`, `TEXT_TONE`, `CHANCE_STYLE`. Rewrite the module docstring to the v3 rules — the old docstring is a specification and leaving it in place will cause a later maintainer to restore v2 behaviour.
2. **`style.py` — the helper set.** `frame`, `plate`, `signature`, `panel_label`, `arrow`, `chip(fill_deep, text)`, `band(ax, x0, x1, tint)`, `case(artist)`. Every one of these is used by three or more figures; nothing single-use goes here.
3. **`figures/tools/measure_ink.py`.** Write it before any figure, so the first rebuilt figure is measured, not guessed at.
4. **`figures/tools/check_parity.py`.**
5. **Fig 11 timeline** (`make_figures.py:1214`) — the smallest full rebuild. It exercises `plate`, `signature`, F1, F2 and casing in one small canvas. Measure. Do not proceed until it clears 0.44 / 0.24.
6. **Fig 1 pipeline** (`make_figures.py:265`).
7. **Fig 4 exemplar** (`make_figures.py:610`).
8. **Fig 3 cohort flow** (`make_figures.py:1298`). Steps 5–8 are the four monochrome schematics and are the largest single presence gain in the set; finish all four before touching a plot.
9. **Fig 8 forest → master summary** (`fig_batch2.py:~216`). Do this before Figs 2 and 9, because both of those delete panels on the promise that Fig 8 now carries their content.
10. **Fig 2** (`make_figures.py:411`) and **Fig 9** (`fig_batch2.py:~110`) — panel deletions.
11. **Fig 6 falsification** (`make_figures_extra.py:~85`).
12. **Figs 5, 7, 10, 12** — reskin and declutter.
13. **Delete the two figure calls**; update `THESIS_FIGS` and `audit_figure_data.py`.
14. **TeX rewiring**, all nineteen sites in §8.
15. **Notebook regeneration** from `style.py`; run `check_parity.py`.

---

## 11. VERIFICATION — the builder runs all of these; each is pass/fail

1. **Provenance, unchanged and absolute.** `python accord/stats/audit_figure_data.py` — every on-canvas number classified `LOADED` or `DERIVED`. **Zero `TYPED` values.** A `TYPED` value is a build failure regardless of how the figure looks. This gate is the reason the entire redesign is drawing-layer only: no directive in §5 changes a value, an interval, or a source file.
2. **`python accord/stats/verify_thesis_deep.py`** — clean, with `THESIS_FIGS` at 12.
3. **Overlap.** Render every PDF at 600 dpi; for every text artist, get its window extent and assert no pair of extents intersects, and no extent crosses the axes frame. This is the gate STIX must survive — it is narrower than DejaVu so it should pass more easily, but assert it, do not assume it.
4. **`python figures/tools/measure_ink.py`** — every figure at or above its §5 floor; set-mean chroma ≥ 0.18.
5. **Arrowheads.** On a 600 dpi render of Fig 3, measure any trunk arrowhead: **6.5–8.5pt long, 6.0–8.5pt wide.** matplotlib's `mutation_scale` semantics are easy to get wrong by a factor of two; measure once, then trust it.
6. **Palette conformance.** Grep every drawing script for `#[0-9A-Fa-f]{6}` — the only permitted hits are inside `style.py`. Grep for `alpha=` — zero hits. Grep for `lean(` — zero hits. Assert every `SLATE` artist is either a dashed line or a marker under 4pt.
7. **Caption length.** Parse every `\caption{...}` body in the four chapter files; word count ≤ 45. Assert zero `\apanote` inside a `figure` environment.
8. **Canvas bans.** Extract text from every figure PDF with `pdftotext`; assert no U+2013, U+2014, U+2212, no `.pdf`, no `.json`, no `fig-`, no `.iloc`.

   **And the same ban on the DOCUMENT, not only the canvases.** The author's reading of the bound draft found eleven repository script paths still set in `\texttt{}` inside sentences — *"i dont think a thesis will have the notebook coded in it"* — and every one was a `.py`, which `verify_thesis_deep.py`'s pass 9 did not look for: it gated `.md` and `.ipynb` only. Pass 9 now gates `.py`, `.csv` and `.json` as well, on the rendered PDF. The extended pattern flags exactly 11 hits on the pre-change PDF and 0 on the current one. A sentence that named a script still says the procedure was implemented; it names the procedure, or says "the analysis code released with this thesis", and nothing about the claim changes.
9. **Placement.** Build, then run the page-line-count script over `pdftotext -layout main_msu.pdf`: **no body page between the first and last numbered page has fewer than 18 non-blank lines**, and no page contains only a float. Nine sheets are under 18 and every one of them is a chapter opening, a full-page table or a section break rather than a float-damaged page; the check is run as a DIFF against the previous build, because its absolute value has never been zero and the number that matters is whether a change made it worse. This is the direct test for failure 6 and it is the one the critic will run.
10. **Page count.** `Output written on main_msu.pdf` minus the 15-page front-matter offset ≤ **150**; expected 139–141.
11. **Parity.** `python figures/tools/check_parity.py` — exit 0.
12. **Greyscale.** Convert all twelve PDFs to greyscale at 300 dpi and read them. Every series on every canvas must remain separable, by direct label if not by value. This is a human check and it is the last one.

---

## TRADE-OFFS

**Rejected: keep all 14 figures and solve presence with fills alone.** This is the least disruptive option and I rejected it because two of the fourteen are duplicates — disc-strata redraws Figure 8's second family, and Figure 2(b) redraws its first. Making a duplicate beautiful makes the duplication *more* visible, not less; a reader who notices the same two intervals twice, 60 pages apart, in the same new house style, will trust the document less than one who never sees them twice. **If the author refuses the second cut**, I would keep disc-strata at `width=0.60\linewidth` with a scarlet-tinted band spanning the intersection of the two intervals (0.32–0.70) and a caption that explicitly points at Figure 8 — but the duplication survives and so does the objection.

**Rejected: saturated `CORE` fills with white text throughout the schematics.** This is the most literal reading of "vibrant fills, not pastel washes" and it produces the highest chroma numbers. I rejected it because at 9pt serif on a saturated field, white text at 5:1 is legible on a screen and marginal on a laser printer at 600 dpi, and a thesis is printed. The two-level system — `TINT` field with `INK` at 11.3:1, `DEEP` chip with `PAPER` at 7.7:1 — puts the saturation in the chips and the strokes, which is where it survives print. **If the author confirms the thesis will only ever be read as a PDF**, I would move stage boxes from F1 to F2 across all four schematics and take chroma from ~0.35 to ~0.60.

**Rejected: an outer plate on plots as well as boxed axes.** It would make all twelve figures share one enclosure. I rejected it because a plate around a plot that already has a boxed axes is two nested rectangles, and two nested rectangles read as bureaucracy rather than as an exhibit. The signature bar (§3.3) does the unification job instead, at one rectangle per figure. **If a proof shows the schematics and plots still reading as two sets**, the cheap fix is to extend the plate to plots and drop the axes box to 0.7pt — but I would exhaust the signature bar first. *(Superseded: the bar is withdrawn, §3.3. The proof was made and the two groups do not read as two sets; the headline, the type scale and the palette carry it. The escape hatch in this paragraph is still the right one if that ever stops being true.)*

**Rejected: keeping Okabe-Ito unchanged.** It is the defensible citation and it is what the current `style.py` documents. I re-based it because Okabe-Ito was designed for categorical marks on white, not for a system with four fill stops, and because its blue and the author's scarlet print at greyscale 87 and 83 — indistinguishable in a black-and-white bound copy of a figure that draws both. The re-based set is measurably safer (min ΔE 14.7 across all three dichromacies, min greyscale gap 8.9) and it is vibrant enough to fill an area without looking sick. **If the committee objects to a non-cited palette**, the escape is to keep the Okabe-Ito hue *angles* and publish the ramp derivation — the tints and deeps are mechanical transforms of the cores and the citation still holds.

---

## RISKS — what is most likely wrong here

1. **The second cut is not the author's.** `fig:disc-strata` dying is a coordinator-level judgement executed inside an author-level instruction ("cut one figure"). It is the right call on the evidence, but it is the one thing in this spec that should be confirmed before step 13 rather than after. Everything upstream of step 13 is reversible; the TeX deletion is the point of no return.

2. **In-figure headlines may read as non-standard to an MSU committee.** No figure in the current document carries a title; twelve figures carrying a 13pt finding line is a visible departure from the departmental exemplar. It is also the single largest contributor to "gallery-grade" and to fixing "nothing larger than 9pt." Mitigation if challenged: the headline degrades to a 9.5pt `MUTE` deck without touching any other decision, and the set loses presence but not coherence.

3. **The schematic scarlet exception is the weakest joint in the colour system.** "Scarlet repeats on a schematic because it is a state" is defensible and it is what requirement 3 demands, but a hostile critic reading Figure 1's four scarlet chips beside Figure 12's single scarlet diamond will ask why, and the answer lives in a spec they have not read. Mitigation: the pipeline's stage *bands* are `SCARLET_TINT` (a ground, outside the budget) and only the chips are saturated, which keeps the saturated-scarlet area on that canvas small even though the count is four.

4. **The block-height arithmetic in §7 is estimated, not measured.** The 1.40in caption allowance assumes 4 lines at 14.4pt plus separations. If real captions build to 3 lines, several Class B figures become Class A and the placement flags are conservative but wrong-ish; if the caption skip differs from my reading of `main_msu.tex:164`, a Class B figure could drift into the forbidden band. Verification step 9 catches this, but only after a full build — budget one resize pass.

5. **Ink targets on the plots may be unreachable without inventing content.** Figures 2, 8 and 9 hit their coverage floors mainly through ground bands and family tints, which are legitimate structure. If a figure cannot reach its floor without adding a region that is not in the data, **the floor loses**. Do not invent a band to pass `measure_ink.py`. Lower the target and say so.

6. **STIX may render a specific glyph differently from DejaVu** — most plausibly the math italics (`α_D`, `‖Δθ‖`, `F₁`) on Figures 1 and 6. `mathtext.fontset: "stix"` should make this consistent, but the Δ and ‖ glyphs are the ones to look at first on the very first render, before any layout work depends on them.

---

**Files this spec touches:**
`/Users/rickygole/provenance-encoder/accord/thesis/figures/style.py` ·
`/Users/rickygole/provenance-encoder/accord/thesis/figures/make_figures.py` ·
`/Users/rickygole/provenance-encoder/accord/thesis/figures/fig_batch2.py` ·
`/Users/rickygole/provenance-encoder/accord/thesis/figures/make_figures_extra.py` ·
`/Users/rickygole/provenance-encoder/accord/thesis/figures/fig_threshold_surface.py` ·
`/Users/rickygole/provenance-encoder/accord/thesis/figures/Thesis_Figure_Material.ipynb` ·
`/Users/rickygole/provenance-encoder/accord/thesis/figures/tools/measure_ink.py` *(new)* ·
`/Users/rickygole/provenance-encoder/accord/thesis/figures/tools/check_parity.py` *(new)* ·
`/Users/rickygole/provenance-encoder/accord/thesis/main_msu.tex` ·
`/Users/rickygole/provenance-encoder/accord/thesis/msu_ch1_introduction.tex` ·
`/Users/rickygole/provenance-encoder/accord/thesis/msu_ch3_methodology.tex` ·
`/Users/rickygole/provenance-encoder/accord/thesis/msu_ch4_results.tex` ·
`/Users/rickygole/provenance-encoder/accord/thesis/msu_ch5_discussion.tex` ·
`/Users/rickygole/provenance-encoder/accord/stats/verify_thesis_deep.py` ·
`/Users/rickygole/provenance-encoder/accord/stats/audit_figure_data.py`