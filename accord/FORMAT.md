# Morgan State thesis format — extracted requirements

Source: `~/Downloads/thesis examples/` — two colleague theses plus a compiling LaTeX skeleton
(`Thesis Final Draft/`) built against **MSU Thesis Template, October 2024**
(`uploads/MSU Thesis Template-October 2024.docx`), and the SGS Handbook & Style Guide for Theses
(Garrison et al., 2023, 5th ed.).

**The current `accord/thesis/main.tex` violates nearly all of this.** It is `book`/11pt/1.25in
margins/single-spaced/10 chapters with per-chapter figure numbering and natbib. Every line below
differs.

---

## 1. Chapter structure — the biggest gap

The official template mandates **five** chapters:

| | official template | colleague's version |
|---|---|---|
| 1 | Introduction | Introduction |
| 2 | Literature Review | Related Works |
| 3 | Methodology | Method |
| 4 | Results | Results |
| 5 | Summary & Conclusion | Discussion |
| 6 | — | Conclusion |
| | References | References |

**`FRAMING.md` proposes nine chapters. That is outside the template.** The diagnostic frame has to
be expressed *within* Introduction / Literature Review / Methodology / Results / Discussion /
Conclusion — the five research questions become sections of Methodology and Results, not chapters.
This is a re-mapping, not a re-framing: the argument survives, the container changes.

---

## 2. Document class and page setup

```latex
\documentclass[12pt]{report}
\usepackage[left=1.5in,right=1in,top=1in,bottom=1in,includefoot]{geometry}
\usepackage{newtxtext}\usepackage{newtxmath}   % Times New Roman
\usepackage{setspace}                           % \doublespacing throughout the body
\setlength{\parindent}{0.5in}
\setlength{\parskip}{0pt}
```

- **12pt, `report` class** — not `book`, not 11pt.
- **Left margin 1.5in** (binding), right/top/bottom 1in.
- **Times New Roman**; Arial for sans, Courier New for mono.
- **Double-spaced body.** Front-matter pages are single-spaced individually.
- First-line indent 0.5in, zero paragraph skip, `\frenchspacing`.
- Widow/orphan penalties at 10000.

## 3. Numbering

- **Figures, tables and equations number consecutively across the whole thesis** — `Figure 1`,
  `Figure 2`, … not `Figure 3.1`. Achieved with `\counterwithout{figure}{chapter}` etc.
- Section numbers appear in the **TOC** but are suppressed in body headings
  (`\setcounter{secnumdepth}{2}` with `titlesec` formats that print no number).
- TOC entries are prefixed `Chapter N:`.

## 4. Headings

- **Chapter:** centered, bold, *normal size*, prefixed `Chapter N:` on the same line.
- **Section:** bold, normal size, flush left, unnumbered in body.
- **Subsection:** bold italic, normal size, flush left.

## 5. Captions — APA style, and different from the current draft

- Label on its **own line**, **bold** (`Table 1`), caption text on the next line in **italic**.
- Captions sit **above** figures and tables, not below.
- Table notes use an italic `Note.` lead-in (`\apanote{}` in the skeleton).
- An **Illustration** (photograph or picture) is a distinct float class from a **Figure** (charts,
  diagrams). They get separate lists.

## 6. Front matter, in order

1. **Declaration of Authorship** — unnumbered, excluded from page count. **See §9.**
2. **Abstract** — unnumbered, excluded from page count.
3. Title page — Roman numerals begin here
4. **Approval / signature page** — committee chair, co-chair, members, oral defense month and year
5. Dedication *(optional, one page)*
6. Acknowledgements *(optional; no personal, familial or religious testimony)*
7. Table of Contents
8. List of Tables
9. List of Figures
10. List of Illustrations *(if any)*
11. List of Abbreviations — two columns, Abbreviation | Term

Body pages restart at Arabic 1 at Chapter 1.

## 7. Abstract — hard rules, all quoted from the template

- **≤ 250 words**
- **Exactly one paragraph**
- **No abbreviations**
- **No citations**
- **No keywords** (those are entered during ETD upload)
- Not numbered, not counted
- Header block above it: `Title of the Thesis:` / NAME, DEGREE / date / `Thesis Chair:` + department

## 8. Citations and references

- **APA 7** (or MLA 9 / Chicago per discipline — CS at MSU uses APA 7 in both examples).
- Reference entries use a **0.5in hanging indent**.
- Section titled **References**, not Bibliography.
- The current draft uses `natbib` with `plainnat`. That must change.

Also from the template, and it applies throughout: **"every paragraph should feature at least three
sentences."**

## 9. The AI declaration — needs a decision before anything is signed

The **official template** text reads:

> "I acknowledge that I have **not** used artificial intelligence (AI) powered productivity tools to
> assist me in editing this thesis. I further declare that I conducted the research in this thesis
> **on my own and not with any AI-assisted tools**. I also have not used the services of any agency
> providing model, ghostwritten work, or AI tools, other than those specified, in the writing of
> this thesis."

The colleague's submitted version **rewrote this clause** to:

> "I acknowledge that any artificial intelligence tools used in preparing this thesis were limited
> to editorial or formatting assistance and that I remain solely responsible for the research,
> analysis, interpretation, and final presentation of this manuscript."

These are materially different declarations, and the second is not the template's text. Note the
official wording itself contains the escape hatch *"other than those specified"*, which implies
disclosure is contemplated somewhere.

**This is the author's decision and it must be made with the department, not inferred from a
colleague's file.** It cannot be resolved from anything in this repository. Nothing should be signed
until the current SGS Handbook text and the department's position are checked directly.

## 10. Scale calibration

| | source lines | note |
|---|---|---|
| colleague's thesis | **481** lines of chapter LaTeX | double-spaced |
| this draft | **5,810** lines | single-spaced, would roughly double when set to spec |

The current draft is an order of magnitude longer in source than the local norm. `FRAMING.md`
targets ~3,300 lines; even that is far above the example. Length is not itself a virtue here, and
the deletion pass matters more than previously assumed.
