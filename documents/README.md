# Paper notes

LaTeX notes on the quant papers in this project. One `.tex` per paper,
standalone (each compiles on its own), sharing a common preamble and
house notation via `\input`. Plus two top-level documents: a verbose
bibliography (`textbook_bib`) and a single-table tracker
(`reading_tracker`).

## Layout

```
papers/
├── README.md                       ← this file
├── reading_tracker.{tex,pdf}       ← one-line-per-paper landscape table
├── textbook_bib.{tex,pdf}          ← verbose source metadata for all papers
├── _common/
│   ├── preamble.tex                ← packages, theorem envs, coloured boxes
│   ├── notation.tex                ← house symbol macros
│   └── template.tex                ← copy this to start a new note
└── <author>_<year>_<slug>/
    └── <author>_<year>_<slug>_note.{tex,pdf}
```

Per-paper notes use `\input{../_common/preamble}` and
`\input{../_common/notation}`. The two top-level documents use
`\input{_common/preamble}` (one level up, not two). Both work as long
as the relative path is intact — **everything must stay under the same
`papers/` root.**

## Recompiling

LaTeX needs two passes to resolve equation cross-references (the first
pass writes the `.aux` file, the second reads it). From any directory
with a `.tex`:

```
pdflatex -interaction=nonstopmode <name>.tex
pdflatex -interaction=nonstopmode <name>.tex
```

Required TeX Live packages (Ubuntu/Debian names):
`texlive-latex-recommended`, `texlive-latex-extra`,
`texlive-fonts-recommended`, `texlive-fonts-extra`, `lmodern`.

## Starting a new paper note

1. Pick a slug: `<firstauthor>_<year>_<shortname>` (e.g. `pucci_2019_tlock`).
2. Make the directory under `papers/` and copy the template:
   ```
   mkdir papers/<slug>
   cp papers/_common/template.tex papers/<slug>/<slug>_note.tex
   ```
3. Edit the first line: `\newcommand{\papertag}{<slug>}`.
4. Fill in the eight sections (see below).
5. Compile twice with `pdflatex`.
6. Add a row to `reading_tracker.tex` and a section to `textbook_bib.tex`.

## Eight-section schema

Each note has, in order:

1. **Metadata** — title, author, year, source, local file, tags, date read.
2. **One-paragraph summary** — in your own words.
3. **Setup and notation** — symbol mapping table (paper → house notation).
4. **Key equations** — 5–15 numbered equations, each with a one-line "Says:" annotation.
5. **Derivations & commentary** — yellow `commentary` boxes for what you worked out yourself.
6. **Validation** — green `validation` box specifying mathematical objects and the values they should take. API-agnostic by design.
7. **Glossary of symbols** — paper's notation, verbatim.
8. **TODO / open questions** — red `todobox`.

Notes are deliberately standalone. No cross-linking between papers —
each note must make sense on its own.

## House notation

`_common/notation.tex` defines macros for recurring symbols (discount
factors, annuities, repo / OIS / collateral rates, recovery, CMT /
swap rates, PV01, etc.). When a paper uses a clashing convention,
document the clash in the §3 mapping table and translate paper symbols
into the house ones — do *not* redefine house macros locally.

Add macros that will recur across papers to `notation.tex`. Macros
specific to one paper stay local to that note.

## Validation block — non-negotiable principles

- It is a **specification of mathematical objects and the values they
  should take**, not a procedure.
- Quantities are named by their mathematics, never by library function
  names.
- Procedural verbs ("invoke", "bump", "round-trip") do not appear.
- The wiring of this spec to actual code lives elsewhere (e.g. a
  sibling `validation.py`), kept separate so the note stays
  library-agnostic.

## Known footguns

- Do **not** redefine `\Delta` in `notation.tex` — it is a LaTeX
  builtin and breaks everything else.
- `\widehat\Macro{arg}` is parsed as `\widehat\Macro` then `{arg}`. If
  `\Macro` takes arguments, write `\widehat{\Macro{arg}}`.
- `\begin{table}` cannot appear inside a `tcolorbox`. Use
  `\begin{center} ... tabular ... \end{center}` plus
  `\refstepcounter{table}\label{...}` for numbering.
- For landscape pages inside a portrait document, use
  `\usepackage{pdflscape}` and `\begin{landscape}...\end{landscape}`.
  The `landscape` geometry option clashes with the preamble.
