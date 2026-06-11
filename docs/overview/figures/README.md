---
title: SALSA figure sources
nav_exclude: true
search_exclude: true
---

# Overview figures — TikZ source → committed SVG

**Single source of truth, rendered artifact** — the Waveflow philosophy applied to diagrams. The TikZ
(`*.tex`, the source) is authored here; `render.sh` compiles it to cropped `*.svg` files (the artifacts the
docs embed). Commit **both**.

The SALSA figures share one source, `salsa.tex`, which holds **two** `tikzpicture`s (the `standalone` class
crops each to its own page). `render.sh` exports them to two named SVGs:

| Source | Page | Output (embedded in `../salsa.md`) | Shows |
|---|---|---|---|
| `salsa.tex` | 1 | `salsa_system.svg` | RF tiles → interconnected distributed tiles → bus → common tiles |
| `salsa.tex` | 2 | `salsa_tile.svg`   | one distributed tile: FIR/FFT/systolic PEs + cores behind serial/bus I/F |

## Editing the SALSA figures

1. Open `salsa.tex`. Paste each paper figure into its marked `tikzpicture` body (Figure 1 = system,
   Figure 2 = tile), and copy whatever `\usetikzlibrary{...}` / `\definecolor` / `\newcommand` lines the
   paper preamble used into the `PREAMBLE` block. (Both figures share that one preamble — no duplication.)
2. Render:

   ```bash
   bash render.sh
   ```

3. Commit `salsa.tex` **and** the two generated `.svg` files.

`render.sh` runs `pdflatex` → `dvisvgm --pdf` (per page) and removes the aux files. Toolchain: MiKTeX
(`pdflatex` + `dvisvgm`), verified present on this machine.

## Adding a new (single) figure

Drop a standalone `<name>.tex` here (one `tikzpicture`); `render.sh` renders it to `<name>.svg`
automatically:

```latex
\documentclass[tikz,border=4pt]{standalone}
\usepackage{tikz}
% \usetikzlibrary{...} etc.
\begin{document}
\begin{tikzpicture}
  % ...
\end{tikzpicture}
\end{document}
```

## Embedding

`../salsa.md` embeds `./figures/<name>.svg`. The `standalone` class crops each page to the figure, so the
SVG drops straight into the page at its natural size.
