# Independent main paper and appendix

- `main_text.tex` builds `main_text.pdf`: the main paper followed by references cited in the main paper.
- `appendix.tex` builds `appendix.pdf`: Appendices A-G followed by references cited in the appendix.
- Both documents share `preamble.tex`, `ref.bib`, the content files, images, tables, and the official ICLR style. Each has its own `.aux`, `.bbl`, and bibliography.
- `main.tex` remains the combined manuscript entry point.

## Build

From this directory, run:

```sh
python3 build_papers.py
```

This runs LaTeX and BibTeX through latexmk for both entry points, refreshes cross-document labels, and repeats until references stabilize. The check verifies that each document lists exactly its own cited works and has no undefined references or overfull boxes.

Each entry point can also be compiled directly with `latexmk -pdf -interaction=nonstopmode -halt-on-error main_text.tex` or `latexmk -pdf -interaction=nonstopmode -halt-on-error appendix.tex`. Keep the supplied `crossrefs/` metadata with the source; after changes affecting numbering or pagination, run the joint build to refresh it.

The `crossrefs/*_labels.aux` files contain labels only, never citation or bibliography records. `xr-hyper` imports these labels with the `nocite` option. Citations in each PDF link to its own references; references to sections or figures in the other document use relative PDF links. Keep `main_text.pdf` and `appendix.pdf` in the same directory for cross-document navigation in PDF readers that support it.

Appendix sections remain A-G. Figure/table/equation numbering continues from the main paper; appendix page and line numbering start afresh. Original manuscript content, official style files, and the original arXiv directory are unchanged.
