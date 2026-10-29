# Kimodo-Policy — ICLR 2027 working draft

Open and compile `main.tex` from this directory. The compiled output is `main.pdf`.

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## Content and layout

- Migrated from `/data/local-data/data/code/yunhengwang/paper/kimodo-policy/main_arxiv.tex` and its current input files.
- Preliminaries (Kimodo and ControlNet) is Appendix A, and Experimental Setup is Appendix B, after the references. The Method opening points to Appendix A, and the Experimental Results opening points to Appendix B. Both sections retain their full original content. All remaining scientific content is retained in the main body without shortening.
- The official ICLR 2027 `.sty`, `.bst`, `natbib.sty`, and `fancyhdr.sty` files are copied unchanged from the official ZIP. The untouched official example remains under `iclr2027/`.
- HCL page branding is replaced by the official ICLR title, margins, fonts, headings, abstract, references, and review line numbers. Original author names, affiliations, contribution notes and logos are retained in the project; the official review mode hides the author block.
- The old abstract box is converted to ICLR's abstract environment. Figure 1 is fixed on page 1 after the title and before the abstract, at the original full text width. The abstract follows Figure 1 immediately, without a forced page break. The Project Page, GitHub, and Hugging Face badges were removed at the user's request. The HCL-specific ablation-heading override is removed.
- The original numeric citation calls are mapped to parenthetical author-year citations using the official ICLR bibliography style; all citation keys and bibliography entries are preserved.
- All source images and table files are copied. Unreferenced legacy tables (`table/ablation.tex` and `table/simple_single_task_supplementary.tex`) remain available but are not newly inserted into the final-version manuscript.
- This is a content-preserving working draft with Preliminaries in Appendix A and Experimental Setup in Appendix B; it is not a page-limit or submission-readiness pass.

## Source preservation and verification

The source directory is read only throughout the migration. `migration_audit/source_manifest_before.json` records the hashes, sizes, and modification times of every source file. The final audit checks them again. The original LaTeX text is also retained in `migration_audit/source_text_snapshot.zip`.

Official source: https://iclr.cc/Conferences/2027/AuthorGuidelines

Official ZIP: https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip
