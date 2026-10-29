# Kimodo-Policy — ICLR 2027 working draft

Open and compile `main.tex` from this directory. The compiled output is `main.pdf`.

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## Content and layout

- Migrated from `/data/local-data/data/code/yunhengwang/paper/kimodo-policy/main_arxiv.tex` and its current input files.
- Preliminaries (Kimodo and ControlNet) is Appendix A, and Experimental Setup is Appendix B, after the references. The Method opening points to Appendix A, and the Experimental Results opening points to Appendix B. Both sections retain their full original content. The remaining scientific content is retained in the main body; the Figure 1 caption, abstract, and first four Introduction paragraphs, three contribution items, and two Related Work paragraphs have been condensed at the user's request while preserving their argument order and core claims.
- The official ICLR 2027 `.sty`, `.bst`, `natbib.sty`, and `fancyhdr.sty` files are copied unchanged from the official ZIP. The untouched official example remains under `iclr2027/`.
- HCL page branding is replaced by the official ICLR title, margins, fonts, headings, abstract, references, and review line numbers. Original author names, affiliations, contribution notes and logos are retained in the project; the official review mode hides the author block.
- The old abstract box is converted to ICLR's abstract environment. Figure 1 is fixed on page 1 after the title and before the abstract, at the original full text width. The three-line caption is followed immediately by the condensed 18-line abstract, which ends at the bottom of page 1. The Introduction starts on page 2. Figure 2 is a bottom float on page 2; the remaining Introduction text continues on the following page. Image size, fonts, and margins remain unchanged. The Project Page, GitHub, and Hugging Face badges were removed at the user's request. The HCL-specific ablation-heading override is removed.
- The original numeric citation calls are mapped to parenthetical author-year citations using the official ICLR bibliography style; all citation keys and bibliography entries are preserved.
- All source images and table files are copied. Unreferenced legacy tables (`table/ablation.tex` and `table/simple_single_task_supplementary.tex`) remain available but are not newly inserted into the final-version manuscript.
- This is a content-preserving working draft with Preliminaries in Appendix A and Experimental Setup in Appendix B; it is not a page-limit or submission-readiness pass.

## Source preservation and verification

The source directory is read only throughout the migration. `migration_audit/source_manifest_before.json` records the hashes, sizes, and modification times of every source file. The final audit checks them again. The original LaTeX text is also retained in `migration_audit/source_text_snapshot.zip`.

Official source: https://iclr.cc/Conferences/2027/AuthorGuidelines

Official ZIP: https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip

- Task scene visualizations formerly numbered Figures 5, 7, and 8 are preserved in Appendix C (HumanoidArena, SIMPLE, and real-world subsections), with direct appendix and figure references in the corresponding Experimental Results paragraphs. The T2M related-work paragraph is further condensed to finish on page 3.

- Figure 3 is placed after the Method opening and immediately before Section 3.1 on page 4. Its caption, the Method overview, and the Three-Stage Architecture paragraph are condensed. The former architecture equation (2) is removed; conditioning and gradient boundaries are retained in prose, and subsequent equations are renumbered automatically.

- The Training Objective paragraph in Section 3.3 is further condensed while preserving the loss equation, target definitions, masks, main-configuration weights, gradient boundaries, and frozen modules.
- The Figure 3 caption is further condensed, and Motion-History Conditioning in Section 3.2 is condensed to approximately 12 rendered lines without changing its mechanism.

- Short terminal lines in Method are tightened through local wording edits. Single-task and multi-task results in Section 4.1 are condensed to approximately 10 lines each while preserving reported values, comparisons, and evaluation conditions.

- Section 4.2 Single-task results and Table 3 caption are condensed; both Section 4.4 paragraphs and the opening of Section 4.5 are shortened while retaining reported results, task-dependent outcomes, and controlled comparison settings.

- Table 1 now gives the shared statistics, pretraining marker, and best/second-best legend for Tables 1--2; Table 2 retains only its joint-training setting. Figure 5 caption is shortened to three lines, with task labels retained in the figure.

- Section 4.1 single-task, multi-task, and attention-analysis prose is further condensed. A scoped 6pt in-text float spacing keeps Tables 1--2 and Figure 5 together on page 7 without changing image/table sizes, fonts, or margins; Section 4.2 starts on page 8.

- Problem Formulation in Section 3.1 is condensed with its equation and variable definitions intact. Dual-Path Fusion and HandNet each occupy eight lines; the fusion paragraph is kept together. The Section 4.1 result group is anchored to page 7 so Tables 1--2 and Figure 5 remain together and Section 4.2 starts on page 8.

- Section 3.3 Training Objective and Inference are further condensed at the wording level. Loss equations, target definitions, masks, weights, frozen modules, gradient boundaries, denoising steps and receding-horizon execution are retained.

- Section 3.2 Motion Prior Transfer and Motion-History Conditioning are condensed while preserving prior transfer, history constraints and receding-horizon control. Figure 4 caption occupies three lines; Motion-History Conditioning ends at the bottom of page 5, with Section 3.3 starting on page 6. Image dimensions and template settings remain unchanged.
