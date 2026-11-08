# Bundled task text embeddings

These small `.pt` files are ordinary Git files, included when cloning this
repository. No Git LFS or Hugging Face download is needed for these embeddings.

| Directory | Embeddings |
| --- | --- |
| `HumanoidArena/` | 8 |
| `Simple/` | 8 |
| `RealWorld/` | 8 |

Each file stores the source, task identity, instruction and a `(1, 4096)` text
embedding used by the training/evaluation loader. Existing paths remain unchanged.
SIMPLE includes only the eight tasks with released evaluation results. Real-World
includes `teleop-banana`, `teleop-boxing`, `teleop-football`, `teleop-long`,
`teleop-pnp`, `teleop-pushing`, `teleop-sitting`, and `teleop-transportation`.
`teleop-long-v2` and the timestamp-named recording are not included. These caches
do not replace embeddings for new recordings or differently worded instructions.

New instructions require matching embeddings generated with the configured
LLM2Vec encoder. Do not rename a cache to reuse it for a different instruction.
Newly generated cache files remain ignored until explicitly added for release.

For an optional offline integrity check of the released SIMPLE embeddings:

```bash
python utils/prepare_simple.py --text-cache
```

This retained option verifies the files bundled in the checkout; it does not
download anything. SIMPLE metadata normalizes surrounding instruction whitespace;
`encoded_instruction`, when present, records the original encoded text.
