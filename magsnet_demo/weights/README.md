# Checkpoint location

The five original checkpoints are already included under:

research_reference/results/magsnet_nested_cv/20260916T210025Z_bc27ff71/seed_42/outer_N/outer_refit/last.pt

Here N runs from 1 through 5. The app locates them relative to the repository,
and checkpoint_manifest.json verifies each SHA-256 before loading.

If you deliberately keep weights elsewhere, set MAGSNET_WEIGHTS_DIR to a directory
containing the same five files renamed outer_1.pt through outer_5.pt. Do not use
inner-fold checkpoints, retrained weights, or arbitrary .pt files with this manifest.
No checkpoint needs to be copied into this documentation directory.
