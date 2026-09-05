# Provenance snapshots

Files here pin sibling-repository state that a documented result depended on but
that was not, at the time, recoverable from any pushed commit.

* `earthsciastdiff-uncommitted-2026-09-05.patch` — the `prepare_jacobian(;
  source_layout)` change in EarthSciASTDiff that `tools/adjoint_gradient.jl`
  (reseact.esm `e18479a`, 2026-08-25) has called ever since, and that sat
  uncommitted in the shared checkout through both 48 h CONUS gradients (slurm
  10359755, 10372969). Committed as EarthSciASTDiff `7f83c11` on 2026-09-05; the
  patch is kept so the exact text the runs used is on record.

The runner prints a provenance block at startup (git HEAD, branch and dirty state
of this repo and of every developed sibling); a result whose log shows `DIRTY` is
one whose code is not fully pinned — commit first, then run.
