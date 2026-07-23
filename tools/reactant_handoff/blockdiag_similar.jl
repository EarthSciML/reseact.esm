# ===========================================================================
# blockdiag_similar.jl -- teach EarthSciMLBase's `BlockDiagonal` the multi-arg
# `similar` / `zero` that the SciML solver stack needs so a block-diagonal
# `jac_prototype` is NOT silently densified at solver `init`.
# ===========================================================================
# Root cause (found 2026-07-22):
#   DiffEqBase's `promote_f` (solve.jl, run at every `init`) promotes the
#   Jacobian prototype to the state element type with
#       f = @set f.jac_prototype = similar(f.jac_prototype, uElType)
#   `blockdiagonal.jl` only defines the NULLARY `similar(B)`; the element-type
#   form `similar(B, ::Type)` falls through to the generic `AbstractArray`
#   fallback, which returns a DENSE `Matrix`. So a plain `ODEProblem` (and
#   `OrdinaryDiffEqOperatorSplitting.LieTrotterGodunov`, which wraps each
#   operator in a plain `ODEProblem`) replaces our 3528x(13x13) BlockDiagonal
#   with a dense 45864x45864 matrix -> O(N^3) LU, the exact cost the split
#   exists to avoid. (A `SplitODEProblem` avoided it only because its
#   `SplitFunction` takes a different `promote_f` branch that never touches the
#   prototype -- an accident, not a fix.)
#
# The fix: define the element-type `similar` (and its natural companion `zero`)
# so both preserve the block structure. `similar(B)` (nullary), `copy`, `inv`,
# `deepcopy` are already defined in EarthSciMLBase's blockdiagonal.jl; this file
# adds ONLY the forms it is missing. With these, a plain `ODEProblem` keeps the
# BlockDiagonal (verified: `cache.J::BlockDiagonal`, plain-vs-SplitODEProblem
# end-state max|Delta| = 0.0).
#
# This is type piracy on `Base.similar`/`Base.zero` for a type owned by
# EarthSciMLBase; it lives here (a runtime patch, loaded by the runners) rather
# than upstream so the repo does not depend on an EarthSciMLBase fork. The
# durable home is EarthSciMLBase's `blockdiagonal.jl` itself -- add it there and
# this file can be deleted. Requires `BlockDiagonal` to be in scope (the runners
# `include(blockdiag_local.jl); using .BlockDiag` before including this).

Base.similar(B::BlockDiagonal, ::Type{T}) where {T} =
    BlockDiagonal(similar(B.data, T), B.n, B.alg)

Base.zero(B::BlockDiagonal) = BlockDiagonal(zero(B.data), B.n, B.alg)
