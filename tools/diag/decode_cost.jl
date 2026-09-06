# ===========================================================================
# decode_cost.jl -- what one forcing refresh costs, per provider, per resolution.
# ===========================================================================
# Every file is already in the cache when this runs, so the number is DECODE
# (netCDF -> Float64 -> record bracket), not download. That is the quantity the
# windowed-read plan is about: the model keeps two records of one variable and
# the reader hands over every variable and all eight records of the file.
#
# MEASURED 2026-09-05/06 (login node, warm cache, t = 64800 s):
#
#   4x5     15 discrete providers, one sample of each = 6.95 s, 31.0 MB kept
#   2x2.5   15 discrete providers                     = 18.04 s, 122.7 MB kept
#
# Per-provider, the cost tracks the COLLECTION, not the variable: A3dyn's four
# consumers (U, V, OMEGA, RH) each pay for all five of its variables, and A1's
# six consumers each pay for all 47 of its.
#
# Over 48 h at macro_dt = 300 the cadences fire ~432 samples (A1 hourly x 6
# providers, A3 3-hourly x 7, I3 3-hourly x 2): ~0.35 s per macro step at 4x5,
# ~0.90 s at 2x2.5, against a 4x5 window budget of 4.86 s.
#
#   julia --project=run-model-jl tools/diag/decode_cost.jl
# ===========================================================================
# How long does ONE forcing refresh actually take, per provider, at each
# resolution? (All files already cached, so this is decode + select, not I/O.)
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const CHEMDIR = joinpath(REPO, "prototypes", "reseact_3d_chem")
include(joinpath(CHEMDIR, "split_common.jl"))
using Printf
const EA = EarthSciAST
for res in ("4x5", "2x2.5")
    ff = reseact_forcing(CHEMDIR; ndays = 1, res = res)
    t0 = 64800.0
    tot = 0.0; bytes = 0.0
    # warm the format machinery once
    EA.provider_sample(ff.providers["GEOSFP.U"], t0)
    for (k, prov) in sort(collect(ff.providers), by = first)
        EA.provider_is_const(prov) && continue
        t = time(); s = EA.provider_sample(prov, t0); dt = time() - t
        fld = EA._provider_const_field(s, String(k))
        mb = fld isa AbstractArray ? length(fld) * 8 / 1e6 : 0.0
        tot += dt; bytes += mb
        @printf("  %-22s %7.3f s   kept %8.2f MB %s\n", k, dt, mb,
                fld isa AbstractArray ? string(size(fld)) : "")
    end
    @printf("%s: %d discrete providers, ONE refresh of all = %.2f s, kept %.1f MB\n\n",
            res, count(p -> !EA.provider_is_const(p), values(ff.providers)), tot, bytes)
end
