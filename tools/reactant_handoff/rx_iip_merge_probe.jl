#!/usr/bin/env julia
# PROBE v2: does the :oop kernel-class merge (EarthSciAST._merge_oop_acc_kernels)
# also pay on the PRODUCTION :inplace path? The acc kernels are shared between
# the two emitters, so the same 4,119-kernel fragmentation sits in `_make_rhs`.
#
# v1 finding (and why this version is leaner): the STOCK codegen'd closure was
# OOM-KILLED AT 34.6 GB on its FIRST CALL — Julia native-compiling the one
# generated function that contains all 4,119 kernel bodies blows the 40 GB
# Slurm cgroup. So the stock-with-codegen arm is unmeasurable here (which is
# itself the fragmentation cost, stated in GB). This version compares:
#   fiB : stock kernels, codegen OFF   (the pre-codegen kernel loop)
#   f2n : MERGED kernels, codegen OFF  (isolates the merge)
#   f2c : MERGED kernels, codegen ON   (production config; risky part LAST)
# All under ESS_XCSE_DISABLE=1: xcse (IIP-only, B4) rewrites kernel invariant
# tiers into SCALAR-cache reads the merge does not model, so disabling it makes
# kernels self-contained and the arms comparable. (Consequence for a future
# build.jl hoist: the merge must run BEFORE xcse.)
import Pkg
const HERE  = @__DIR__
const REPO  = normpath(joinpath(HERE, "..", ".."))
Pkg.activate(get(ENV, "RESEACT_RXENV", joinpath(REPO, "run-model-jl")); io=devnull)
using EarthSciAST, EarthSciIO, JSON3
using EarthSciASTSplitter
using EarthSciASTSplitter: split_system, stencil_vs_pointwise
using Logging
const EA = EarthSciAST
const MODEL = get(ENV, "RESEACT_MODEL", "/tmp/reseact_7x7x8.esm")
say(s) = (println(s); flush(stdout))
rss_gb() = round(parse(Int, split(read("/proc/self/status", String),
    r"VmRSS:\s+")[2] |> s -> split(s)[1]) / 1048576; digits=1)
const CHEMDIR = joinpath(REPO, "prototypes", "reseact_3d_chem")
include(joinpath(CHEMDIR, "split_common.jl"))
const T0 = 64800.0

ENV["ESS_XCSE_DISABLE"] = "1"

fiB = u0 = p0 = nothing
tb = time()
withenv("ESS_CODEGEN_DISABLE" => "1") do
    Logging.with_logger(Logging.NullLogger()) do
        global fiB, u0, p0
        file = EA.load(MODEL); flat = EA.flatten(file)
        flat = EA.promote_downstream_shapes(EA.algebraic_states_to_observeds(flat))
        # stencil_following_rule (split_common.jl), not the shipped
    # stencil_vs_pointwise: the air-mass equation reads `divh_fix`, an OBSERVED
    # that is a stencil, and the syntactic rule would post that term to the
    # chemistry half.
    parts = split_system(flat, stencil_following_rule(flat); nparts = 2)
        doc   = EA.flattened_to_esm(parts[1])                      # transport only
        ff = reseact_forcing(CHEMDIR)
        mc = Dict{String,Any}(String(k)=>v for (k,v) in ff.const_arrays)
        mp = Dict{String,Any}()
        for (rawk, prov) in ff.providers
            k = String(rawk); fld = EA._provider_const_field(EA.provider_sample(prov, T0), k)
            (EA.provider_is_const(prov) ? mc : mp)[k] = fld
        end
        ov = Dict{String,Float64}(String(k)=>Float64(v) for (k,v) in ff.parameters)
        fiB, u0, p0, _, _ = EA.build_evaluator(doc;   # form defaults to :inplace
            parameter_overrides=ov, const_arrays=mc, param_arrays=mp)
    end
end
ks = getfield(fiB, :kernel_section).kernels
say("stock :inplace (codegen OFF) built in $(round(time()-tb, digits=1)) s; " *
    "nstates=$(length(u0)) kernels=$(length(ks)) rss=$(rss_gb())GB")

tm = time()
oplans = EA._OopAccPlan[EA._build_oop_acc_plan(K) for K in ks]
merged, _mpl, diag = EA._merge_oop_acc_kernels(ks, oplans)
oplans = nothing; _mpl = nothing; GC.gc()
say("IIP merge: $(length(ks)) -> $(length(merged)) kernels " *
    "in $(round(time()-tm, digits=1)) s; diag=$diag rss=$(rss_gb())GB")

gfB(s) = getfield(fiB, s)
args = (gfB(:rhs_list), gfB(:cse_prelude), gfB(:cse_cache),
        gfB(:const_slots), gfB(:time_slots), gfB(:dyn_slots))

nst = length(u0)
du_s = zeros(nst); du_m = zeros(nst)

bench(f, du) = (fill!(du, 0.0); f(du, u0, p0, T0);   # warm/compile
                (@elapsed for _ in 1:20; f(du, u0, p0, T0); end) / 20 * 1000)

t = bench(fiB, du_s)
say("IIP RHS [stock, codegen OFF]:  $(round(t, digits=2)) ms/call")

f2n = withenv("ESS_CODEGEN_DISABLE" => "1") do
    EA._make_rhs(args[1], args[2], args[3], merged, args[4], args[5], args[6])
end
t = bench(f2n, du_m)
d = maximum(abs, du_m .- du_s); nex = count(du_m .== du_s)
say("IIP RHS [MERGED, codegen OFF]: $(round(t, digits=2)) ms/call")
say("VERIFY merged vs stock: maxabs=$d exact=$nex/$nst " *
    (d == 0.0 ? "BIT-IDENTICAL ✓" : "NOT bit-identical ✗"))
f2n = nothing; GC.gc()

# Production config on merged kernels — the arm stock could not afford. LAST,
# so an OOM here cannot eat the results above.
t_mk = @elapsed f2c = EA._make_rhs(args[1], args[2], args[3], merged,
                                   args[4], args[5], args[6])
say("_make_rhs(merged, codegen ON) generated in $(round(t_mk, digits=1)) s; rss=$(rss_gb())GB")
t_first = @elapsed (fill!(du_m, 0.0); f2c(du_m, u0, p0, T0))
d = maximum(abs, du_m .- du_s)
say("first call (native compile): $(round(t_first, digits=1)) s; rss=$(rss_gb())GB; " *
    "maxabs vs stock=$d " * (d == 0.0 ? "BIT-IDENTICAL ✓" : "✗"))
t = bench(f2c, du_m)
say("IIP RHS [MERGED, codegen ON]:  $(round(t, digits=2)) ms/call")
say("IIP PROBE DONE")
