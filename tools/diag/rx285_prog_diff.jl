#!/usr/bin/env julia
# rx285_prog_diff.jl -- diff two rx285_prog.jl dumps.  Scale-relative, not
# bare-relative: opposite-signed ~1e-310 components made a bare metric print a
# spurious `max rel = 2.0` earlier in this campaign.
using Serialization, Printf
a = deserialize(ARGS[1]); b = deserialize(ARGS[2])
@printf("A: Reactant %s label=%s excl=%s\nB: Reactant %s label=%s excl=%s\n",
        a["version"], a["label"], repr(a["excl"]), b["version"], b["label"], repr(b["excl"]))
function scal(x, y)
    isempty(x) && return (0.0, 0.0, 0)
    d = maximum(abs.(x .- y)); s = max(maximum(abs.(x)), 1e-300)
    (d, d / s, count(>(1e-9 * s), abs.(x .- y)))
end
for p in intersect(a["progs"], b["progs"])
    A = a[p]; B = b[p]
    bits = A["state"] == B["state"] && A["pgrad"] == B["pgrad"]
    ss = scal(A["state"], B["state"]); sp = scal(A["pgrad"], B["pgrad"])
    @printf("%-9s A %8.3f ms  B %8.3f ms  A/B %.3f  (mins %7.3f / %7.3f)  compile %6.1f / %6.1f s\n",
            p, 1e3A["med"], 1e3B["med"], A["med"] / B["med"], 1e3A["min"], 1e3B["min"],
            A["compile"], B["compile"])
    @printf("%-9s bit-for-bit %-5s  state dev %.3e (/scale %.3e, %d over)  pgrad dev %.3e (/scale %.3e, %d of %d over)\n",
            "", bits, ss[1], ss[2], ss[3], sp[1], sp[2], sp[3], length(A["pgrad"]))
end
