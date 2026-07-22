# Shared runtime patch: lower the :oop runner's primitive broadcasts to NATIVE
# stablehlo ops instead of Reactant's one-`<op>_broadcast_scalar`-helper-per-site
# lowering. Include AFTER `using Reactant`. Non-destructive (runtime @eval only).
#
# Why this works: Reactant's `_copyto!` (TracedRArray.jl:394-404) calls
# `broadcast_to_size` on every broadcast arg BEFORE `elem_apply`, so array args
# already share one shape; `elem_apply(::typeof(*), a, b)` can be exactly
# `Ops.multiply(a, b)` — zero helper funcs. Anything not covered falls back to
# the stock generic method (via `invoke`).
#
# Also instruments: `_FAST_HITS` (native lowerings by op), `_NAME_COUNTS`
# (helpers still minted, by name), `_CAP_CALLS`, and a heartbeat printed from
# inside the hooks every ~60 s (timers don't fire during a non-yielding trace).

@eval Reactant.TracedUtils begin
    const _FAST_HITS = Dict{String,Int}()
    const _CAP_CALLS = Ref(0)
    const _NAME_COUNTS = Dict{String,Int}()
    const _HB_T = Ref(Base.time() - 30.0)   # first heartbeat after ~30 s

    function _rss_gb()
        try
            for l in eachline("/proc/self/status")
                startswith(l, "VmRSS") && return round(parse(Int, split(l)[2]) / 1048576; digits=1)
            end
        catch
        end
        return 0.0
    end
    function _hb()
        (Base.time() - _HB_T[] < 60.0) && return nothing
        _HB_T[] = Base.time()
        tops = sort!(collect(_NAME_COUNTS); by=last, rev=true)
        fh   = sort!(collect(_FAST_HITS); by=last, rev=true)
        println("  [hb] fast=", sum(values(_FAST_HITS); init=0),
                " helpers=", _CAP_CALLS[], " rss=", _rss_gb(), "GB",
                " top_helpers=", first(tops, min(length(tops), 4)),
                " fast_hits=", first(fh, min(length(fh), 6)))
        flush(stdout)
        return nothing
    end
    _hit(k::String) = (_FAST_HITS[k] = get(_FAST_HITS, k, 0) + 1; _hb(); nothing)

    # all args same-shape non-scalar traced arrays? (guaranteed for calls coming
    # from _copyto!, which broadcast_to_size's everything first)
    _fast_arrs(args) = !isempty(args) &&
        all(a -> a isa Reactant.TracedRArray && ndims(a) > 0, args) &&
        allequal(map(size, args))

    _uelt(a) = Reactant.unwrapped_eltype(a)
    _cvt(::Type{T}, a) where {T} = _uelt(a) === T ? a :
        Reactant.Ops.convert(Reactant.TracedRArray{T,ndims(a)}, a)
    function _pro(args...)
        T = Base.promote_type(map(_uelt, args)...)
        return map(a -> _cvt(T, a), args)
    end
    _fallback(f, args...) = Base.invoke(elem_apply, Tuple{Any,Vararg{Any}}, f, args...)

    function _nat_fold(opf, tag, args)
        a = _pro(args...)
        r = opf(a[1], a[2])
        for i in 3:length(a); r = opf(r, a[i]); end
        _hit(tag); return r
    end

    # n-ary folds
    function elem_apply(f::typeof(Base.:+), args::Vararg{Any,N}) where {N}
        (N >= 2 && _fast_arrs(args)) ? _nat_fold(Reactant.Ops.add, "add", args) : _fallback(f, args...)
    end
    function elem_apply(f::typeof(Base.:*), args::Vararg{Any,N}) where {N}
        (N >= 2 && _fast_arrs(args)) ? _nat_fold(Reactant.Ops.multiply, "multiply", args) : _fallback(f, args...)
    end
    function elem_apply(f::typeof(Base.min), args::Vararg{Any,N}) where {N}
        (N >= 2 && _fast_arrs(args)) ? _nat_fold(Reactant.Ops.minimum, "minimum", args) : _fallback(f, args...)
    end
    function elem_apply(f::typeof(Base.max), args::Vararg{Any,N}) where {N}
        (N >= 2 && _fast_arrs(args)) ? _nat_fold(Reactant.Ops.maximum, "maximum", args) : _fallback(f, args...)
    end

    # minus: unary negate / binary subtract
    function elem_apply(f::typeof(Base.:-), args::Vararg{Any,N}) where {N}
        if _fast_arrs(args)
            if N == 1
                _hit("negate"); return Reactant.Ops.negate(args[1])
            elseif N == 2
                a = _pro(args...); _hit("subtract"); return Reactant.Ops.subtract(a[1], a[2])
            end
        end
        return _fallback(f, args...)
    end

    # fixed binary
    function elem_apply(f::typeof(Base.:/), args::Vararg{Any,N}) where {N}
        (N == 2 && _fast_arrs(args)) ? _nat_fold(Reactant.Ops.divide, "divide", args) : _fallback(f, args...)
    end
    function elem_apply(f::typeof(Base.:^), args::Vararg{Any,N}) where {N}
        if N == 2 && _fast_arrs(args)
            a = _pro(args...)
            if _uelt(a[1]) <: AbstractFloat
                _hit("power"); return Reactant.Ops.power(a[1], a[2])
            end
        end
        return _fallback(f, args...)
    end
    function elem_apply(f::typeof(Base.muladd), args::Vararg{Any,N}) where {N}
        if N == 3 && _fast_arrs(args)
            a = _pro(args...); _hit("muladd")
            return Reactant.Ops.add(Reactant.Ops.multiply(a[1], a[2]), a[3])
        end
        return _fallback(f, args...)
    end

    # select
    function elem_apply(f::typeof(Base.ifelse), args::Vararg{Any,N}) where {N}
        if N == 3 && _fast_arrs(args) && _uelt(args[1]) === Bool
            a = _pro(args[2], args[3]); _hit("select")
            return Reactant.Ops.select(args[1], a[1], a[2])
        end
        return _fallback(f, args...)
    end

    # Bool logic
    function elem_apply(f::typeof(Base.:&), args::Vararg{Any,N}) where {N}
        (N >= 2 && _fast_arrs(args) && all(a -> _uelt(a) === Bool, args)) ?
            _nat_fold(Reactant.Ops.and, "and", args) : _fallback(f, args...)
    end
    function elem_apply(f::typeof(Base.:|), args::Vararg{Any,N}) where {N}
        (N >= 2 && _fast_arrs(args) && all(a -> _uelt(a) === Bool, args)) ?
            _nat_fold(Reactant.Ops.or, "or", args) : _fallback(f, args...)
    end

    function elem_apply(f::typeof(Base.identity), args::Vararg{Any,N}) where {N}
        (N == 1 && _fast_arrs(args)) ? (_hit("identity"); args[1]) : _fallback(f, args...)
    end

    # raised name cap (safety net) + histogram of helpers still minted + heartbeat
    function __lookup_unique_name_in_module(mod, name)
        _CAP_CALLS[] += 1
        _NAME_COUNTS[name] = get(_NAME_COUNTS, name, 0) + 1
        _hb()
        new_name = name
        tab = MLIR.IR.SymbolTable(MLIR.IR.Operation(mod))
        for i in 0:10_000_000
            new_name = i == 0 ? name : name * "_" * string(i)
            MLIR.IR.mlirIsNull(MLIR.API.mlirSymbolTableLookup(tab, new_name)) && return new_name
        end
        return error("raised cap (10M) exceeded for $name")
    end
end

# comparisons + common unaries: generated OUTSIDE the big @eval block (nesting
# @eval inside @eval breaks `$` interpolation — the outer eval consumes it)
for (jlf, dir) in ((:(==), "EQ"), (:(!=), "NE"), (:(<), "LT"),
                   (:(<=), "LE"), (:(>), "GT"), (:(>=), "GE"))
    Core.eval(Reactant.TracedUtils, quote
        function elem_apply(f::typeof(Base.$jlf), args::Vararg{Any,N}) where {N}
            if N == 2 && _fast_arrs(args)
                a = _pro(args...); _hit("compare" * $dir)
                return Reactant.Ops.compare(a[1], a[2]; comparison_direction=$dir)
            end
            return _fallback(f, args...)
        end
    end)
end
for (jlf, opf) in ((:abs, :abs), (:sqrt, :sqrt), (:exp, :exponential),
                   (:log, :log), (:tanh, :tanh), (:sin, :sine), (:cos, :cosine))
    Core.eval(Reactant.TracedUtils, quote
        function elem_apply(f::typeof(Base.$jlf), args::Vararg{Any,N}) where {N}
            if N == 1 && _fast_arrs(args) && _uelt(args[1]) <: AbstractFloat
                _hit($(string(opf))); return Reactant.Ops.$opf(args[1])
            end
            return _fallback(f, args...)
        end
    end)
end

# EarthSciAST compiled-IR objects are OPAQUE host data to a trace: _Node trees,
# acc kernels, and oop plans hold only host Ints/Float64 tables and never a
# traced value, yet Reactant's `make_tracer` capture walk recurses the whole
# object graph of every closure argument — O(IR size) per @compile. On the
# merged transport closure (1.06M nodes) that walk costs tens of MINUTES; on
# the unmerged 8M-node closure it ran 4+ hours. Registering them as leaf types
# (precedent: Reactant's own `make_tracer(::Union{ExceptionStack,MethodInstance})`)
# makes the walk skip them. Under TracedToTypes (the compile-cache key mode)
# the object itself is pushed so distinct kernel sets keep distinct keys.
if isdefined(Main, :EarthSciAST)
    for T in (Main.EarthSciAST._Node, Main.EarthSciAST._AccKernel,
              Main.EarthSciAST._OopAccPlan, Main.EarthSciAST._AccScratch)
        @eval function Reactant.make_tracer(seen, @nospecialize(prev::$T),
                                            @nospecialize(path), mode; kwargs...)
            mode == Reactant.TracedToTypes && (push!(path, prev); return nothing)
            return prev
        end
    end
end

function report_patch_stats()
    TU = Reactant.TracedUtils
    fh = sort(collect(TU._FAST_HITS); by=last, rev=true)
    println("  native fast-path hits (total=", sum(values(TU._FAST_HITS); init=0), "): ",
        isempty(fh) ? "(none)" : join(["$(p.first)=$(p.second)" for p in fh], ", "))
    println("  helpers minted anyway: total=", TU._CAP_CALLS[])
    for p in first(sort(collect(TU._NAME_COUNTS); by=last, rev=true), min(length(TU._NAME_COUNTS), 12))
        println("    ", p.first, ": ", p.second)
    end
    flush(stdout)
end
