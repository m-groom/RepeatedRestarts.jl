# Seed evaluation
function evaluate_seed!(
    mach, wrapper::RepeatedModel, resampling=InSample(); verbosity::Int=0
)
    # Build one history entry by evaluating `mach`
    E = evaluate!(
        mach;
        resampling=resampling,
        measure=wrapper.measure,
        operation=wrapper.operation, # if nothing, MLJ determines from measure
        weights=wrapper.weights,
        class_weights=wrapper.class_weights,
        check_measure=wrapper.check_measure,
        acceleration=wrapper.acceleration_resampling,
        compact=wrapper.compact_history,
        verbosity=verbosity,
    )
    entry = (
        model=mach.model,
        measure=E.measure,
        operation=E.operation,
        measurement=E.measurement,
        per_fold=E.per_fold,
    )
    return entry
end

# --- Core fit helpers ---

# Evaluate a batch of seeds, returning all intermediate results.
# Dispatches on the acceleration resource type for parallelisation.
function fit_seeds!(wrapper, seeds, path, args...; verbosity=0)
    return _fit_seeds!(
        wrapper, seeds, path, wrapper.acceleration, args...; verbosity=verbosity
    )
end

# Sequential (CPU1)
function _fit_seeds!(wrapper, seeds, path, ::CPU1, args...; verbosity=0)
    n = length(seeds)
    history = Vector{NamedTuple}(undef, n)
    models = Vector{Any}(undef, n)
    inner_fitresults = Vector{Any}(undef, n)
    inner_reports = Vector{Any}(undef, n)
    inner_caches = wrapper.cache ? Vector{Any}(undef, n) : nothing

    for (i, s) in enumerate(seeds)
        m = deepcopy(wrapper.model)
        set_rng!(m, path, s)
        mach = machine(m, args...)
        # Prepare a per-seed resampling object as a deep copy so its RNG state (if any) is
        # reset to the intiial state for every seed.
        resampling = deepcopy(wrapper.resampling)
        history[i] = evaluate_seed!(mach, wrapper, resampling; verbosity=verbosity)
        models[i] = m
        inner_fitresults[i] = mach.fitresult
        if inner_caches !== nothing
            inner_caches[i] = mach.cache
        end
        inner_reports[i] = MLJBase.report(mach)
    end

    return (; history, models, inner_fitresults, inner_reports, inner_caches)
end

# Multi-threaded (CPUThreads)
function _fit_seeds!(wrapper, seeds, path, ::CPUThreads, args...; verbosity=0)
    if Threads.nthreads() == 1
        return _fit_seeds!(wrapper, seeds, path, CPU1(), args...; verbosity=verbosity)
    end

    n = length(seeds)
    history = Vector{NamedTuple}(undef, n)
    models = Vector{Any}(undef, n)
    inner_fitresults = Vector{Any}(undef, n)
    inner_reports = Vector{Any}(undef, n)
    inner_caches = wrapper.cache ? Vector{Any}(undef, n) : nothing

    # Each iteration writes to a unique index — no shared mutable state.
    Threads.@threads for i in 1:n
        s = seeds[i]
        m = deepcopy(wrapper.model)
        set_rng!(m, path, s)
        mach = machine(m, args...)
        resampling = deepcopy(wrapper.resampling)
        history[i] = evaluate_seed!(mach, wrapper, resampling; verbosity=verbosity)
        models[i] = m
        inner_fitresults[i] = mach.fitresult
        if inner_caches !== nothing
            inner_caches[i] = mach.cache
        end
        inner_reports[i] = MLJBase.report(mach)
    end

    return (; history, models, inner_fitresults, inner_reports, inner_caches)
end

# Multi-process (CPUProcesses)
function _fit_seeds!(wrapper, seeds, path, ::CPUProcesses, args...; verbosity=0)
    if Distributed.nworkers() <= 1
        return _fit_seeds!(wrapper, seeds, path, CPU1(), args...; verbosity=verbosity)
    end

    f = function (s)
        m = deepcopy(wrapper.model)
        set_rng!(m, path, s)
        mach = machine(m, args...)
        resampling = deepcopy(wrapper.resampling)
        h = evaluate_seed!(mach, wrapper, resampling; verbosity=verbosity)
        fr = mach.fitresult
        c = wrapper.cache ? mach.cache : nothing
        r = MLJBase.report(mach)
        (; history_entry=h, model=m, fitresult=fr, report=r, cache=c)
    end

    pool = _cpu_processes_pool(wrapper.acceleration)
    results = if pool === nothing
        Distributed.pmap(f, seeds)
    else
        Distributed.pmap(pool, f, seeds)
    end

    n = length(seeds)
    history = [results[i].history_entry for i in 1:n]
    models = [results[i].model for i in 1:n]
    inner_fitresults = [results[i].fitresult for i in 1:n]
    inner_reports = [results[i].report for i in 1:n]
    inner_caches = wrapper.cache ? [results[i].cache for i in 1:n] : nothing

    return (; history, models, inner_fitresults, inner_reports, inner_caches)
end

# Select best seed, optionally refit on full data, and build the
# (fitresult, cache, report) Tuple.  `refit_indices` controls which
# models to refit (all indices for a fresh fit; only new indices for an
# incremental update).
function _build_result(
    wrapper,
    seeds,
    history,
    models,
    inner_fitresults,
    inner_reports,
    inner_caches,
    path,
    args...;
    verbosity=0,
    refit_indices=eachindex(seeds),
)
    # Choose best using MLJTuning.losses over the history
    losses = MLJTuning.losses(wrapper.selection_heuristic, history)
    best_idx = argmin(losses)

    # --- Refit and build fitresult ---
    if wrapper.return_mode == :best
        if wrapper.refit
            best_model = models[best_idx]
            set_rng!(best_model, path, seeds[best_idx])
            mach = machine(best_model, args...)
            fit!(mach; verbosity=verbosity)
            inner_fitresults[best_idx] = mach.fitresult
            if inner_caches !== nothing
                inner_caches[best_idx] = mach.cache
            end
            inner_reports[best_idx] = MLJBase.report(mach)
        end
        fitresult = RepeatedFitResult([inner_fitresults[best_idx]], [seeds[best_idx]], 1)
        c = inner_caches !== nothing ? inner_caches[best_idx] : nothing
        r = inner_reports[best_idx]
    else
        if wrapper.refit
            for i in refit_indices
                model = models[i]
                set_rng!(model, path, seeds[i])
                mach = machine(model, args...)
                fit!(mach; verbosity=verbosity)
                inner_fitresults[i] = mach.fitresult
                if inner_caches !== nothing
                    inner_caches[i] = mach.cache
                end
                inner_reports[i] = MLJBase.report(mach)
            end
        end
        fitresult = RepeatedFitResult(inner_fitresults, seeds, best_idx)
        c = inner_caches
        r = inner_reports
    end

    # --- Generate report ---
    report = (
        best_index=best_idx,
        best_seed=seeds[best_idx],
        best_model=models[best_idx],
        best_history_entry=history[best_idx],
        history=history,
        seeds=seeds,
        inner_report=r,
    )

    # --- Generate cache (report + extra fields for update) ---
    cache = (
        report...,
        inner_cache=c,
        models=models,
        inner_fitresults=inner_fitresults,
        inner_reports=inner_reports,
        inner_caches=inner_caches,
    )

    return fitresult, cache, report
end
