# ==============================================================================
# RepeatedModel Definition
# ==============================================================================

const MMI = MLJModelInterface

# Define wrapper structs for each MLJ model type
mutable struct DeterministicRepeatedModel{M} <: MMI.Deterministic
    model::M
    rng_field::Union{Symbol,Expr,String}
    n_repeats::Int
    resampling
    measure
    weights::Union{Nothing,AbstractVector{<:Real}}
    class_weights::Union{Nothing,AbstractDict}
    operation
    selection_heuristic::MLJTuning.SelectionHeuristic
    return_mode::Symbol
    aggregation::Symbol
    refit::Bool
    acceleration::AbstractResource
    acceleration_resampling::AbstractResource
    check_measure::Bool
    cache::Bool
    compact_history::Bool
    random_state::Union{AbstractRNG,Integer}
end

mutable struct ProbabilisticRepeatedModel{M} <: MMI.Probabilistic
    model::M
    rng_field::Union{Symbol,Expr,String}
    n_repeats::Int
    resampling
    measure
    weights::Union{Nothing,AbstractVector{<:Real}}
    class_weights::Union{Nothing,AbstractDict}
    operation
    selection_heuristic::MLJTuning.SelectionHeuristic
    return_mode::Symbol
    aggregation::Symbol
    refit::Bool
    acceleration::AbstractResource
    acceleration_resampling::AbstractResource
    check_measure::Bool
    cache::Bool
    compact_history::Bool
    random_state::Union{AbstractRNG,Integer}
end

mutable struct UnsupervisedRepeatedModel{M} <: MMI.Unsupervised
    model::M
    rng_field::Union{Symbol,Expr,String}
    n_repeats::Int
    resampling
    measure
    weights::Union{Nothing,AbstractVector{<:Real}}
    class_weights::Union{Nothing,AbstractDict}
    operation
    selection_heuristic::MLJTuning.SelectionHeuristic
    return_mode::Symbol
    aggregation::Symbol
    refit::Bool
    acceleration::AbstractResource
    acceleration_resampling::AbstractResource
    check_measure::Bool
    cache::Bool
    compact_history::Bool
    random_state::Union{AbstractRNG,Integer}
end

# Union type for all variants
const RepeatedModel{M} = Union{
    DeterministicRepeatedModel{M},
    ProbabilisticRepeatedModel{M},
    UnsupervisedRepeatedModel{M},
} where {M}
Base.parentmodule(::Type{<:RepeatedModel}) = RepeatedRestarts
function Base.fieldnames(::Type{<:RepeatedModel})
    (
        :model,
        :rng_field,
        :n_repeats,
        :resampling,
        :measure,
        :weights,
        :class_weights,
        :operation,
        :selection_heuristic,
        :return_mode,
        :aggregation,
        :refit,
        :acceleration,
        :acceleration_resampling,
        :check_measure,
        :cache,
        :compact_history,
        :random_state,
    )
end

# External keyword constructor
function RepeatedModel(
    args...;
    model=nothing,
    rng_field=:rng,
    n_repeats=10,
    resampling=InSample(),
    measures=nothing,
    measure=measures,
    weights=nothing,
    class_weights=nothing,
    operations=nothing,
    operation=operations,
    selection_heuristic=NaiveSelection(),
    return_mode=:best,
    aggregation=:mean,
    refit=true,
    acceleration=CPU1(), # TODO: currently unused
    acceleration_resampling=CPU1(),
    check_measure=true,
    cache=true,
    compact_history=true,
    random_state=Random.default_rng(),
)
    length(args) < 2 || throw(ArgumentError("Too many positional arguments"))

    if length(args) === 1
        atom = first(args)
        model === nothing ||
            @warn "Using `model=$atom`. Ignoring specification " * "`model=$model`. "
    else
        model === nothing &&
            throw(ArgumentError("model parameter is required for RepeatedModel"))
        atom = model
    end

    # Create appropriate wrapper type based on wrapped model type
    if atom isa Type
        throw(AssertionError("Type encountered where model instance expected"))
    else
        M = typeof(atom)
    end
    _args = (
        atom,
        rng_field,
        n_repeats,
        resampling,
        measure,
        weights,
        class_weights,
        operation,
        selection_heuristic,
        return_mode,
        aggregation,
        refit,
        acceleration,
        acceleration_resampling,
        check_measure,
        cache,
        compact_history,
        random_state,
    )
    if atom isa MMI.Deterministic
        wrapper = DeterministicRepeatedModel{M}(_args...)
    elseif atom isa MMI.Probabilistic
        wrapper = ProbabilisticRepeatedModel{M}(_args...)
    elseif atom isa MMI.Unsupervised
        wrapper = UnsupervisedRepeatedModel{M}(_args...)
    else
        throw(
            ArgumentError(
                "$(typeof(atom)) does not appear to be a supported MLJ model type"
            ),
        )
    end

    message = MMI.clean!(wrapper)
    isempty(message) || @warn message

    return wrapper
end

# Clean method for parameter validation
function MMI.clean!(wrapper::RepeatedModel)
    message = ""
    # Check that rng_field is a valid field of the model
    path = to_path(wrapper.rng_field)
    if !has_nested(wrapper.model, path)
        error(
            "rng_field=$(wrapper.rng_field) does not resolve to a field on $(typeof(wrapper.model)).",
        )
    end
    # Check that n_repeats is positive
    if wrapper.n_repeats <= 0
        message *= "n_repeats must be positive, got $(wrapper.n_repeats). Resetting to 10."
        wrapper.n_repeats = 10
    end
    # Check return_mode
    if !(wrapper.return_mode in [:best, :aggregate, :all])
        message *= "return_mode must be :best, :aggregate, or :all. Got $(wrapper.return_mode). Resetting to :best. "
        wrapper.return_mode = :best
    end
    # Check aggregation
    if !(wrapper.aggregation in [:mean, :median, :mode, :vote])
        message *= "aggregation must be :mean, :median, :mode, or :vote. Got $(wrapper.aggregation). Resetting to :mean. "
        wrapper.aggregation = :mean
    end
    # Check refit
    if wrapper.refit && wrapper.resampling isa InSample
        message *= "refit=true is not required when resampling is InSample. Resetting to false. "
        wrapper.refit = false
    end
    # Check measure
    if wrapper.measure === nothing
        wrapper.measure = default_measure(wrapper.model)
        if wrapper.measure === nothing
            error(
                "Unable to deduce a default measure for specified model. " *
                "You must specify `measure=...`. ",
            )
        else
            message *= "No measure specified. " * "Setting measure=$(wrapper.measure). "
        end
    end
    # Check acceleration
    if (
        wrapper.acceleration isa CPUProcesses &&
        wrapper.acceleration_resampling isa CPUProcesses
    )
        message *=
            "The combination acceleration=$(wrapper.acceleration) and" *
            " acceleration_resampling=$(wrapper.acceleration_resampling) is" *
            "  not generally optimal. You may want to consider setting" *
            " `acceleration = CPUProcesses()` and" *
            " `acceleration_resampling = CPUThreads()`."
    end
    if (
        wrapper.acceleration isa CPUThreads &&
        wrapper.acceleration_resampling isa CPUProcesses
    )
        message *=
            "The combination acceleration=$(wrapper.acceleration) and" *
            " acceleration_resampling=$(wrapper.acceleration_resampling) isn't" *
            " supported. Resetting to" *
            " `acceleration = CPUProcesses()` and" *
            " `acceleration_resampling = CPUThreads()`."

        wrapper.acceleration = CPUProcesses()
        wrapper.acceleration_resampling = CPUThreads()
    end
    wrapper.acceleration = MLJBase._process_accel_settings(wrapper.acceleration)

    return message
end

# ==============================================================================
# MLJ traits
# ==============================================================================

MMI.constructor(::Type{<:RepeatedModel{M}}) where {M} = RepeatedModel{M}
MMI.metadata_model(
    RepeatedModel; human_name="Repeated Model", load_path="RepeatedRestarts.RepeatedModel"
)
function MMI.reports_feature_importances(::Type{<:RepeatedModel{M}}) where {M}
    MMI.reports_feature_importances(M)
end
MMI.is_pure_julia(::Type{<:RepeatedModel{M}}) where {M} = MMI.is_pure_julia(M)
MMI.reporting_operations(::Type{<:RepeatedModel{M}}) where {M} = MMI.reporting_operations(M)
MMI.supports_weights(::Type{<:RepeatedModel{M}}) where {M} = MMI.supports_weights(M)
function MMI.supports_class_weights(::Type{<:RepeatedModel{M}}) where {M}
    MMI.supports_class_weights(M)
end

# Input/output scitypes - inherit from wrapped model
MMI.input_scitype(::Type{<:RepeatedModel{M}}) where {M} = MMI.input_scitype(M)
function MMI.target_scitype(::Type{<:DeterministicRepeatedModel{M}}) where {M}
    MMI.target_scitype(M)
end
function MMI.target_scitype(::Type{<:ProbabilisticRepeatedModel{M}}) where {M}
    MMI.target_scitype(M)
end

# Support for MLJ iteration API
MMI.iteration_parameter(::Type{<:RepeatedModel{M}}) where {M} = :n_repeats
MMI.supports_training_losses(::Type{<:RepeatedModel{M}}) where {M} = true

function MMI.training_losses(wrapper::RepeatedModel, _report)
    _losses = MLJTuning.losses(wrapper.selection_heuristic, _report.history)
    MLJTuning._length(_losses) == 0 && return nothing

    ret = similar(_losses)
    lowest = first(_losses)
    for i in eachindex(_losses)
        current = _losses[i]
        lowest = min(current, lowest)
        ret[i] = lowest
    end
    return ret
end

# ==============================================================================
# Fit Result Structure
# ==============================================================================

struct RepeatedFitResult
    inner_fitresult::Vector
    seeds::Vector{Int}
    best_index::Int
    # Inner constructor
    function RepeatedFitResult(inner_fitresult, seeds, best_index)
        @assert length(inner_fitresult) == length(seeds) "length(inner_fitresult) must equal length(seeds)"
        @assert 1 <= best_index <= length(inner_fitresult) "best_index must be between 1 and length(inner_fitresult)"
        new(inner_fitresult, seeds, best_index)
    end
end

# ==============================================================================
# Fit Methods
# ==============================================================================

include("core.jl")

# Common implementation for fit
function MMI.fit(wrapper::RepeatedModel, verbosity::Int, args...)
    rng = get_rng(wrapper.random_state)
    seeds = rand(rng, 1:typemax(Int32), wrapper.n_repeats)
    path = to_path(wrapper.rng_field)

    # evaluate n_repeats candidates
    history = Vector{NamedTuple}(undef, wrapper.n_repeats)
    models = Vector{Any}(undef, wrapper.n_repeats)
    inner_fitresults = Vector{Any}(undef, wrapper.n_repeats)
    inner_reports = Vector{Any}(undef, wrapper.n_repeats)
    if wrapper.cache
        inner_caches = Vector{Any}(undef, wrapper.n_repeats)
    end

    for (i, s) in enumerate(seeds)
        m = deepcopy(wrapper.model)
        set_rng!(m, path, s)
        mach = machine(m, args...)
        # Prepare a per-seed resampling object as a deep copy so its RNG state
        # (if any) is reset to the initial state for every seed.
        resampling = deepcopy(wrapper.resampling)
        history[i] = evaluate_seed!(mach, wrapper, resampling; verbosity=verbosity)
        models[i] = m
        inner_fitresults[i] = mach.fitresult
        if wrapper.cache
            inner_caches[i] = mach.cache
        end
        inner_reports[i] = MLJBase.report(mach)
    end

    # Choose best using MLJTuning.losses over the history
    losses = MLJTuning.losses(wrapper.selection_heuristic, history)
    best_idx = argmin(losses)

    # Generate fitresult
    if wrapper.return_mode == :best
        if wrapper.refit # Refit models on all data if requested
            best_model = models[best_idx]
            set_rng!(best_model, path, seeds[best_idx])
            mach = machine(best_model, args...)
            fit!(mach; verbosity=verbosity)
            inner_fitresults[best_idx] = mach.fitresult
            if wrapper.cache
                inner_caches[best_idx] = mach.cache
            end
            inner_reports[best_idx] = MLJBase.report(mach)
        end
        fitresult = RepeatedFitResult([inner_fitresults[best_idx]], [seeds[best_idx]], 1)
        c = wrapper.cache ? inner_caches[best_idx] : nothing
        r = inner_reports[best_idx]
    else
        if wrapper.refit # Refit models on all data if requested
            for i in 1:wrapper.n_repeats
                model = models[i]
                set_rng!(model, path, seeds[i])
                mach = machine(model, args...)
                fit!(mach; verbosity=verbosity)
                inner_fitresults[i] = mach.fitresult
                if wrapper.cache
                    inner_caches[i] = mach.cache
                end
                inner_reports[i] = MLJBase.report(mach)
            end
        end
        fitresult = RepeatedFitResult(inner_fitresults, seeds, best_idx)
        c = wrapper.cache ? inner_caches : nothing
        r = inner_reports
    end

    # Generate report
    report = (
        best_index=best_idx,
        best_seed=seeds[best_idx],
        best_model=models[best_idx],
        best_history_entry=history[best_idx],
        history=history,
        seeds=seeds,
        inner_report=r,
    )

    # Generate cache
    if wrapper.cache
        cache = (inner_cache=c,)
    else
        cache = nothing
    end

    return fitresult, cache, report
end

# Common implementation for update - handles cases where n_repeats has been increased
function MMI.update(
    wrapper::RepeatedModel, verbosity::Int, fitresult::RepeatedFitResult, old_cache, args...
)
    # TODO: implement
end

# ==============================================================================
# Transform and Predict Methods
# ==============================================================================

function MMI.transform(
    wrapper::UnsupervisedRepeatedModel, fitresult::RepeatedFitResult, args...
)
    # TODO: implement
end

# Common implementation for predict - supervised models only
function MMI.predict(
    wrapper::Union{DeterministicRepeatedModel,ProbabilisticRepeatedModel},
    fitresult::RepeatedFitResult,
    args...,
)
    # TODO: implement
end

# ==============================================================================
# Fitted Parameters and Feature Importances
# ==============================================================================

function MMI.fitted_params(wrapper::RepeatedModel, fitresult::RepeatedFitResult)
    if wrapper.return_mode == :best
        return MMI.fitted_params(wrapper.model, fitresult.inner_fitresult[1])
    else
        return [
            MMI.fitted_params(wrapper.model, fitresult.inner_fitresult[i]) for
            i in eachindex(fitresult.inner_fitresult)
        ]
    end
end

function MMI.feature_importances(
    wrapper::RepeatedModel, fitresult::RepeatedFitResult, report
)
    if wrapper.return_mode == :best
        return MMI.feature_importances(wrapper.model, fitresult.inner_fitresult[1], report)
    else
        return [
            MMI.feature_importances(wrapper.model, fitresult.inner_fitresult[i], report) for
            i in eachindex(fitresult.inner_fitresult)
        ]
    end
end

# ==============================================================================
# Documentation
# ==============================================================================

"""
$(MMI.doc_header(RepeatedModel))

TODO: add documentation

# Training data

In MLJ or MLJBase, bind an instance `repeated_model` to data with

    mach = machine(repeated_model, X, y)

where

- `X`: any table of input features (e.g., a `DataFrame`) whose columns
  each have a scitype compatible with the wrapped model `repeated_model.model`; check column scitypes with `schema(X)` and model-compatible scitypes with `input_scitype(repeated_model.model)`

- `y`: the target, which can be any `AbstractVector` whose element
  scitype is compatible with the wrapped model `repeated_model.model`; check the scitype
  with `scitype(y)` and model-compatible scitypes with `target_scitype(repeated_model.model)`

Train the machine with `fit!(mach, rows=...)`.


# Hyperparameters

- `model::M`: The wrapped MLJ model

- `n_repeats::Int = 10`: Number of times to repeat the model

- `resampling::Union{Nothing,ResamplingStrategy} = nothing`: Resampling strategy to use for repeated training

- `measure::Union{Nothing,Measure} = nothing`: Measure to use for repeated training

- `weights::Union{Nothing,AbstractVector{<:Real}} = nothing`: Sample weights to use for repeated training

- TODO: complete this

# Operations

- `transform(mach, Xnew)`: transform new data `Xnew` having the same scitype as `X` above.

- `predict(mach, Xnew)`: return predictions of the target from the wrapped model given
  features `Xnew` having the same scitype as `X` above.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `inner_params`: The fitted parameters of the wrapped model


# Report

The fields of `report(mach)` are:

- TODO: complete this


# Examples

```
using MLJ
using RepeatedRestarts

# TODO: add example

# Access fitted parameters
fp = fitted_params(mach)
fp.inner_params                # fitted parameters of the wrapped model
```

"""
RepeatedModel
