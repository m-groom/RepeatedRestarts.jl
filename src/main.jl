# ==============================================================================
# RepeatedModel Definition
# ==============================================================================

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

# Union type for all variants
const RepeatedModel{M} =
    Union{DeterministicRepeatedModel{M},ProbabilisticRepeatedModel{M}} where {M}
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
    acceleration=CPU1(),
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
        throw(
            ArgumentError(
                "Unsupervised models are not supported by RepeatedRestarts. " *
                "Provide a supervised model and a target `y`.",
            ),
        )
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
    if wrapper.aggregation == :median && wrapper isa ProbabilisticRepeatedModel
        message *= "aggregation=:median is not supported for probabilistic models. Resetting to :mean. "
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
    wrapper.acceleration_resampling = MLJBase._process_accel_settings(
        wrapper.acceleration_resampling
    )

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
# Prediction aggregation
# ==============================================================================

# Aggregate predictions for deterministic models
function aggregate_predictions(
    preds_vecs::Vector, aggregation::Symbol, ::DeterministicRepeatedModel
)
    if aggregation == :mean
        return mean(preds_vecs)
    elseif aggregation == :median
        n = length(preds_vecs[1])
        return [
            Statistics.median([preds_vecs[j][i] for j in eachindex(preds_vecs)]) for
            i in 1:n
        ]
    elseif aggregation in (:mode, :vote)
        out = similar(preds_vecs[1])
        return vote_majority!(out, preds_vecs)
    else
        error("Unsupported aggregation method: $aggregation")
    end
end

#=
TODO: Normalise class pools for :vote (outline):

1) Build a union of classes across all prediction vectors so every label
   appears in a single, consistent class pool.
2) Map each voted label to this unioned pool; if a label is missing, raise a
   clear error or expand the pool to include it.
3) Construct the indicator `UnivariateFinite` using the unioned class list to
   avoid `findfirst` returning `nothing`.

Other files to update:
- test/main.jl: add a case where different repeats emit different class pools.
- README.md: document :vote behaviour when class pools differ.
=#

# Aggregate predictions for probabilistic models
function aggregate_predictions(
    preds_vecs::Vector, aggregation::Symbol, ::ProbabilisticRepeatedModel
)
    if aggregation == :mean
        return aggregate_probs(preds_vecs)
    elseif aggregation == :median
        error(
            "Aggregation :median is not supported for " *
            "probabilistic models. Use :mean instead.",
        )
    elseif aggregation in (:mode, :vote)
        # Extract hard labels via mode, then vote
        mode_vecs = [MLJBase.mode.(yh) for yh in preds_vecs]
        out = similar(mode_vecs[1])
        votes = vote_majority!(out, mode_vecs)
        # Wrap as UnivariateFinite (0/1 probabilities)
        classes = MMI.classes(preds_vecs[1][1])
        n = length(votes)
        L = length(classes)
        P = zeros(Float64, n, L)
        for i in 1:n
            j = findfirst(==(votes[i]), classes)
            P[i, j] = 1.0
        end
        return MMI.UnivariateFinite(classes, P)
    else
        error("Unsupported aggregation method: $aggregation")
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

    # Evaluate all seeds
    results = fit_seeds!(wrapper, seeds, path, args...; verbosity=verbosity)

    return _build_result(
        wrapper,
        seeds,
        results.history,
        results.models,
        results.inner_fitresults,
        results.inner_reports,
        results.inner_caches,
        path,
        args...;
        verbosity=verbosity,
    )
end

# Common implementation for update — handles incremental increases of
# n_repeats by evaluating only the new seeds.
function MMI.update(
    wrapper::RepeatedModel, verbosity::Int, fitresult::RepeatedFitResult, old_cache, args...
)
    old_n = length(old_cache.seeds)
    new_n = wrapper.n_repeats

    # If n_repeats decreased or unchanged, exit
    if new_n <= old_n
        @warn "n_repeats decreased or unchanged. Exiting update."
        return fitresult, old_cache, extract_report_from_cache(old_cache)
    end

    # Generate the full seed sequence deterministically
    rng = get_rng(wrapper.random_state)
    all_seeds = rand(rng, 1:typemax(Int32), new_n)
    path = to_path(wrapper.rng_field)

    # Verify old seeds match (same random_state)
    if all_seeds[1:old_n] != old_cache.seeds
        @warn "wrapper.random_state has changed. Exiting update."
        return fitresult, old_cache, extract_report_from_cache(old_cache)
    end

    # --- Evaluate only new seeds ---
    new_seeds = all_seeds[(old_n + 1):new_n]
    new_results = fit_seeds!(wrapper, new_seeds, path, args...; verbosity=verbosity)

    # --- Combine old and new results ---
    all_history = vcat(old_cache.history, new_results.history)
    all_models = vcat(old_cache.models, new_results.models)
    all_inner_fitresults = vcat(old_cache.inner_fitresults, new_results.inner_fitresults)
    all_inner_reports = vcat(old_cache.inner_reports, new_results.inner_reports)
    all_inner_caches =
        if old_cache.inner_caches !== nothing && new_results.inner_caches !== nothing
            vcat(old_cache.inner_caches, new_results.inner_caches)
        else
            nothing
        end

    return _build_result(
        wrapper,
        all_seeds,
        all_history,
        all_models,
        all_inner_fitresults,
        all_inner_reports,
        all_inner_caches,
        path,
        args...;
        verbosity=verbosity,
        refit_indices=(old_n + 1):new_n,
    )
end

# ==============================================================================
# Predict Methods
# ==============================================================================

# Common implementation for predict - supervised models only
function MMI.predict(
    wrapper::Union{DeterministicRepeatedModel,ProbabilisticRepeatedModel},
    fitresult::RepeatedFitResult,
    args...,
)
    # Reformat data for the inner model
    reformatted_args = MMI.reformat(wrapper.model, args...)

    # Check if inner model reports on :predict
    inner_reports_predict = :predict in MMI.reporting_operations(typeof(wrapper.model))

    if wrapper.return_mode == :best
        return MMI.predict(wrapper.model, fitresult.inner_fitresult[1], reformatted_args...)
    end

    # Collect raw returns from all inner fitresults
    raw_returns = [
        MMI.predict(wrapper.model, fitresult.inner_fitresult[i], reformatted_args...) for
        i in eachindex(fitresult.inner_fitresult)
    ]

    # Split predictions and reports if inner model reports on :predict
    if inner_reports_predict
        preds = [first(r) for r in raw_returns]
        predict_reports = [last(r) for r in raw_returns]
    else
        preds = raw_returns
        predict_reports = nothing
    end

    if wrapper.return_mode == :all
        if inner_reports_predict
            wrapper_report = (predict_reports=predict_reports,)
            return preds, wrapper_report
        else
            return preds
        end
    else  # :aggregate
        aggregated = aggregate_predictions(preds, wrapper.aggregation, wrapper)
        if inner_reports_predict
            wrapper_report = (
                predict_reports=predict_reports, aggregation=wrapper.aggregation
            )
            return aggregated, wrapper_report
        else
            return aggregated
        end
    end
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
        return MMI.feature_importances(
            wrapper.model, fitresult.inner_fitresult[1], report.inner_report
        )
    else
        return [
            MMI.feature_importances(
                wrapper.model, fitresult.inner_fitresult[i], report.inner_report[i]
            ) for i in eachindex(fitresult.inner_fitresult)
        ]
    end
end

# ==============================================================================
# Documentation
# ==============================================================================

"""
$(MMI.doc_header(RepeatedModel))

Wraps any supervised MLJ model to train it multiple times with different random seeds,
evaluating each repeat with a resampling strategy (or in-sample) and
selecting the best result according to a selection heuristic.  This is
useful for models whose performance depends on random initialisation, such
as neural networks or decision trees. Unsupervised models are not supported.

# Training data

In MLJ or MLJBase, bind an instance `repeated_model` to data with

    mach = machine(repeated_model, X, y)

where

- `X`: any table of input features (e.g., a `DataFrame`) whose columns
  each have a scitype compatible with the wrapped model; check column
  scitypes with `schema(X)` and model-compatible scitypes with
  `input_scitype(repeated_model.model)`

- `y`: the target, which can be any `AbstractVector` whose element
  scitype is compatible with the wrapped model; check the scitype with
  `scitype(y)` and model-compatible scitypes with
  `target_scitype(repeated_model.model)`

Train the machine with `fit!(mach, rows=...)`.


# Hyperparameters

- `model::M`: The MLJ model to be wrapped.  Must have a field
  (or nested field) that accepts a random seed or RNG, as specified by
  `rng_field`.

- `rng_field::Union{Symbol,Expr,String} = :rng`: Name (or dotted path)
  of the field on `model` that controls random number generation.  Each
  repeat sets this field to a different seed derived from `random_state`.

- `n_repeats::Int = 10`: Number of times to fit the model with different
  random seeds.  Increasing `n_repeats` on an already-fitted machine
  triggers an incremental update (only the new seeds are evaluated).

- `resampling = InSample()`: Resampling strategy used to evaluate each
  repeat.  Any MLJ `ResamplingStrategy` is supported (e.g., `Holdout`,
  `CV`).  When set to `InSample()`, each repeat is evaluated on the
  training data directly and `refit` is automatically set to `false`.

- `measure = nothing`: Performance measure used to rank repeats.  If
  `nothing`, a default measure is inferred from the wrapped model.

- `weights::Union{Nothing,AbstractVector{<:Real}} = nothing`: Per-sample
  weights passed to the resampling evaluation.

- `class_weights::Union{Nothing,AbstractDict} = nothing`: Class weights
  passed to the resampling evaluation.

- `operation = nothing`: Operation used for evaluation (e.g.,
  `predict`, `predict_mode`).  If `nothing`, MLJ determines the
  appropriate operation from the measure.

- `selection_heuristic::MLJTuning.SelectionHeuristic = NaiveSelection()`:
  Heuristic used to select the best repeat from the evaluation history.

- `return_mode::Symbol = :best`: Controls what `predict` returns. One of:
  - `:best` — use only the best repeat's fitresult.
  - `:all`  — return a vector of predictions, one per repeat.
  - `:aggregate` — aggregate predictions across all repeats
    according to `aggregation`.

- `aggregation::Symbol = :mean`: Aggregation method when
  `return_mode = :aggregate`. One of `:mean`, `:median` (deterministic
  only), `:mode`, or `:vote`.

- `refit::Bool = true`: Whether to refit the selected model(s) on the
  full training data after selection.  Automatically set to `false` when
  `resampling = InSample()`.

- `acceleration::AbstractResource = CPU1()`: Computational resource
  for the outer seed-evaluation loop.  Use `CPUThreads()` for
  multi-threaded or `CPUProcesses()` for multi-process parallelism.

- `acceleration_resampling::AbstractResource = CPU1()`: Computational
  resource used for resampling evaluation (e.g., `CPUThreads()`).

- `check_measure::Bool = true`: Whether to verify compatibility between
  the measure, model, and data scitypes.

- `cache::Bool = true`: Whether to cache the inner models' caches.
  Setting to `false` reduces memory usage but disables access to inner
  model caches.

- `compact_history::Bool = true`: Whether to use compact history entries
  (omits per-fold details from the evaluation history).

- `random_state::Union{AbstractRNG,Integer} = Random.default_rng()`:
  Seed or RNG used to generate the per-repeat random seeds.  Set to a
  fixed integer for reproducible results.


# Operations

- `predict(mach, Xnew)`: For supervised wrapped models, return predictions of the target
  given features `Xnew`.  The output depends on `return_mode`:
  - `:best` — a single prediction vector from the best repeat.
  - `:all`  — a vector of prediction vectors, one per repeat.
  - `:aggregate` — a single prediction vector aggregated across repeats.

  If the wrapped model reports on `:predict` (i.e., `:predict in reporting_operations(model)`),
  the wrapper correctly unpacks the inner `(prediction, report)` tuples, aggregates only
  the predictions, and returns a combined report via `report(mach)`.


# Fitted parameters

The fields of `fitted_params(mach)` are:

When `return_mode = :best`:
- The fitted parameters of the best repeat's inner model (same structure
  as `fitted_params` of the wrapped model).

When `return_mode ∈ {:all, :aggregate}`:
- A vector of fitted parameter NamedTuples, one per repeat.


# Report

The fields of `report(mach)` are:

- `best_index`: Index of the best repeat.

- `best_seed`: Random seed used for the best repeat.

- `best_model`: The inner model instance of the best repeat (with its
  `rng_field` set to `best_seed`).

- `best_history_entry`: Evaluation history entry for the best repeat.

- `history`: Vector of evaluation history entries, one per repeat.

- `seeds`: Vector of random seeds, one per repeat.

- `inner_report`: Report from the inner model.  A single NamedTuple when
  `return_mode = :best`, or a vector of NamedTuples otherwise.

When the wrapped model reports on `:predict` (via `reporting_operations`), the following
additional fields appear in `report(mach)` after calling `predict`:

- `seed_used`, `n_predictions`, etc.: Fields from the inner model's predict report
  (for `return_mode = :best`).

- `predict_reports`: A vector of per-repeat predict reports (for `return_mode ∈ {:all, :aggregate}`).

- `aggregation`: The aggregation method used (for `return_mode = :aggregate` only).


# Examples

```
using MLJ
using RepeatedRestarts
using DecisionTree: DecisionTreeClassifier

# Load example dataset
X, y = @load_iris

# Create a repeated classifier
model = RepeatedModel(
    DecisionTreeClassifier(rng=42);
    n_repeats=5,
    resampling=Holdout(rng=123),
    measure=LogLoss(),
    random_state=101,
)
mach = machine(model, X, y)
fit!(mach)

# Make predictions (from the best repeat)
yhat = predict(mach, X)

# Inspect the report
report(mach).best_index        # which repeat was best
report(mach).best_seed         # its random seed
report(mach).history           # full evaluation history

# Access fitted parameters of the best inner model
fp = fitted_params(mach)

# Incremental update: increase n_repeats and refit
model.n_repeats = 8
fit!(mach)                     # evaluates only 3 new seeds
length(report(mach).history)   # 8
```

"""
RepeatedModel
