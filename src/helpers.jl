# ==============================================================================
# Helper functions
# ==============================================================================

# Helper function to get the worker pool for CPUProcesses
function _cpu_processes_pool(accel::CPUProcesses)
    settings = accel.settings
    if settings === nothing
        return nothing
    end
    if isdefined(Distributed, :AbstractWorkerPool) &&
        settings isa Distributed.AbstractWorkerPool
        return settings
    elseif settings isa AbstractVector{<:Integer}
        return Distributed.WorkerPool(collect(settings))
    elseif settings isa Integer
        n = Int(settings)
        n <= 0 && return nothing
        w = Distributed.workers()
        isempty(w) && return nothing
        return Distributed.WorkerPool(w[1:min(n, length(w))])
    else
        return nothing
    end
end

# Helper function to get RNG
get_rng(random_state::Integer) = Xoshiro(random_state)
get_rng(random_state::AbstractRNG) = random_state

# Extract report NamedTuple from cache
function extract_report_from_cache(cache)
    return (
        best_index=cache.best_index,
        best_seed=cache.best_seed,
        best_model=cache.best_model,
        best_history_entry=cache.best_history_entry,
        history=cache.history,
        seeds=cache.seeds,
        inner_report=cache.inner_report,
    )
end

# Normalize rng_field into a Vector{Symbol}
to_path(s::Symbol) = Symbol[s]
to_path(s::AbstractString) = Symbol.(split(s, "."))
to_path(v::AbstractVector) = Symbol.(v)

# Normalise field node (Symbol or QuoteNode) to Symbol
function field_symbol(x)
    if x isa Symbol
        x
    elseif x isa Base.QuoteNode
        x.value::Symbol
    else
        error("Unsupported field node in rng_field expression: $(x)")
    end
end

# Support Expr input, e.g. :(a.b.c) or :(rng)
function to_path(e::Expr)
    # Allow quoted symbols: :(rng)
    if e.head == :quote && length(e.args) == 1
        return Symbol[field_symbol(e.args[1])]
    end

    # Handle dotted field access: :(a.b.c) represented as nested Expr(:., ...)
    if e.head == :.
        fields = Symbol[]
        current = e
        while current isa Expr && current.head == :.
            pushfirst!(fields, field_symbol(current.args[2]))
            current = current.args[1]
        end
        pushfirst!(fields, field_symbol(current))
        return fields
    end

    error("Unsupported rng_field expression: $(e)")
end

to_path(e::QuoteNode) = Symbol[e.value]

#=
TODO: Robust nested-field validation (outline):

1) Traverse the actual object values with `getfield` rather than only field
   types, so abstract/Union-typed fields are handled correctly.
2) At each step, guard against missing fields; if a field is missing on the
   current value, return false immediately.
3) Optionally fall back to type-based checks when `getfield` is not safe
   (e.g., if a field is declared but uninitialised in a mutable struct).

Other files to update:
- test/main.jl: add a test with an abstract/Union-typed intermediate field
  that still resolves at runtime.
- README.md: document `rng_field` validation behaviour for nested fields.
=#

# Does a nested field path exist?
function has_nested(obj, path::Vector{Symbol})
    T = typeof(obj)
    for (i, s) in enumerate(path)
        hasfield(T, s) || return false
        if i < length(path)
            T = fieldtype(T, s)
        end
    end
    return true
end

# Get the object that contains the last field in path (and its type)
function get_parent_and_last(obj, path::Vector{Symbol})
    parent = obj
    for i in 1:(length(path) - 1)
        parent = getfield(parent, path[i])
    end
    return parent, path[end]
end

# Set RNG/seed appropriately depending on field type
function set_rng!(model, rng_path::Vector{Symbol}, seed::Int)
    @assert !isempty(rng_path)
    parent, last = get_parent_and_last(model, rng_path)
    # Field type determines what to set:
    if isdefined(parent, last)
        T = fieldtype(typeof(parent), last)
        if typeintersect(T, Integer) !== Union{} # Covers unions that include Integer
            # Choose a concrete Integer subtype IT that is allowed by T
            intpart = typeintersect(T, Integer)
            IT = if intpart isa Union
                members = filter(
                    t -> t <: Integer && isconcretetype(t) && isbitstype(t),
                    Base.uniontypes(intpart),
                )
                if any(t -> t === Int, members)
                    Int
                elseif !isempty(members)
                    # pick the member with the largest range
                    members[findmax(map(t -> typemax(t), members))[2]]
                else
                    Int
                end
            elseif intpart === Integer || !isconcretetype(intpart)
                Int
            else
                intpart
            end
            # Map seed into the width of IT using its unsigned counterpart
            seedu = unsigned(seed)
            UT = unsigned(IT)  # unsigned type with the same width as IT
            u = UT(seedu)      # wraps/truncates as needed to the correct width
            val = IT <: Unsigned ? convert(IT, u) : reinterpret(IT, u)
            setfield!(parent, last, val)
        else
            # fallback: try to set an RNG
            setfield!(parent, last, Random.Xoshiro(seed))
        end
    else
        error("Field $(last) not defined on $(typeof(parent)).")
    end
    return model
end

# ==============================================================================
# Aggregation helpers
# ==============================================================================

#=
TODO: Deterministic tie-breaking via an ordered map (outline):

1) Replace Dict with an ordered map (e.g. DataStructures.OrderedDict) so
   iteration order matches insertion order ("first seen").
2) When counting, only insert a label on first sight; then update its count.
3) When selecting the winner, iterate the ordered map and only update the
   current best when the new count is strictly greater (ties keep first seen).

Other files to update if you adopt OrderedDict:
- Project.toml: add DataStructures to [deps].
- src/RepeatedRestarts.jl (or src/helpers.jl): add `using DataStructures`.
- test/main.jl: add a tie case asserting deterministic "first seen" behaviour.
- README.md: note deterministic tie-breaking and the extra dependency.
=#

# Majority vote for categorical hard labels (tie-breaker = first seen)
function vote_majority!(out, preds_vecs::Vector{<:AbstractVector})
    n = length(preds_vecs[1])
    @inbounds for i in 1:n
        counts = Dict{Any,Int}()
        for p in preds_vecs
            v = p[i]
            counts[v] = get(counts, v, 0) + 1
        end
        # Pick argmax count
        best = nothing
        bestc = -1
        for (k, c) in counts
            if c > bestc
                best = k
                bestc = c
            end
        end
        out[i] = best
    end
    return out
end

# Aggregate deterministic vector predictions.
function aggregate_prediction_vector(
    preds_vecs::Vector{<:AbstractVector}, aggregation::Symbol
)
    n = length(preds_vecs[1])
    for preds in preds_vecs
        length(preds) == n || error("Cannot aggregate predictions with mismatched lengths.")
    end

    if aggregation == :mean
        return mean(preds_vecs)
    elseif aggregation == :median
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

function _table_aggregation_error(message)
    error("Cannot aggregate deterministic table predictions: $message")
end

function _materialise_aggregated_table(first_pred, names::Tuple, values::Vector)
    columntable = NamedTuple{names}(Tuple(values))
    materializer = Tables.materializer(first_pred)
    try
        materialised = materializer(columntable)
        if Tables.istable(materialised) &&
            Tuple(Tables.columnnames(Tables.columns(materialised))) == names
            return materialised
        end
    catch
        nothing
    end
    return columntable
end

function aggregate_table_predictions(preds_tables::Vector, aggregation::Symbol)
    first_pred = preds_tables[1]
    Tables.istable(first_pred) ||
        _table_aggregation_error("received a non-table prediction as the reference output.")

    first_cols = Tables.columns(first_pred)
    names = Tuple(Tables.columnnames(first_cols))
    isempty(names) &&
        _table_aggregation_error("received a prediction table with no columns.")

    reference_columns = [Tables.getcolumn(first_cols, name) for name in names]
    n = length(reference_columns[1])
    for col in reference_columns
        length(col) == n || _table_aggregation_error(
            "the reference prediction has inconsistent column lengths."
        )
    end

    all_columns = Vector{Any}(undef, length(preds_tables))
    all_columns[1] = reference_columns

    for i in 2:length(preds_tables)
        pred = preds_tables[i]
        Tables.istable(pred) || _table_aggregation_error(
            "repeat $i returned a non-table prediction while other repeats returned tables.",
        )
        cols = Tables.columns(pred)
        Tuple(Tables.columnnames(cols)) == names ||
            _table_aggregation_error("repeat $i returned different column names or order.")
        current_columns = [Tables.getcolumn(cols, name) for name in names]
        for col in current_columns
            length(col) == n || _table_aggregation_error(
                "repeat $i returned a different number of rows from the reference prediction.",
            )
        end
        all_columns[i] = current_columns
    end

    aggregated_values = Vector{Any}(undef, length(names))
    for (j, name) in enumerate(names)
        column_predictions = [cols[j] for cols in all_columns]
        try
            aggregated_values[j] = aggregate_prediction_vector(
                column_predictions, aggregation
            )
        catch err
            if aggregation in (:mean, :median)
                error(
                    "Failed to aggregate table column `$(name)` with aggregation=$(aggregation). " *
                    "Column values do not support this operation.",
                )
            end
            rethrow(err)
        end
    end

    return _materialise_aggregated_table(first_pred, names, aggregated_values)
end

# Averaging of UnivariateFinite vectors
function aggregate_probs(
    yhat_vecs::Vector{<:MLJBase.CategoricalDistributions.UnivariateFiniteVector}
)
    first_vec = yhat_vecs[1]
    n = length(first_vec)
    classes = MMI.classes(first_vec[1])
    L = length(classes)
    # Aggregated probabilities: rows = observations, cols = classes
    P = zeros(Float64, n, L)

    for yh in yhat_vecs
        @inbounds for j in 1:L
            c = classes[j]
            pj = MLJBase.pdf.(yh, Ref(c))
            @inbounds @simd for i in 1:n
                P[i, j] += pj[i]
            end
        end
    end
    P ./= length(yhat_vecs)
    return MMI.UnivariateFinite(classes, P)
end
