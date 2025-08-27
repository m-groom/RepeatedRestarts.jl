# ==============================================================================
# Helper functions
# ==============================================================================

# Helper function to get RNG
get_rng(random_state::Integer) = Xoshiro(random_state)
get_rng(random_state::AbstractRNG) = random_state

# Normalize rng_field into a Vector{Symbol}
to_path(s::Symbol) = Symbol[s]
to_path(s::AbstractString) = Symbol.(split(s, "."))
to_path(v::AbstractVector) = Symbol.(v)

# Normalise field node (Symbol or QuoteNode) to Symbol
field_symbol(x) = x isa Symbol ? x : x isa Base.QuoteNode ? x.value :: Symbol : error("Unsupported field node in rng_field expression: $(x)")

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
function set_rng!(model, rng_path::Vector{Symbol}, seed::UInt64)
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
            # Map seed into the width of IT
            nb = sizeof(IT)
            u = if nb == 16
                UInt128(seed)
            elseif nb == 8
                seed
            elseif nb == 4
                UInt32(seed)
            elseif nb == 2
                UInt16(seed)
            else
                UInt8(seed)
            end
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

# # Majority vote for categorical hard labels (tie-breaker = first seen)
# function vote_majority!(out, preds_vecs::Vector{<:AbstractVector})
#     n = length(preds_vecs[1])
#     @inbounds for i in 1:n
#         counts = Dict{Any,Int}()
#         for p in preds_vecs
#             v = p[i]
#             counts[v] = get(counts, v, 0) + 1
#         end
#         # pick argmax count
#         best = nothing
#         bestc = -1
#         for (k,c) in counts
#             if c > bestc
#                 best = k; bestc = c
#             end
#         end
#         out[i] = best
#     end
#     return out
# end

# # Efficient probability averaging for UnivariateFinite vectors (probabilistic classification)
# function prob_mean_uf(yhat_vecs::Vector)
#     # assume each element is AbstractVector{<:UnivariateFinite}
#     first_vec = yhat_vecs[1]
#     n = length(first_vec)
#     classes = MMI.classes(first_vec[1])
#     L = length(classes)
#     P = zeros(Float64, L, n)  # rows=classes, cols=observations

#     for yh in yhat_vecs
#         @inbounds for j in 1:L
#             c = classes[j]
#             # pdf.(yh, c) returns a vector of length n
#             # accumulate into P[j, :]
#             pj = Distributions.pdf.(yh, Ref(c))
#             @inbounds @simd for i in 1:n
#                 P[j, i] += pj[i]
#             end
#         end
#     end
#     P ./= length(yhat_vecs)
#     # Construct UnivariateFinite array from a probability matrix (see MMI quick-start note)
#     # This uses the "dummy" constructor bound when MLJBase is loaded.
#     return MMI.UnivariateFinite(classes, P)
# end
