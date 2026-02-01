using MLJModelInterface
using MLJClusteringInterface
using Clustering
using Random

const MMI = MLJModelInterface

# Local extension of KMeans with a deterministic seed hyperparameter.
@mlj_model mutable struct KMeansSeeded <: MMI.Unsupervised
    k::Int = 3::(_ ≥ 2)
    metric::Clustering.SemiMetric = Clustering.SqEuclidean()
    init = :kmpp
    seed::Int = 0
end

function MMI.fit(model::KMeansSeeded, verbosity::Int, X)
    Xarray = MMI.matrix(X)'
    rng = Random.Xoshiro(model.seed)
    result = Clustering.kmeans(
        Xarray,
        model.k;
        distance=model.metric,
        init=model.init,
        rng=rng,
    )
    cluster_labels = MMI.categorical(1:model.k)
    fitresult = (result.centers, cluster_labels)
    cache = nothing
    report = (
        assignments=result.assignments,
        cluster_labels=cluster_labels,
    )
    return fitresult, cache, report
end

MMI.fitted_params(::KMeansSeeded, fitresult) = (centers=fitresult[1],)

function MMI.transform(model::KMeansSeeded, fitresult, X)
    X̃ = Clustering.pairwise(model.metric, MMI.matrix(X)', fitresult[1], dims=2)
    return MMI.table(X̃, prototype=X)
end

function MMI.input_scitype(::Type{KMeansSeeded})
    return MMI.Table(MMI.Continuous)
end
