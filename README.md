# RepeatedRestarts.jl

[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/JuliaDiff/BlueStyle)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![CI](https://github.com/m-groom/RepeatedRestarts.jl/actions/workflows/CI.yaml/badge.svg)](https://github.com/m-groom/RepeatedRestarts.jl/actions/workflows/CI.yaml)
[![Licence](https://img.shields.io/github/license/m-groom/RepeatedRestarts.jl.svg)](https://github.com/m-groom/RepeatedRestarts.jl/blob/main/LICENSE)
[![Julia](https://img.shields.io/badge/julia-1.9%20%7C%201.10%20%7C%201.11-9558B2.svg)](https://julialang.org)

**RepeatedRestarts.jl** provides a wrapper model for [MLJ](https://github.com/JuliaAI/MLJ.jl) that fits any supervised model multiple times with different random seeds, evaluates each repeat using resampling (or in-sample), and returns either the best result, all results, or an aggregation of the results.

## Overview

`RepeatedModel` wraps any supervised MLJ model to train it multiple times with different random seeds, evaluates each repeat (using a resampling strategy or in-sample), and selects the best result using a selection heuristic. This is useful for models whose performance depends on random initialisation (e.g. neural networks or decision trees). Unsupervised models are not supported.

### Hyperparameters

- **Seed control**: The wrapped model must expose a seed/RNG field (possibly nested), configured via `rng_field`; per-repeat seeds are derived from `random_state` for reproducible runs.
- **Evaluation**: Any MLJ `ResamplingStrategy` is supported (e.g., `Holdout`, `CV`) and can be specified via `resampling`. When set to `InSample()`, each repeat is evaluated on the training data directly. Setting `refit` to `true` will refit the best model(s), according to the specified `measure`, on the full training data.
- **Incremental updates**: Increasing `n_repeats` on an already-fitted machine triggers an update that evaluates only the new seeds.
- **Parallelism**: The outer seed-evaluation loop can be parallelised using `acceleration` (e.g., `CPUThreads()` for multi-threaded or `CPUProcesses()` for multi-process parallelism). The resampling evaluation can be parallelised using `acceleration_resampling`.
- **Prediction modes**: `return_mode` controls what `predict` returns: `:best` (best repeat only), `:all` (one prediction per repeat), or `:aggregate` (aggregate across repeats using `aggregation` = `:mean`, `:median` (deterministic only), `:mode`, or `:vote`).


## Installation

```julia
using Pkg
Pkg.add("RepeatedRestarts")
```

Or for development:

```julia
using Pkg
Pkg.develop(path="/path/to/RepeatedRestarts.jl")
```

## Quick Start

```julia
using MLJBase
using MLJDecisionTreeInterface
using StatisticalMeasures
using RepeatedRestarts

# Load some data
X, y = @load_iris

# Base model and wrapper
base_model = RandomForestClassifier(rng=101, n_trees=2, max_depth=3)
model = RepeatedModel(
    model=base_model,
    rng_field=:rng,
    n_repeats=5,
    resampling=Holdout(rng=123),
    random_state=101,
    measure=LogLoss(),
    refit=true,
    return_mode=:best,
)

mach = machine(model, X, y)
fit!(mach)

# Predictions from the best repeat
yhat = predict(mach, X)

# Inspect the report
report(mach).best_index
```

## Licence

This software is distributed under the MIT Licence.
