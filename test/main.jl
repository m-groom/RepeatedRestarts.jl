using Test
using RepeatedRestarts
using MLJBase
using MLJModelInterface
using MLJDecisionTreeInterface
using StatisticalMeasures
using Random

const MMI = MLJModelInterface

@testset "Best model refit reproducibility" begin
    # Load data
    X, y = MLJBase.@load_iris

    # Base model that uses an `rng` field
    base_model = RandomForestClassifier(rng=101, n_trees=2, max_depth=3)

    # Wrap in RepeatedModel; with InSample resampling, refit gets disabled by clean!
    repeated = RepeatedModel(
        model=base_model,
        rng_field=:rng,
        n_repeats=5,
        resampling=Holdout(rng=42),
        random_state=123,
        measure=LogLoss(),
        refit=true,
    )

    mach_rep = machine(repeated, X, y)
    fit!(mach_rep, verbosity=0)

    rep = report(mach_rep)
    loss = rep.best_history_entry.measurement

    yhat_internal = MLJBase.predict(mach_rep, X)

    # Now refit a fresh machine using the same best seed and model, and compare predictions
    best_model = deepcopy(rep.best_model)
    mach_best = machine(best_model, X, y)
    E = evaluate!(mach_best, resampling=Holdout(rng=42), measure=LogLoss())
    loss_best = E.measurement

    @test loss == loss_best

    # Refit on full data to match the RepeatedModel's refit behaviour
    fit!(mach_best, verbosity=0)
    yhat_refit = MLJBase.predict(mach_best, X)

    # Compare probabilities
    classes_ref = MLJBase.classes(yhat_internal[1])
    for c in classes_ref
        @test MLJBase.pdf.(yhat_internal, c) ≈ MLJBase.pdf.(yhat_refit, c)
    end
end

# ==============================================================================
# Constructor Tests
# ==============================================================================

@testset "RepeatedModel Constructor Tests" begin
    X, y = MLJBase.@load_iris

    @testset "deterministic model construction" begin
        base_model = DecisionTreeClassifier(rng=42)
        repeated = RepeatedModel(base_model, n_repeats=3, measure=Accuracy(), refit=false)

        @test repeated isa RepeatedRestarts.ProbabilisticRepeatedModel
        @test repeated.model === base_model
        @test repeated.n_repeats == 3
        @test repeated.rng_field == :rng
        @test repeated.random_state isa Random.AbstractRNG
    end

    @testset "probabilistic model construction" begin
        # Use DecisionTreeClassifier which is probabilistic by default
        base_model = DecisionTreeClassifier(rng=42)
        # Force probabilistic behavior by wrapping appropriately
        repeated = RepeatedModel(
            model=base_model, n_repeats=5, measure=LogLoss(), refit=false
        )

        @test repeated isa Union{
            RepeatedRestarts.DeterministicRepeatedModel,
            RepeatedRestarts.ProbabilisticRepeatedModel,
        }
        @test repeated.model === base_model
        @test repeated.n_repeats == 5
        @test repeated.measure === LogLoss()
    end

    @testset "regressor model construction" begin
        # Create regression data
        X_reg = (x1=rand(100), x2=rand(100))
        y_reg = 2 * X_reg.x1 + 3 * X_reg.x2 + 0.1 * randn(100)

        base_model = DecisionTreeRegressor(rng=42)
        repeated = RepeatedModel(base_model, n_repeats=2, measure=LPLoss(), refit=false)

        @test repeated isa RepeatedRestarts.DeterministicRepeatedModel
        @test repeated.model === base_model
    end

    @testset "constructor with keyword arguments" begin
        base_model = DecisionTreeClassifier(rng=42)
        repeated = RepeatedModel(
            model=base_model,
            rng_field=:rng,
            n_repeats=7,
            resampling=CV(nfolds=3),
            measure=Accuracy(),
            refit=false,
            random_state=123,
        )

        @test repeated.n_repeats == 7
        @test repeated.resampling isa MLJBase.CV
        @test repeated.refit == false
    end

    @testset "constructor error cases" begin
        # Too many positional arguments
        @test_throws ArgumentError RepeatedModel(
            DecisionTreeClassifier(), DecisionTreeRegressor(), n_repeats=3, refit=false
        )

        # No model provided
        @test_throws ArgumentError RepeatedModel()

        # Type instead of instance
        @test_throws AssertionError RepeatedModel(DecisionTreeClassifier)

        # Unsupported model type
        struct UnsupportedModel end
        @test_throws ArgumentError RepeatedModel(UnsupportedModel())

        # Unsupervised models are not supported
        struct DummyUnsupervised <: MMI.Unsupervised end
        @test_throws ArgumentError RepeatedModel(DummyUnsupervised())
    end
end

# ==============================================================================
# Helper Function Tests
# ==============================================================================

@testset "Helper Function Tests" begin
    @testset "get_rng function" begin
        # Test with integer seed
        rng1 = RepeatedRestarts.get_rng(42)
        rng2 = RepeatedRestarts.get_rng(42)
        @test rng1 isa Random.Xoshiro
        @test rng2 isa Random.Xoshiro
        @test typeof(rng1) == typeof(rng2)

        # Test with AbstractRNG
        original_rng = Random.Xoshiro(123)
        returned_rng = RepeatedRestarts.get_rng(original_rng)
        @test returned_rng === original_rng
    end

    @testset "to_path function" begin
        # Test with Symbol
        @test RepeatedRestarts.to_path(:rng) == Symbol[:rng]
        @test RepeatedRestarts.to_path(:field) == Symbol[:field]

        # Test with String
        @test RepeatedRestarts.to_path("rng") == Symbol[:rng]
        @test RepeatedRestarts.to_path("a.b.c") == Symbol[:a, :b, :c]
        @test RepeatedRestarts.to_path("model.rng") == Symbol[:model, :rng]

        # Test with Vector
        @test RepeatedRestarts.to_path(["a", "b"]) == Symbol[:a, :b]
        @test RepeatedRestarts.to_path([:a, :b]) == Symbol[:a, :b]

        # Test with Expr - quoted symbol
        @test RepeatedRestarts.to_path(:(rng)) == Symbol[:rng]
        @test RepeatedRestarts.to_path(:(:field)) == Symbol[:field]

        # Test with Expr - dotted access
        @test RepeatedRestarts.to_path(:(a.b)) == Symbol[:a, :b]
        @test RepeatedRestarts.to_path(:(model.rng)) == Symbol[:model, :rng]
        @test RepeatedRestarts.to_path(:(a.b.c)) == Symbol[:a, :b, :c]

        # Test error cases
        @test_throws ErrorException RepeatedRestarts.to_path(:(a + b))
    end

    @testset "field_symbol function" begin
        @test RepeatedRestarts.field_symbol(:rng) == :rng
        @test RepeatedRestarts.field_symbol(:field) == :field
        @test RepeatedRestarts.field_symbol(QuoteNode(:rng)) == :rng
        @test RepeatedRestarts.field_symbol(QuoteNode(:field)) == :field

        # Test error case
        @test_throws ErrorException RepeatedRestarts.field_symbol("invalid")
    end

    @testset "has_nested function" begin
        # Create test structs
        struct TestInner
            value::Int
        end

        struct TestOuter
            rng::Int
            inner::TestInner
        end

        obj = TestOuter(42, TestInner(100))

        # Test existing single field
        @test RepeatedRestarts.has_nested(obj, Symbol[:rng]) == true
        @test RepeatedRestarts.has_nested(obj, Symbol[:inner]) == true

        # Test existing nested field
        @test RepeatedRestarts.has_nested(obj, Symbol[:inner, :value]) == true

        # Test non-existing field
        @test RepeatedRestarts.has_nested(obj, Symbol[:nonexistent]) == false
        @test RepeatedRestarts.has_nested(obj, Symbol[:inner, :nonexistent]) == false
        @test RepeatedRestarts.has_nested(obj, Symbol[:nonexistent, :value]) == false

        # Test with DecisionTreeClassifier (real model)
        model = DecisionTreeClassifier(rng=42)
        @test RepeatedRestarts.has_nested(model, Symbol[:rng]) == true
        @test RepeatedRestarts.has_nested(model, Symbol[:nonexistent]) == false
    end

    @testset "get_parent_and_last function" begin
        struct TestInner2
            value::Int
            flag::Bool
        end

        struct TestOuter2
            rng::Int
            inner::TestInner2
        end

        obj = TestOuter2(42, TestInner2(100, true))

        # Test single field
        parent, last = RepeatedRestarts.get_parent_and_last(obj, Symbol[:rng])
        @test parent === obj
        @test last == :rng

        # Test nested field
        parent, last = RepeatedRestarts.get_parent_and_last(obj, Symbol[:inner, :value])
        @test parent === obj.inner
        @test last == :value

        parent, last = RepeatedRestarts.get_parent_and_last(obj, Symbol[:inner, :flag])
        @test parent === obj.inner
        @test last == :flag
    end

    @testset "set_rng! function" begin
        # Test with Integer field
        model = DecisionTreeClassifier(rng=42)
        original_rng = model.rng

        RepeatedRestarts.set_rng!(model, Symbol[:rng], 12345)
        @test model.rng != original_rng
        @test model.rng isa Integer

        # Test with RNG field (create a mutable struct for testing)
        mutable struct TestModelWithRNG
            rng::Random.AbstractRNG
        end

        test_model = TestModelWithRNG(Random.Xoshiro(1))
        original_rng = test_model.rng

        RepeatedRestarts.set_rng!(test_model, Symbol[:rng], 54321)
        @test test_model.rng != original_rng
        @test test_model.rng isa Random.Xoshiro

        # Test error case - undefined field
        @test_throws ErrorException RepeatedRestarts.set_rng!(
            model, Symbol[:nonexistent], 1
        )
    end
end

# ==============================================================================
# Validation Tests (clean! method)
# ==============================================================================

@testset "Parameter Validation Tests" begin
    @testset "rng_field validation" begin
        base_model = DecisionTreeClassifier(rng=42)

        # Valid rng_field should work (may generate other warnings about default values)
        repeated1 = RepeatedModel(
            base_model, rng_field=:rng, measure=Accuracy(), refit=false
        )
        @test repeated1.rng_field == :rng

        repeated2 = RepeatedModel(
            base_model, rng_field="rng", measure=Accuracy(), refit=false
        )
        @test repeated2.rng_field == "rng"

        # Invalid rng_field should throw during construction
        @test_throws ErrorException RepeatedModel(
            base_model, rng_field=:nonexistent, refit=false
        )
        @test_throws ErrorException RepeatedModel(
            base_model, rng_field="nonexistent", refit=false
        )
    end

    @testset "n_repeats validation" begin
        base_model = DecisionTreeClassifier(rng=42)

        # Valid n_repeats
        repeated = RepeatedModel(base_model, n_repeats=5, measure=Accuracy(), refit=false)
        @test repeated.n_repeats == 5

        # Invalid n_repeats should be corrected with warning
        repeated = @test_logs (:warn, r"n_repeats.*Resetting to 10") RepeatedModel(
            base_model, n_repeats=0, measure=Accuracy(), refit=false
        )
        @test repeated.n_repeats == 10

        repeated = @test_logs (:warn, r"n_repeats.*Resetting to 10") RepeatedModel(
            base_model, n_repeats=-5, measure=Accuracy(), refit=false
        )
        @test repeated.n_repeats == 10
    end

    @testset "return_mode validation" begin
        base_model = DecisionTreeClassifier(rng=42)

        # Valid return_mode values
        for mode in [:best, :aggregate, :all]
            repeated = RepeatedModel(
                base_model, return_mode=mode, measure=Accuracy(), refit=false
            )
            @test repeated.return_mode == mode
        end

        # Invalid return_mode should be corrected with warning
        repeated = @test_logs (:warn, r"return_mode.*Resetting to :best") RepeatedModel(
            base_model, return_mode=:invalid, measure=Accuracy(), refit=false
        )
        @test repeated.return_mode == :best
    end

    @testset "aggregation validation" begin
        base_model = DecisionTreeClassifier(rng=42)

        # Valid aggregation values (excluding :median which is reset for probabilistic)
        for agg in [:mean, :mode, :vote]
            repeated = RepeatedModel(
                base_model, aggregation=agg, measure=Accuracy(), refit=false
            )
            @test repeated.aggregation == agg
        end

        # :median is valid for deterministic models
        reg_model = DecisionTreeRegressor(rng=42)
        repeated_det = RepeatedModel(
            reg_model, aggregation=:median, measure=LPLoss(), refit=false
        )
        @test repeated_det.aggregation == :median

        # Invalid aggregation should be corrected with warning
        repeated = @test_logs (:warn, r"aggregation.*Resetting to :mean") RepeatedModel(
            base_model, aggregation=:invalid, measure=Accuracy(), refit=false
        )
        @test repeated.aggregation == :mean
    end

    @testset "refit validation with InSample" begin
        base_model = DecisionTreeClassifier(rng=42)

        # refit=true with InSample should be corrected with warning
        repeated = @test_logs (:warn, r"refit=true is not required.*Resetting to false") RepeatedModel(
            base_model, refit=true, resampling=InSample(), measure=Accuracy()
        )
        @test repeated.refit == false

        # refit=false with InSample should be fine
        repeated = RepeatedModel(
            base_model, refit=false, resampling=InSample(), measure=Accuracy()
        )
        @test repeated.refit == false

        # refit=true with other resampling should be fine
        repeated = RepeatedModel(base_model, refit=true, resampling=Holdout())
        @test repeated.refit == true
    end

    @testset "measure validation" begin
        base_model = DecisionTreeClassifier(rng=42)

        # measure=nothing should be auto-detected with warning
        repeated = @test_logs (:warn, r"No measure specified.*Setting measure=") RepeatedModel(
            base_model, measure=nothing, refit=false
        )
        @test repeated.measure !== nothing
    end

    @testset "acceleration validation" begin
        base_model = DecisionTreeClassifier(rng=42)

        # CPUProcesses + CPUProcesses should generate warning
        repeated = @test_logs (:warn, r"not generally optimal") RepeatedModel(
            base_model,
            acceleration=CPUProcesses(),
            acceleration_resampling=CPUProcesses(),
            measure=Accuracy(),
            refit=false,
        )

        # CPUThreads + CPUProcesses should be corrected with warning
        # We just test the behavior, not the specific warning message
        repeated = RepeatedModel(
            base_model,
            acceleration=CPUThreads(),
            acceleration_resampling=CPUProcesses(),
            measure=Accuracy(),
            refit=false,
        )
        @test repeated.acceleration isa CPUProcesses
        @test repeated.acceleration_resampling isa CPUThreads
    end
end

# ==============================================================================
# Core Functionality Tests
# ==============================================================================

@testset "Core Functionality Tests" begin
    @testset "evaluate_seed! function" begin
        X, y = MLJBase.@load_iris
        base_model = DecisionTreeClassifier(rng=42)

        # Create a RepeatedModel
        repeated = RepeatedModel(
            base_model, n_repeats=3, resampling=CV(nfolds=3, rng=42), measure=Accuracy()
        )

        # Create a machine with the base model
        mach = machine(base_model, X, y)

        # Test evaluate_seed! returns correct structure
        result = RepeatedRestarts.evaluate_seed!(
            mach, repeated, repeated.resampling; verbosity=0
        )

        @test result isa NamedTuple
        @test haskey(result, :model)
        @test haskey(result, :measure)
        @test haskey(result, :operation)
        @test haskey(result, :measurement)
        @test haskey(result, :per_fold)

        @test result.model === mach.model
        @test eltype(result.measurement) <: Real
        @test result.per_fold isa AbstractVector

        # Test that measurement is reasonable (accuracy should be between 0 and 1)
        @test all(0 .<= result.measurement .<= 1)

        # Test with different resampling strategies
        repeated_holdout = RepeatedModel(
            base_model,
            n_repeats=2,
            resampling=Holdout(fraction_train=0.7, rng=42),
            measure=Accuracy(),
        )

        result_holdout = RepeatedRestarts.evaluate_seed!(
            mach, repeated_holdout, repeated_holdout.resampling; verbosity=0
        )
        @test result_holdout isa NamedTuple
        @test all(0 .<= result_holdout.measurement .<= 1)

        # Test with InSample
        repeated_insample = RepeatedModel(
            base_model, n_repeats=2, resampling=InSample(), measure=Accuracy(), refit=false
        )

        result_insample = RepeatedRestarts.evaluate_seed!(
            mach, repeated_insample, repeated_insample.resampling; verbosity=0
        )
        @test result_insample isa NamedTuple
        @test all(0 .<= result_insample.measurement .<= 1)
    end

    @testset "full fit integration test" begin
        X, y = MLJBase.@load_iris
        base_model = DecisionTreeClassifier(rng=42)

        # Test basic fit
        repeated = RepeatedModel(
            base_model, n_repeats=3, resampling=CV(nfolds=3, rng=42), measure=Accuracy()
        )

        mach = machine(repeated, X, y)
        @test_nowarn fit!(mach, verbosity=0)

        # Test fitresult structure
        @test mach.fitresult isa RepeatedRestarts.RepeatedFitResult
        @test length(mach.fitresult.inner_fitresult) >= 1
        @test length(mach.fitresult.seeds) >= 1
        @test 1 <= mach.fitresult.best_index <= length(mach.fitresult.inner_fitresult)

        # Test report structure
        rep = report(mach)
        @test haskey(rep, :best_index)
        @test haskey(rep, :best_seed)
        @test haskey(rep, :best_model)
        @test haskey(rep, :best_history_entry)
        @test haskey(rep, :history)
        @test haskey(rep, :seeds)
        @test haskey(rep, :inner_report)

        @test rep.best_index isa Integer
        @test rep.best_seed isa Integer
        @test rep.best_model isa typeof(base_model)
        @test rep.history isa AbstractVector
        @test rep.seeds isa AbstractVector{<:Integer}
        @test length(rep.history) == repeated.n_repeats
        @test length(rep.seeds) == repeated.n_repeats

        # Test fitted_params
        fp = fitted_params(mach)
        @test fp !== nothing  # Should return fitted params from inner model
    end

    @testset "return_mode tests" begin
        X, y = MLJBase.@load_iris
        base_model = DecisionTreeClassifier(rng=42)

        # Test return_mode = :best (default)
        repeated_best = RepeatedModel(
            base_model,
            n_repeats=3,
            return_mode=:best,
            resampling=CV(nfolds=2, rng=42),
            measure=Accuracy(),
        )

        mach_best = machine(repeated_best, X, y)
        fit!(mach_best, verbosity=0)

        @test length(mach_best.fitresult.inner_fitresult) == 1
        @test length(mach_best.fitresult.seeds) == 1
        @test mach_best.fitresult.best_index == 1

        # Test return_mode = :all
        repeated_all = RepeatedModel(
            base_model,
            n_repeats=3,
            return_mode=:all,
            resampling=CV(nfolds=2, rng=42),
            measure=Accuracy(),
        )

        mach_all = machine(repeated_all, X, y)
        fit!(mach_all, verbosity=0)

        @test length(mach_all.fitresult.inner_fitresult) == 3
        @test length(mach_all.fitresult.seeds) == 3
        @test 1 <= mach_all.fitresult.best_index <= 3
    end
end

# ==============================================================================
# MLJ Trait Tests
# ==============================================================================

@testset "MLJ Trait Tests" begin
    @testset "trait inheritance from wrapped models" begin
        # Test with DecisionTreeClassifier
        dt_classifier = DecisionTreeClassifier(rng=42)
        repeated_dt = RepeatedModel(dt_classifier, measure=Accuracy(), refit=false)

        # Test that traits are inherited properly
        @test MLJModelInterface.input_scitype(
            RepeatedRestarts.RepeatedModel{typeof(dt_classifier)}
        ) == MLJModelInterface.input_scitype(typeof(dt_classifier))

        @test MLJModelInterface.target_scitype(
            RepeatedRestarts.DeterministicRepeatedModel{typeof(dt_classifier)}
        ) == MLJModelInterface.target_scitype(typeof(dt_classifier))

        @test MLJModelInterface.is_pure_julia(
            RepeatedRestarts.RepeatedModel{typeof(dt_classifier)}
        ) == MLJModelInterface.is_pure_julia(typeof(dt_classifier))

        @test MLJModelInterface.supports_weights(
            RepeatedRestarts.RepeatedModel{typeof(dt_classifier)}
        ) == MLJModelInterface.supports_weights(typeof(dt_classifier))

        @test MLJModelInterface.supports_class_weights(
            RepeatedRestarts.RepeatedModel{typeof(dt_classifier)}
        ) == MLJModelInterface.supports_class_weights(typeof(dt_classifier))

        # Test with DecisionTreeRegressor
        dt_regressor = DecisionTreeRegressor(rng=42)
        repeated_reg = RepeatedModel(dt_regressor, measure=LPLoss(), refit=false)

        @test MLJModelInterface.input_scitype(
            RepeatedRestarts.RepeatedModel{typeof(dt_regressor)}
        ) == MLJModelInterface.input_scitype(typeof(dt_regressor))

        @test MLJModelInterface.target_scitype(
            RepeatedRestarts.DeterministicRepeatedModel{typeof(dt_regressor)}
        ) == MLJModelInterface.target_scitype(typeof(dt_regressor))
    end

    @testset "constructor trait" begin
        dt_classifier = DecisionTreeClassifier(rng=42)

        constructor_func = MLJModelInterface.constructor(
            RepeatedRestarts.RepeatedModel{typeof(dt_classifier)}
        )
        @test constructor_func == RepeatedRestarts.RepeatedModel{typeof(dt_classifier)}
    end

    @testset "iteration parameter trait" begin
        dt_classifier = DecisionTreeClassifier(rng=42)
        repeated = RepeatedModel(dt_classifier, measure=Accuracy(), refit=false)

        @test MLJModelInterface.iteration_parameter(typeof(repeated)) == :n_repeats
        @test MLJModelInterface.supports_training_losses(typeof(repeated)) == true
    end

    @testset "training losses" begin
        X, y = MLJBase.@load_iris
        dt_classifier = DecisionTreeClassifier(rng=42)
        repeated = RepeatedModel(
            dt_classifier, n_repeats=3, resampling=CV(nfolds=2, rng=42), measure=Accuracy()
        )

        mach = machine(repeated, X, y)
        fit!(mach, verbosity=0)

        rep = report(mach)
        losses = MLJModelInterface.training_losses(repeated, rep)

        # Training losses should be a vector of decreasing (or equal) values
        @test losses isa AbstractVector
        @test length(losses) == repeated.n_repeats
        @test all(losses[i] >= losses[i + 1] for i in 1:(length(losses) - 1))
    end

    @testset "reporting operations trait" begin
        dt_classifier = DecisionTreeClassifier(rng=42)
        repeated = RepeatedModel(dt_classifier, measure=Accuracy(), refit=false)

        # Should inherit reporting operations from wrapped model
        repeated_ops = MLJModelInterface.reporting_operations(typeof(repeated))
        base_ops = MLJModelInterface.reporting_operations(typeof(dt_classifier))
        @test repeated_ops == base_ops
    end

    @testset "feature importances trait" begin
        dt_classifier = DecisionTreeClassifier(rng=42)

        # Check if the base model reports feature importances
        reports_fi = MLJModelInterface.reports_feature_importances(typeof(dt_classifier))

        # RepeatedModel should inherit this trait
        repeated_reports_fi = MLJModelInterface.reports_feature_importances(
            RepeatedRestarts.RepeatedModel{typeof(dt_classifier)}
        )
        @test repeated_reports_fi == reports_fi
    end
end

# ==============================================================================
# Feature Importances Integration Tests
# ==============================================================================

@testset "Feature Importances Integration Tests" begin
    X, y = MLJBase.@load_iris

    @testset "feature_importances return_mode=:best" begin
        base_model = RandomForestClassifier(rng=42, n_trees=5)
        repeated = RepeatedModel(
            base_model;
            n_repeats=3,
            return_mode=:best,
            resampling=Holdout(rng=42),
            measure=LogLoss(),
            random_state=101,
        )
        mach = machine(repeated, X, y)
        fit!(mach, verbosity=0)

        fi = MLJModelInterface.feature_importances(repeated, mach.fitresult, report(mach))
        @test fi isa AbstractVector
        @test length(fi) > 0
        @test all(x -> x isa Pair, fi)
    end

    @testset "feature_importances return_mode=:all" begin
        base_model = RandomForestClassifier(rng=42, n_trees=5)
        repeated = RepeatedModel(
            base_model;
            n_repeats=3,
            return_mode=:all,
            resampling=Holdout(rng=42),
            measure=LogLoss(),
            random_state=101,
        )
        mach = machine(repeated, X, y)
        fit!(mach, verbosity=0)

        fi = MLJModelInterface.feature_importances(repeated, mach.fitresult, report(mach))
        @test fi isa Vector
        @test length(fi) == 3
        @test all(x -> x isa AbstractVector, fi)
        @test all(x -> all(p -> p isa Pair, x), fi)
    end
end

# ==============================================================================
# RepeatedFitResult Constructor Tests
# ==============================================================================

@testset "RepeatedFitResult Constructor Tests" begin
    @testset "valid construction" begin
        # Test basic valid construction
        inner_results = [1, 2, 3]
        seeds = [100, 200, 300]
        best_idx = 2

        result = RepeatedRestarts.RepeatedFitResult(inner_results, seeds, best_idx)

        @test result.inner_fitresult == inner_results
        @test result.seeds == seeds
        @test result.best_index == best_idx
    end

    @testset "boundary cases" begin
        # Single result
        result = RepeatedRestarts.RepeatedFitResult([42], [123], 1)
        @test length(result.inner_fitresult) == 1
        @test length(result.seeds) == 1
        @test result.best_index == 1

        # Multiple results with best_index at boundaries
        inner_results = ["a", "b", "c", "d"]
        seeds = [1, 2, 3, 4]

        # best_index = 1 (first)
        result1 = RepeatedRestarts.RepeatedFitResult(inner_results, seeds, 1)
        @test result1.best_index == 1

        # best_index = 4 (last)
        result4 = RepeatedRestarts.RepeatedFitResult(inner_results, seeds, 4)
        @test result4.best_index == 4
    end

    @testset "constructor validation errors" begin
        inner_results = [1, 2, 3]
        seeds = [100, 200, 300]

        # Mismatched lengths should throw assertion error
        @test_throws AssertionError RepeatedRestarts.RepeatedFitResult(
            inner_results,
            [100, 200],  # shorter seeds
            2,
        )

        @test_throws AssertionError RepeatedRestarts.RepeatedFitResult(
            [1, 2],  # shorter inner_results
            seeds,
            2,
        )

        # Invalid best_index should throw assertion error
        @test_throws AssertionError RepeatedRestarts.RepeatedFitResult(
            inner_results,
            seeds,
            0,  # best_index too small
        )

        @test_throws AssertionError RepeatedRestarts.RepeatedFitResult(
            inner_results,
            seeds,
            4,  # best_index too large
        )

        @test_throws AssertionError RepeatedRestarts.RepeatedFitResult(
            inner_results,
            seeds,
            -1,  # negative best_index
        )
    end

    @testset "constructor with different data types" begin
        # Test with different inner result types
        string_results = ["model1", "model2"]
        seeds = [1, 2]

        result = RepeatedRestarts.RepeatedFitResult(string_results, seeds, 1)
        @test result.inner_fitresult == string_results

        # Test with mixed types in inner results
        mixed_results = Any[42, "test", 3.14]
        mixed_seeds = [10, 20, 30]

        result = RepeatedRestarts.RepeatedFitResult(mixed_results, mixed_seeds, 2)
        @test result.inner_fitresult == mixed_results
        @test result.best_index == 2
    end

    @testset "immutability" begin
        inner_results = [1, 2, 3]
        seeds = [100, 200, 300]
        result = RepeatedRestarts.RepeatedFitResult(inner_results, seeds, 2)

        # RepeatedFitResult should be immutable - these should fail
        @test_throws ErrorException result.inner_fitresult = [4, 5, 6]
        @test_throws ErrorException result.seeds = [400, 500, 600]
        @test_throws ErrorException result.best_index = 3
    end
end

# ==============================================================================
# Aggregation Helper Tests
# ==============================================================================

@testset "Aggregation Helper Tests" begin
    @testset "vote_majority!" begin
        # Basic majority vote
        preds = [[:a, :b, :a], [:a, :a, :b], [:b, :a, :a]]
        out = similar(preds[1])
        result = RepeatedRestarts.vote_majority!(out, preds)
        @test result[1] == :a  # 2 votes :a vs 1 vote :b
        @test result[2] == :a  # 2 votes :a vs 1 vote :b
        @test result[3] == :a  # 2 votes :a vs 1 vote :b

        # Unanimous vote
        preds_unanimous = [[:x, :y], [:x, :y], [:x, :y]]
        out2 = similar(preds_unanimous[1])
        result2 = RepeatedRestarts.vote_majority!(out2, preds_unanimous)
        @test result2 == [:x, :y]
    end

    @testset "aggregate_probs" begin
        # Create two sets of UnivariateFinite predictions
        classes = MLJBase.categorical(["a", "b", "c"])

        # First prediction set: high confidence on "a"
        P1 = [0.8 0.1 0.1; 0.1 0.8 0.1]
        uf1 = MLJBase.UnivariateFinite(classes, P1)

        # Second prediction set: high confidence on "b"
        P2 = [0.2 0.7 0.1; 0.1 0.2 0.7]
        uf2 = MLJBase.UnivariateFinite(classes, P2)

        result = RepeatedRestarts.aggregate_probs([uf1, uf2])

        # Averaged probabilities should sum to 1.0
        for i in 1:2
            probs = [MLJBase.pdf(result[i], c) for c in classes]
            @test sum(probs) ≈ 1.0
        end

        # Check specific averaged values
        @test MLJBase.pdf(result[1], classes[1]) ≈ 0.5  # (0.8 + 0.2) / 2
        @test MLJBase.pdf(result[1], classes[2]) ≈ 0.4  # (0.1 + 0.7) / 2
        @test MLJBase.pdf(result[1], classes[3]) ≈ 0.1  # (0.1 + 0.1) / 2
    end
end

# ==============================================================================
# Predict Tests
# ==============================================================================

@testset "Predict Tests" begin
    X, y = MLJBase.@load_iris

    @testset "predict return_mode=:best (classifier)" begin
        base_model = DecisionTreeClassifier(rng=42)
        repeated = RepeatedModel(
            base_model;
            n_repeats=3,
            return_mode=:best,
            resampling=Holdout(rng=42),
            measure=LogLoss(),
            random_state=101,
        )
        mach = machine(repeated, X, y)
        fit!(mach, verbosity=0)

        yhat = MLJBase.predict(mach, X)
        @test length(yhat) == length(y)
        @test eltype(yhat) <: MLJBase.UnivariateFinite
    end

    @testset "predict return_mode=:best (regressor)" begin
        X_reg = (x1=rand(100), x2=rand(100))
        y_reg = 2 .* X_reg.x1 .+ 3 .* X_reg.x2 .+ 0.1 .* randn(100)

        base_model = DecisionTreeRegressor(rng=42)
        repeated = RepeatedModel(
            base_model;
            n_repeats=3,
            return_mode=:best,
            resampling=Holdout(rng=42),
            measure=LPLoss(),
            random_state=101,
        )
        mach = machine(repeated, X_reg, y_reg)
        fit!(mach, verbosity=0)

        yhat = MLJBase.predict(mach, X_reg)
        @test length(yhat) == 100
        @test eltype(yhat) <: Real
    end

    @testset "predict return_mode=:all" begin
        n_repeats = 3
        base_model = DecisionTreeClassifier(rng=42)
        repeated = RepeatedModel(
            base_model;
            n_repeats=n_repeats,
            return_mode=:all,
            resampling=Holdout(rng=42),
            measure=LogLoss(),
            random_state=101,
        )
        mach = machine(repeated, X, y)
        fit!(mach, verbosity=0)

        yhat = MLJBase.predict(mach, X)
        @test yhat isa Vector
        @test length(yhat) == n_repeats
        for yh in yhat
            @test length(yh) == length(y)
        end
    end

    @testset "predict return_mode=:aggregate (mean)" begin
        base_model = DecisionTreeClassifier(rng=42)
        repeated = RepeatedModel(
            base_model;
            n_repeats=3,
            return_mode=:aggregate,
            aggregation=:mean,
            resampling=Holdout(rng=42),
            measure=LogLoss(),
            random_state=101,
        )
        mach = machine(repeated, X, y)
        fit!(mach, verbosity=0)

        yhat = MLJBase.predict(mach, X)
        @test length(yhat) == length(y)
        @test eltype(yhat) <: MLJBase.UnivariateFinite

        # All probabilities should sum to 1.0
        classes = MLJBase.classes(yhat[1])
        for i in eachindex(yhat)
            probs = [MLJBase.pdf(yhat[i], c) for c in classes]
            @test sum(probs) ≈ 1.0
        end
    end

    @testset "predict return_mode=:aggregate (vote)" begin
        base_model = DecisionTreeClassifier(rng=42)
        repeated = RepeatedModel(
            base_model;
            n_repeats=5,
            return_mode=:aggregate,
            aggregation=:vote,
            resampling=Holdout(rng=42),
            measure=LogLoss(),
            random_state=101,
        )
        mach = machine(repeated, X, y)
        fit!(mach, verbosity=0)

        yhat = MLJBase.predict(mach, X)
        @test length(yhat) == length(y)
        @test eltype(yhat) <: MLJBase.UnivariateFinite

        # Indicator distributions: exactly one class should have prob ≈ 1.0
        classes = MLJBase.classes(yhat[1])
        for i in eachindex(yhat)
            probs = [MLJBase.pdf(yhat[i], c) for c in classes]
            @test maximum(probs) ≈ 1.0
            @test count(≈(1.0), probs) == 1
        end
    end

    @testset "predict return_mode=:aggregate (mean, regressor)" begin
        X_reg = (x1=rand(50), x2=rand(50))
        y_reg = 2 .* X_reg.x1 .+ 3 .* X_reg.x2 .+ 0.1 .* randn(50)

        base_model = DecisionTreeRegressor(rng=42)
        repeated = RepeatedModel(
            base_model;
            n_repeats=3,
            return_mode=:aggregate,
            aggregation=:mean,
            resampling=Holdout(rng=42),
            measure=LPLoss(),
            random_state=101,
        )
        mach = machine(repeated, X_reg, y_reg)
        fit!(mach, verbosity=0)

        yhat = MLJBase.predict(mach, X_reg)
        @test length(yhat) == 50
        @test eltype(yhat) <: Real
    end

    @testset "predict return_mode=:aggregate (median, regressor)" begin
        X_reg = (x1=rand(50), x2=rand(50))
        y_reg = 2 .* X_reg.x1 .+ 3 .* X_reg.x2 .+ 0.1 .* randn(50)

        base_model = DecisionTreeRegressor(rng=42)
        repeated = RepeatedModel(
            base_model;
            n_repeats=3,
            return_mode=:aggregate,
            aggregation=:median,
            resampling=Holdout(rng=42),
            measure=LPLoss(),
            random_state=101,
        )
        mach = machine(repeated, X_reg, y_reg)
        fit!(mach, verbosity=0)

        yhat = MLJBase.predict(mach, X_reg)
        @test length(yhat) == 50
        @test eltype(yhat) <: Real
    end

    @testset "predict reproducibility" begin
        base_model = DecisionTreeClassifier(rng=42)

        repeated1 = RepeatedModel(
            base_model;
            n_repeats=3,
            return_mode=:best,
            resampling=Holdout(rng=42),
            measure=LogLoss(),
            random_state=101,
        )
        mach1 = machine(repeated1, X, y)
        fit!(mach1, verbosity=0)
        yhat1 = MLJBase.predict(mach1, X)

        repeated2 = RepeatedModel(
            base_model;
            n_repeats=3,
            return_mode=:best,
            resampling=Holdout(rng=42),
            measure=LogLoss(),
            random_state=101,
        )
        mach2 = machine(repeated2, X, y)
        fit!(mach2, verbosity=0)
        yhat2 = MLJBase.predict(mach2, X)

        # Compare probabilities
        classes_ref = MLJBase.classes(yhat1[1])
        for c in classes_ref
            @test MLJBase.pdf.(yhat1, c) ≈ MLJBase.pdf.(yhat2, c)
        end
    end
end

# ==============================================================================
# Median Validation Tests
# ==============================================================================

@testset "Median validation for probabilistic models" begin
    base_model = DecisionTreeClassifier(rng=42)

    # :median on probabilistic model should be reset to :mean with warning
    repeated = @test_logs(
        (:warn, r"aggregation=:median is not supported for probabilistic"),
        RepeatedModel(base_model; aggregation=:median, measure=LogLoss(), refit=false),
    )
    @test repeated.aggregation == :mean

    # :median on deterministic model should be fine
    reg_model = DecisionTreeRegressor(rng=42)
    repeated_det = RepeatedModel(
        reg_model; aggregation=:median, measure=LPLoss(), refit=false
    )
    @test repeated_det.aggregation == :median
end

# ==============================================================================
# Update Tests
# ==============================================================================

@testset "Update Tests" begin
    X, y = MLJBase.@load_iris

    @testset "incremental n_repeats increase" begin
        base_model = DecisionTreeClassifier(rng=42)
        repeated = RepeatedModel(
            base_model;
            n_repeats=3,
            resampling=Holdout(rng=42),
            measure=LogLoss(),
            random_state=101,
        )
        mach = machine(repeated, X, y)
        fit!(mach, verbosity=0)

        rep_old = report(mach)
        old_history = deepcopy(rep_old.history)
        old_seeds = deepcopy(rep_old.seeds)
        @test length(old_history) == 3
        @test length(old_seeds) == 3

        # Increase n_repeats — should trigger update, not full refit
        repeated.n_repeats = 5
        fit!(mach, verbosity=0)

        rep_new = report(mach)
        @test length(rep_new.history) == 5
        @test length(rep_new.seeds) == 5

        # First 3 seeds should match
        @test rep_new.seeds[1:3] == old_seeds

        # First 3 history entries should match
        for i in 1:3
            @test rep_new.history[i].measurement == old_history[i].measurement
        end
    end

    @testset "changed random_state falls back to full refit" begin
        base_model = DecisionTreeClassifier(rng=42)
        repeated = RepeatedModel(
            base_model;
            n_repeats=3,
            resampling=Holdout(rng=42),
            measure=LogLoss(),
            random_state=101,
        )
        mach = machine(repeated, X, y)
        fit!(mach, verbosity=0)

        # Change random_state and increase n_repeats
        repeated.random_state = 999
        repeated.n_repeats = 5
        @test_logs (:warn, "wrapper.random_state has changed. Falling back to full refit.") match_mode=:any fit!(
            mach, verbosity=1
        )

        @test length(report(mach).history) == 5
        @test length(report(mach).seeds) == 5
    end

    @testset "decreased n_repeats falls back to full refit" begin
        base_model = DecisionTreeClassifier(rng=42)
        repeated = RepeatedModel(
            base_model;
            n_repeats=5,
            resampling=Holdout(rng=42),
            measure=LogLoss(),
            random_state=101,
        )
        mach = machine(repeated, X, y)
        fit!(mach, verbosity=0)

        @test length(report(mach).history) == 5

        # Decrease n_repeats — update falls back to a full refit
        repeated.n_repeats = 3
        @test_logs (:warn, "n_repeats decreased or unchanged. Falling back to full refit.") match_mode=:any fit!(
            mach, verbosity=1
        )

        @test length(report(mach).history) == 3
        @test length(report(mach).seeds) == 3
    end
end

# ==============================================================================
# Acceleration Tests
# ==============================================================================

@testset "Acceleration Tests" begin
    X, y = MLJBase.@load_iris
    base_model = DecisionTreeClassifier(rng=42)

    @testset "CPUThreads produces same result as CPU1" begin
        # CPU1 baseline
        repeated_seq = RepeatedModel(
            base_model;
            n_repeats=5,
            resampling=Holdout(rng=42),
            measure=LogLoss(),
            random_state=101,
            acceleration=CPU1(),
        )
        mach_seq = machine(repeated_seq, X, y)
        fit!(mach_seq, verbosity=0)
        rep_seq = report(mach_seq)

        # CPUThreads
        repeated_par = RepeatedModel(
            base_model;
            n_repeats=5,
            resampling=Holdout(rng=42),
            measure=LogLoss(),
            random_state=101,
            acceleration=CPUThreads(),
        )
        mach_par = machine(repeated_par, X, y)
        fit!(mach_par, verbosity=0)
        rep_par = report(mach_par)

        # Same seeds and best selection
        @test rep_seq.seeds == rep_par.seeds
        @test rep_seq.best_index == rep_par.best_index
        @test rep_seq.best_seed == rep_par.best_seed

        # Same predictions
        yhat_seq = MLJBase.predict(mach_seq, X)
        yhat_par = MLJBase.predict(mach_par, X)
        classes_ref = MLJBase.classes(yhat_seq[1])
        for c in classes_ref
            @test MLJBase.pdf.(yhat_seq, c) ≈ MLJBase.pdf.(yhat_par, c)
        end
    end

    @testset "CPUProcesses produces same result as CPU1" begin
        # CPU1 baseline
        repeated_seq = RepeatedModel(
            base_model;
            n_repeats=4,
            resampling=Holdout(rng=42),
            measure=LogLoss(),
            random_state=101,
            acceleration=CPU1(),
        )
        mach_seq = machine(repeated_seq, X, y)
        fit!(mach_seq, verbosity=0)
        rep_seq = report(mach_seq)

        # CPUProcesses should match CPU1 (falls back to CPU1 if no workers)
        repeated_par = RepeatedModel(
            base_model;
            n_repeats=4,
            resampling=Holdout(rng=42),
            measure=LogLoss(),
            random_state=101,
            acceleration=CPUProcesses(),
        )
        mach_par = machine(repeated_par, X, y)
        fit!(mach_par, verbosity=0)
        rep_par = report(mach_par)

        # Same seeds and best selection
        @test rep_seq.seeds == rep_par.seeds
        @test rep_seq.best_index == rep_par.best_index
        @test rep_seq.best_seed == rep_par.best_seed

        # Same predictions
        yhat_seq = MLJBase.predict(mach_seq, X)
        yhat_par = MLJBase.predict(mach_par, X)
        classes_ref = MLJBase.classes(yhat_seq[1])
        for c in classes_ref
            @test MLJBase.pdf.(yhat_seq, c) ≈ MLJBase.pdf.(yhat_par, c)
        end
    end
end

# ==============================================================================
# Deterministic Aggregation Tests
# ==============================================================================

@testset "Deterministic Aggregation Tests" begin
    X_reg = (x1=rand(50), x2=rand(50))
    y_reg = 2 .* X_reg.x1 .+ 3 .* X_reg.x2 .+ 0.1 .* randn(50)

    @testset "aggregate :vote (regressor)" begin
        base_model = DecisionTreeRegressor(rng=42)
        repeated = RepeatedModel(
            base_model;
            n_repeats=3,
            return_mode=:aggregate,
            aggregation=:vote,
            resampling=Holdout(rng=42),
            measure=LPLoss(),
            random_state=101,
        )
        mach = machine(repeated, X_reg, y_reg)
        fit!(mach, verbosity=0)

        yhat = MLJBase.predict(mach, X_reg)
        @test length(yhat) == 50
        @test eltype(yhat) <: Real
    end

    @testset "aggregate :mode (regressor)" begin
        base_model = DecisionTreeRegressor(rng=42)
        repeated = RepeatedModel(
            base_model;
            n_repeats=3,
            return_mode=:aggregate,
            aggregation=:mode,
            resampling=Holdout(rng=42),
            measure=LPLoss(),
            random_state=101,
        )
        mach = machine(repeated, X_reg, y_reg)
        fit!(mach, verbosity=0)

        yhat = MLJBase.predict(mach, X_reg)
        @test length(yhat) == 50
        @test eltype(yhat) <: Real
    end
end

# ==============================================================================
# Reporting Predict Tests
# ==============================================================================

# A dummy probabilistic classifier that reports on predict
mutable struct ReportingClassifier <: MMI.Probabilistic
    rng::Int
end
ReportingClassifier(; rng=42) = ReportingClassifier(rng)

MMI.reporting_operations(::Type{<:ReportingClassifier}) = (:predict,)

function MMI.fit(model::ReportingClassifier, verbosity::Int, X, y)
    classes = MMI.classes(y[1])
    rng_state = model.rng
    fitresult = (classes=classes, rng_state=rng_state)
    cache = nothing
    report = (n_classes=length(classes),)
    return fitresult, cache, report
end

function MMI.predict(model::ReportingClassifier, fitresult, X)
    classes = fitresult.classes
    n = MMI.nrows(X)
    L = length(classes)
    rng = Random.Xoshiro(fitresult.rng_state)
    P = rand(rng, n, L)
    P ./= sum(P; dims=2)
    predictions = MMI.UnivariateFinite(classes, P)
    predict_report = (seed_used=fitresult.rng_state, n_predictions=n)
    return predictions, predict_report
end

MMI.input_scitype(::Type{<:ReportingClassifier}) = MMI.Table(MMI.Continuous)
MMI.target_scitype(::Type{<:ReportingClassifier}) = AbstractVector{<:MMI.Finite}

@testset "Reporting Predict Tests" begin
    X, y = MLJBase.@load_iris

    @testset "wrapper inherits reporting_operations" begin
        base_model = ReportingClassifier(rng=42)
        repeated = RepeatedModel(base_model, n_repeats=3, measure=LogLoss(), refit=false)

        wrapper_ops = MMI.reporting_operations(typeof(repeated))
        base_ops = MMI.reporting_operations(typeof(base_model))
        @test wrapper_ops == base_ops
        @test :predict in wrapper_ops
    end

    @testset "return_mode=:best with reporting model" begin
        base_model = ReportingClassifier(rng=42)
        repeated = RepeatedModel(
            base_model;
            n_repeats=3,
            return_mode=:best,
            resampling=Holdout(rng=42),
            measure=LogLoss(),
            random_state=101,
        )
        mach = machine(repeated, X, y)
        fit!(mach, verbosity=0)

        # Machine-level predict should return only predictions
        yhat = MLJBase.predict(mach, X)
        @test length(yhat) == length(y)
        @test eltype(yhat) <: MLJBase.UnivariateFinite

        # Report should contain the predict_report fields (merged flat by MLJ)
        rep = report(mach)
        @test haskey(rep, :seed_used)
        @test haskey(rep, :n_predictions)
        @test rep.n_predictions == length(y)
    end

    @testset "return_mode=:all with reporting model" begin
        n_repeats = 3
        base_model = ReportingClassifier(rng=42)
        repeated = RepeatedModel(
            base_model;
            n_repeats=n_repeats,
            return_mode=:all,
            resampling=Holdout(rng=42),
            measure=LogLoss(),
            random_state=101,
        )
        mach = machine(repeated, X, y)
        fit!(mach, verbosity=0)

        # Machine-level predict should return vector of predictions
        yhat = MLJBase.predict(mach, X)
        @test yhat isa Vector
        @test length(yhat) == n_repeats
        for yh in yhat
            @test length(yh) == length(y)
            @test eltype(yh) <: MLJBase.UnivariateFinite
        end

        # Report should contain predict_reports for all repeats (merged flat by MLJ)
        rep = report(mach)
        @test haskey(rep, :predict_reports)
        @test length(rep.predict_reports) == n_repeats
        for pr in rep.predict_reports
            @test haskey(pr, :seed_used)
            @test haskey(pr, :n_predictions)
            @test pr.n_predictions == length(y)
        end
    end

    @testset "return_mode=:aggregate with reporting model" begin
        n_repeats = 3
        base_model = ReportingClassifier(rng=42)
        repeated = RepeatedModel(
            base_model;
            n_repeats=n_repeats,
            return_mode=:aggregate,
            aggregation=:mean,
            resampling=Holdout(rng=42),
            measure=LogLoss(),
            random_state=101,
        )
        mach = machine(repeated, X, y)
        fit!(mach, verbosity=0)

        # Machine-level predict should return aggregated predictions
        yhat = MLJBase.predict(mach, X)
        @test length(yhat) == length(y)
        @test eltype(yhat) <: MLJBase.UnivariateFinite

        # Probabilities should sum to 1.0
        classes = MLJBase.classes(yhat[1])
        for i in eachindex(yhat)
            probs = [MLJBase.pdf(yhat[i], c) for c in classes]
            @test sum(probs) ≈ 1.0
        end

        # Report should contain predict_reports and aggregation info (merged flat by MLJ)
        rep = report(mach)
        @test haskey(rep, :predict_reports)
        @test haskey(rep, :aggregation)
        @test rep.aggregation == :mean
        @test length(rep.predict_reports) == n_repeats
    end

    @testset "low-level MMI.predict with reporting model" begin
        n_repeats = 3
        base_model = ReportingClassifier(rng=42)

        # Test :all mode at low level
        repeated_all = RepeatedModel(
            base_model;
            n_repeats=n_repeats,
            return_mode=:all,
            resampling=Holdout(rng=42),
            measure=LogLoss(),
            random_state=101,
        )
        mach_all = machine(repeated_all, X, y)
        fit!(mach_all, verbosity=0)

        # Low-level predict should return (predictions, wrapper_report)
        result = MMI.predict(repeated_all, mach_all.fitresult, X)
        @test result isa Tuple
        @test length(result) == 2
        preds, wrapper_report = result
        @test preds isa Vector
        @test length(preds) == n_repeats
        @test haskey(wrapper_report, :predict_reports)

        # Test :aggregate mode at low level
        repeated_agg = RepeatedModel(
            base_model;
            n_repeats=n_repeats,
            return_mode=:aggregate,
            aggregation=:mean,
            resampling=Holdout(rng=42),
            measure=LogLoss(),
            random_state=101,
        )
        mach_agg = machine(repeated_agg, X, y)
        fit!(mach_agg, verbosity=0)

        result_agg = MMI.predict(repeated_agg, mach_agg.fitresult, X)
        @test result_agg isa Tuple
        @test length(result_agg) == 2
        preds_agg, wrapper_report_agg = result_agg
        @test length(preds_agg) == length(y)
        @test haskey(wrapper_report_agg, :predict_reports)
        @test haskey(wrapper_report_agg, :aggregation)
    end

    @testset "non-reporting model unchanged behaviour" begin
        # DecisionTreeClassifier does not report on predict
        base_model = DecisionTreeClassifier(rng=42)
        @test !(:predict in MMI.reporting_operations(typeof(base_model)))

        n_repeats = 3

        # :all mode should return plain vector of predictions
        repeated_all = RepeatedModel(
            base_model;
            n_repeats=n_repeats,
            return_mode=:all,
            resampling=Holdout(rng=42),
            measure=LogLoss(),
            random_state=101,
        )
        mach_all = machine(repeated_all, X, y)
        fit!(mach_all, verbosity=0)

        result_all = MMI.predict(repeated_all, mach_all.fitresult, X)
        @test result_all isa Vector
        @test !(result_all isa Tuple)
        @test length(result_all) == n_repeats

        # :aggregate mode should return plain aggregated predictions
        repeated_agg = RepeatedModel(
            base_model;
            n_repeats=n_repeats,
            return_mode=:aggregate,
            aggregation=:mean,
            resampling=Holdout(rng=42),
            measure=LogLoss(),
            random_state=101,
        )
        mach_agg = machine(repeated_agg, X, y)
        fit!(mach_agg, verbosity=0)

        result_agg = MMI.predict(repeated_agg, mach_agg.fitresult, X)
        @test !(result_agg isa Tuple)
        @test length(result_agg) == length(y)
    end
end
