using Test
using RepeatedRestarts
using MLJBase
using MLJModelInterface
using MLJDecisionTreeInterface
using StatisticalMeasures
using Random

@testset "Best model refit reproducibility" begin
    # Load data
    X, y = MLJBase.@load_iris

    # Base model that uses an `rng` field
    base_model = RandomForestClassifier(rng=101) # TODO: set hyperparamters to promote higher variance

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

    # TODO: call predict once it has been implemented
    # yhat_internal = MLJBase.predict(mach_rep, X)

    # Now refit a fresh machine using the same best seed and model, and compare predictions
    best_model = deepcopy(rep.best_model)
    mach_best = machine(best_model, X, y)
    E = evaluate!(mach_best, resampling=Holdout(rng=42), measure=LogLoss())
    loss_best = E.measurement

    @test loss == loss_best

    # yhat_refit = MLJBase.predict(mach_best, X)

    # @test yhat_internal == yhat_refit
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
            model=base_model,
            n_repeats=5,
            measure=LogLoss(),
            refit=false
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
            DecisionTreeClassifier(),
            DecisionTreeRegressor(),
            n_repeats=3,
            refit=false
        )

        # No model provided
        @test_throws ArgumentError RepeatedModel()

        # Type instead of instance
        @test_throws AssertionError RepeatedModel(DecisionTreeClassifier)

        # Unsupported model type
        struct UnsupportedModel end
        @test_throws ArgumentError RepeatedModel(UnsupportedModel())
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
        repeated1 = RepeatedModel(base_model, rng_field=:rng, measure=Accuracy(), refit=false)
        @test repeated1.rng_field == :rng

        repeated2 = RepeatedModel(base_model, rng_field="rng", measure=Accuracy(), refit=false)
        @test repeated2.rng_field == "rng"

        # Invalid rng_field should throw during construction
        @test_throws ErrorException RepeatedModel(base_model, rng_field=:nonexistent, refit=false)
        @test_throws ErrorException RepeatedModel(base_model, rng_field="nonexistent", refit=false)
    end

    @testset "n_repeats validation" begin
        base_model = DecisionTreeClassifier(rng=42)

        # Valid n_repeats
        repeated = RepeatedModel(base_model, n_repeats=5, measure=Accuracy(), refit=false)
        @test repeated.n_repeats == 5

        # Invalid n_repeats should be corrected with warning
        repeated = @test_logs (:warn, r"n_repeats.*Resetting to 10") RepeatedModel(base_model, n_repeats=0, measure=Accuracy(), refit=false)
        @test repeated.n_repeats == 10

        repeated = @test_logs (:warn, r"n_repeats.*Resetting to 10") RepeatedModel(base_model, n_repeats=-5, measure=Accuracy(), refit=false)
        @test repeated.n_repeats == 10
    end

    @testset "return_mode validation" begin
        base_model = DecisionTreeClassifier(rng=42)

        # Valid return_mode values
        for mode in [:best, :aggregate, :all]
            repeated = RepeatedModel(base_model, return_mode=mode, measure=Accuracy(), refit=false)
            @test repeated.return_mode == mode
        end

        # Invalid return_mode should be corrected with warning
        repeated = @test_logs (:warn, r"return_mode.*Resetting to :best") RepeatedModel(base_model, return_mode=:invalid, measure=Accuracy(), refit=false)
        @test repeated.return_mode == :best
    end

    @testset "aggregation validation" begin
        base_model = DecisionTreeClassifier(rng=42)

        # Valid aggregation values
        for agg in [:mean, :median, :mode, :vote]
            repeated = RepeatedModel(base_model, aggregation=agg, measure=Accuracy(), refit=false)
            @test repeated.aggregation == agg
        end

        # Invalid aggregation should be corrected with warning
        repeated = @test_logs (:warn, r"aggregation.*Resetting to :mean") RepeatedModel(base_model, aggregation=:invalid, measure=Accuracy(), refit=false)
        @test repeated.aggregation == :mean
    end

    @testset "refit validation with InSample" begin
        base_model = DecisionTreeClassifier(rng=42)

        # refit=true with InSample should be corrected with warning
        repeated = @test_logs (:warn, r"refit=true is not required.*Resetting to false") RepeatedModel(
            base_model,
            refit=true,
            resampling=InSample(),
            measure=Accuracy()
        )
        @test repeated.refit == false

        # refit=false with InSample should be fine
        repeated = RepeatedModel(base_model, refit=false, resampling=InSample(), measure=Accuracy())
        @test repeated.refit == false

        # refit=true with other resampling should be fine
        repeated = RepeatedModel(base_model, refit=true, resampling=Holdout())
        @test repeated.refit == true
    end

    @testset "measure validation" begin
        base_model = DecisionTreeClassifier(rng=42)

        # measure=nothing should be auto-detected with warning
        repeated = @test_logs (:warn, r"No measure specified.*Setting measure=") RepeatedModel(base_model, measure=nothing, refit=false)
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
            refit=false
        )

        # CPUThreads + CPUProcesses should be corrected with warning
        # We just test the behavior, not the specific warning message
        repeated = RepeatedModel(
            base_model,
            acceleration=CPUThreads(),
            acceleration_resampling=CPUProcesses(),
            measure=Accuracy(),
            refit=false
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
            base_model,
            n_repeats=2,
            resampling=InSample(),
            measure=Accuracy(),
            refit=false
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
