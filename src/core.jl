# build one history entry by evaluating `mach` (resampling or in-sample)
function evaluate_seed!(mach, wrapper::RepeatedModel; verbosity::Int=0)
    E = evaluate!(
        mach;
        resampling=wrapper.resampling,
        measure=wrapper.measure,
        operation=wrapper.operation, # if nothing, MLJ determines from measure (predict/predict_mode)
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

# TODO: core fit function to be called from both MMI.fit anbd MMI.update
