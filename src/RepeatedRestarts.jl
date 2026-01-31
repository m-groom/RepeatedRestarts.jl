module RepeatedRestarts

using MLJModelInterface
using Random
using Statistics
using MLJTuning
using MLJBase
using ComputationalResources
using Distributed

# Include model
include("helpers.jl")
include("main.jl")
export RepeatedModel

# Common package metadata
const PKG_METADATA = (
    package_name="RepeatedRestarts",
    package_uuid="c5ef7d8a-4f32-44df-9bf1-d1a5ca83e999",
    package_url="https://github.com/m-groom/RepeatedRestarts.jl",
    package_license="MIT",
)

MLJModelInterface.metadata_pkg(
    RepeatedModel;
    PKG_METADATA...,
    is_wrapper=true,  # RepeatedModel is a wrapper model
)

end # module RepeatedRestarts
