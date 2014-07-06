# Run all tests.
module TestRunner

using FactCheck

include("test_util.jl")
include("test_loss.jl")
include("test_gb.jl")
include("test_gb_dt.jl")
include("test_gb_learner.jl")
include("test_ml.jl")

exitstatus()

end # module
