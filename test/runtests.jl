# Run all tests.
module TestRunner

using FactCheck

include("test_util.jl")
include("test_loss.jl")
include("test_gb.jl")
include("test_gb_dt.jl")
include("test_gb_bl.jl")
include("test_ml.jl")
include("test_system.jl")

exitstatus()

end # module
