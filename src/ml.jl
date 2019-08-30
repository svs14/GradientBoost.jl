# Machine learning API for gradient boosting.
module ML

using GradientBoost.LossFunctions
using GradientBoost.GB
using GradientBoost.GBDecisionTree
using GradientBoost.GBBaseLearner

export GBLearner,
       fit!,
       predict!,
       LossFunction,
       LeastSquares,
       LeastAbsoluteDeviation,
       BinomialDeviance,
       GBDT,
       GBBL,
       learner_fit,
       learner_predict


# Gradient boosting learner as defined by ML API.
mutable struct GBLearner
  algorithm::GBAlgorithm
  output::Symbol
  model

  function GBLearner(algorithm, output=:regression)
    new(algorithm, output, nothing)
  end
end

function fit!(gbl::GBLearner, instances, labels)
  error("Instance type: $(typeof(instances))
    and label type: $(typeof(labels)) together is currently not supported.")
end
function predict!(gbl::GBLearner, instances)
  error("Instance type: $(typeof(instances)) is currently not supported.")
end

function fit!(gbl::GBLearner,
  instances::Matrix{Float64}, labels::Vector{Float64})

  # No special processing required.
  gbl.model = fit(gbl.algorithm, instances, labels)
end

function predict!(gbl::GBLearner,
  instances::Matrix{Float64})

  # Predict with GB algorithm
  predictions = predict(gbl.model, instances)

  # Postprocess according to output and loss function
  predictions = postprocess_pred(
    gbl.output, gbl.algorithm.loss_function, predictions
  )

  predictions
end

# Postprocesses predictions according to
# output and loss function.
function postprocess_pred(
  output::Symbol, lf::LossFunction, predictions::Vector{Float64})

  if output == :class && typeof(lf) <: BinomialDeviance
    return round.(logistic(predictions), RoundNearestTiesAway)
  elseif output == :class_prob && typeof(lf) <: BinomialDeviance
    return logistic(predictions)
  elseif output == :regression && !(typeof(lf) <: BinomialDeviance)
    return predictions
  else
    error("Cannot handle $(output) and $(typeof(lf)) together.")
  end
end

# Logistic function.
function logistic(x)
  1 ./ (1 .+ exp.(-x))
end

end # module
