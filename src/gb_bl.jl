# Gradient Boosted Learner
module GBBaseLearner

export GBBL,
       build_base_func,
       learner_fit,
       learner_predict

importall GradientBoost.GB
importall GradientBoost.LossFunctions
importall GradientBoost.Util

# Gradient boosted base learner algorithm.
type GBBL <: GBAlgorithm
  loss_function::LossFunction
  sampling_rate::FloatingPoint
  learning_rate::FloatingPoint
  num_iterations::Int
  learner

  function GBBL(learner; loss_function=LeastSquares(),
    sampling_rate=0.8, learning_rate=0.1, 
    num_iterations=100)

    new(loss_function, sampling_rate, learning_rate, num_iterations, learner)
  end
end

function GB.build_base_func(
    gb::GBBL,
    instances,
    labels,
    prev_func_pred,
    psuedo)

  # Train learner
  lf = gb.loss_function
  learner = gb.learner
  model = learner_fit(lf, learner, instances, psuedo)
  psuedo_pred = learner_predict(lf, learner, model, instances)
  model_const =
    fit_best_constant(lf, labels, psuedo, psuedo_pred, prev_func_pred)

  # Produce function that delegates prediction to model
  return (instances) ->
    model_const .* learner_predict(lf, learner, model, instances)
end

# Fits base learner.
# The learner must be instantiated within this function.
#
# @param lf Loss function (typically, this is not used).
# @param learner Base learner.
# @param instances Instances.
# @param labels Labels.
# @return Model.
function learner_fit(lf::LossFunction, learner, instances, labels)
  error("This function must be implemented by $(learner) for $(lf)")
end

# Predicts on base learner.
#
# @param lf Loss function (typically, this is not used).
# @param learner Base learner.
# @param model Model produced by base learner.
# @param instances Instances.
# @return Predictions.
function learner_predict(lf::LossFunction, learner, model, instances)
  error("This function must be implemented by $(learner) for $(lf)")
end

# Loss function fits
function fit_best_constant(lf::LeastSquares,
  labels, psuedo, psuedo_pred, prev_func_pred)

  # No refitting required
  1.0
end

function fit_best_constant(lf::LeastAbsoluteDeviation,
  labels, psuedo, psuedo_pred, prev_func_pred)

  weights = abs(psuedo_pred)
  values = labels .- prev_func_pred

  for i = 1:length(labels)
    if weights[i] != 0.0
      values[i] /= psuedo_pred[i]
    end
  end

  weighted_median(weights, values)
end
function fit_best_constant(lf::BinomialDeviance,
  labels, psuedo, psuedo_pred, prev_func_pred)

  # TODO(svs14): Add fit_best_constant (BinomialDeviance) for base learner.
  error("$(typeof(lf)) is not implemented for GBBaseLearner.")
end

end # module
