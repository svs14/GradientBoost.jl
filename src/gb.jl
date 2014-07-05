# Gradient boosting.
module GB

importall GradientBoost.Util
importall GradientBoost.LossFunctions

export GradientBoost,
       GBModel,
       stochastic_gradient_boost,
       fit,
       predict,
       build_base_func,
       create_sample_indices


# Gradient boost algorithm.
abstract GradientBoost

# Gradient boost model.
type GBModel
  learning_rate::FloatingPoint
  base_funcs::Vector{Function}
end

# Perform stochastic gradient boost.
#
# @param gb Gradient boosting algorithm.
# @param instances Instances.
# @param labels Labels.
# @return Gradient boost model.
function stochastic_gradient_boost(gb::GradientBoost, instances, labels)
  # Initialize base functions and psuedo labels
  num_iterations = gb.num_iterations
  base_funcs = Array(Function, num_iterations+1)
  base_funcs[1] = (instances) ->
    fill(minimizing_scalar(gb.loss_function, labels), size(instances, 1))

  # Build base functions
  stage_base_func = base_funcs[1]
  psuedo = labels
  for iter_ind = 2:num_iterations+1
    # Update residuals
    prev_func_pred = stage_base_func(instances)
    psuedo = negative_gradient(
      gb.loss_function,
      psuedo,
      gb.learning_rate .* prev_func_pred
    )

    # Sample instances
    stage_sample_ind = create_sample_indices(gb, instances, labels)

    # Add optimal base function to ensemble
    stage_base_func = build_base_func(
      gb,
      instances[stage_sample_ind],
      labels[stage_sample_ind],
      prev_func_pred[stage_sample_ind],
      psuedo[stage_sample_ind]
    )
    base_funcs[iter_ind] = stage_base_func
  end

  # Return model
  return GBModel(gb.learning_rate, base_funcs)
end

function fit(gb::GradientBoost, instances, labels)
  stochastic_gradient_boost(gb, instances, labels)
end
function predict(gb_model::GBModel, instances)
  outputs = zeros(size(instances, 1))
  for i = 1:length(gb_model.base_funcs)
    outputs .+= gb_model.learning_rate .* gb_model.base_funcs[i](instances)
  end
  return outputs
end

# Build base (basis) function for gradient boosting algorithm.
#
# @param gb Gradient boosting algorithm.
# @param instances Instances.
# @param labels Labels.
# @param prev_func_pred Previous base function's predictions.
# @param psuedo Psuedo-labels (psuedo-response).
# @return Function of form (instances) -> predictions.
function build_base_func(
  gb::GradientBoost,
  instances,
  labels,
  prev_func_pred,
  psuedo)

  err_must_be_overriden()
end

# Default sample method for gradient boosting algorithms.
# By default, it is sampling without replacement.
#
# @param gb Gradient boosting algorithm.
# @param instances Instances.
# @param labels Labels.
# @return Sample indices.
function create_sample_indices(gb::GradientBoost, instances, labels)
  n = size(instances, 1)
  prop = gb.sampling_rate

  ind = randperm(n)[1:int(prop * n)]
end

end # module
