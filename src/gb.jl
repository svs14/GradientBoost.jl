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
  # Initialize base functions collection
  num_iterations = gb.num_iterations
  base_funcs = Array(Function, num_iterations+1)

  # Create initial base function
  initial_val = minimizing_scalar(gb.loss_function, labels)
  initial_base_func = (instances) -> fill(initial_val, size(instances, 1))

  # Add initial base function to ensemble
  base_funcs[1] = initial_base_func

  # Build consecutive base functions
  prev_func_pred = initial_base_func(instances)
  for iter_ind = 2:num_iterations+1
    # Obtain current residuals
    psuedo = negative_gradient(
      gb.loss_function,
      labels,
      prev_func_pred
    )

    # Sample instances
    stage_sample_ind = create_sample_indices(gb, instances, labels)

    # Build current base function
    stage_base_func = build_base_func(
      gb,
      instances[stage_sample_ind, :],
      labels[stage_sample_ind],
      prev_func_pred[stage_sample_ind],
      psuedo[stage_sample_ind]
    )

    # Update previous function prediction
    prev_func_pred .+= gb.learning_rate .* stage_base_func(instances)

    # Add optimal base function to ensemble
    base_funcs[iter_ind] = stage_base_func
  end

  # Return model
  return GBModel(gb.learning_rate, base_funcs)
end

function fit(gb::GradientBoost, instances, labels)
  stochastic_gradient_boost(gb, instances, labels)
end
function predict(gb_model::GBModel, instances)
  outputs = gb_model.base_funcs[1](instances)
  for i = 2:length(gb_model.base_funcs)
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
