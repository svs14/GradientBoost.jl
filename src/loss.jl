# Loss functions.
module LossFunctions

importall GradientBoost.Util

export LossFunction,
       loss,
       negative_gradient,
       minimizing_scalar,
       fit_best_constant,
       GaussianLoss,
       LaplaceLoss,
       BernoulliLoss

# Loss function.
abstract LossFunction

# Calculates loss.
# 
# @param lf Loss function.
# @param y True response.
# @param y_pred Approximated response.
# @return Loss.
loss(lf::LossFunction, y, y_pred) = err_must_be_overriden()

# Calculates negative gradient of loss function.
#
# @param lf Loss function.
# @param y True response.
# @param y_pred Approximated response.
# @return Vector of negative gradient residuals.
negative_gradient(lf::LossFunction, y, y_pred) = err_must_be_overriden()

# Finds scalar value c that minimizes loss L(y, c).
#
# @param lf Loss function.
# @param y True response.
# @return Scalar value.
minimizing_scalar(lf::LossFunction, y) = err_must_be_overriden()

# Fit best constant to base function in gradient boosting algorithm.
#
# @param lf Loss function.
# @param labels Labels (response).
# @param psuedo Psuedo-labels (psuedo-response).
# @param psuedo_pred Current predictions on psuedo_labels.
# @param prev_func_pred Previous base function's predictions.
# @return Constant.
fit_best_constant(lf::LossFunction,
  labels, psuedo, psuedo_pred, prev_func_pred) = err_must_be_overriden()


# Gaussian (Least Squares)
type GaussianLoss <: LossFunction; end

function loss(lf::GaussianLoss, y, y_pred)
  mean((y .- y_pred) .^ 2.0)
end

function negative_gradient(lf::GaussianLoss, y, y_pred)
  y .- y_pred
end

function minimizing_scalar(lf::GaussianLoss, y)
  mean(y)
end

function fit_best_constant(lf::GaussianLoss,
  labels, psuedo, psuedo_pred, prev_func_pred)

  # No refitting required
  1.0
end


# Laplace (Least Absolute Deviation)
type LaplaceLoss<: LossFunction; end

function loss(lf::LaplaceLoss, y, y_pred)
  mean(abs(y .- y_pred))
end

function negative_gradient(lf::LaplaceLoss, y, y_pred)
  sign(y .- y_pred)
end

function minimizing_scalar(lf::LaplaceLoss, y)
  median(y)
end

function fit_best_constant(lf::LaplaceLoss,
  labels, psuedo, psuedo_pred, prev_func_pred)

  weights = abs(psuedo_pred)
  values = labels .- prev_func_pred

  for i = 1:size(labels, 1)
    if weights[i] != 0.0
      values[i] /= psuedo_pred[i]
    end
  end

  weighted_median(weights, values)
end

# Bernoulli Loss (Two Classess {0,1})
type BernoulliLoss <: LossFunction; end

function loss(lf::BernoulliLoss, y, y_pred)
  -2.0 .* mean(y .* y_pred .- log(1.0 .+ exp(y_pred)))
end

function negative_gradient(lf::BernoulliLoss, y, y_pred)
  y .- 1.0 ./ (1.0 .+ exp(-y_pred))
end

function minimizing_scalar(lf::BernoulliLoss, y)
  y_sum = sum(y)
  y_length = length(y)
  log(y_sum / (y_length - y_sum))
end

function fit_best_constant(lf::BernoulliLoss,
  labels, psuedo, psuedo_pred, prev_func_pred)

  # TODO(svs14): Verify this works fine for base learner algorithm.
  num = sum(psuedo)
  denom = sum((labels .- psuedo) .* (1 .- labels .+ psuedo))
  if denom == 0.0
    return 0.0
  else
    return num / denom
  end
end

end # module
