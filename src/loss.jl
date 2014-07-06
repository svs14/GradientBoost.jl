# Loss functions.
module LossFunctions

importall GradientBoost.Util

export LossFunction,
       loss,
       negative_gradient,
       minimizing_scalar,
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


# Laplace (Least Absolute Deviation)
type LaplaceLoss <: LossFunction; end

function loss(lf::LaplaceLoss, y, y_pred)
  mean(abs(y .- y_pred))
end

function negative_gradient(lf::LaplaceLoss, y, y_pred)
  sign(y .- y_pred)
end

function minimizing_scalar(lf::LaplaceLoss, y)
  median(y)
end


# Bernoulli Loss (Two Classes {0,1})
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

end # module
