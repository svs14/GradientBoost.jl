# Loss functions.
module LossFunctions

using Statistics
using GradientBoost.Util

export LossFunction,
       loss,
       negative_gradient,
       minimizing_scalar,
       LeastSquares,
       LeastAbsoluteDeviation,
       BinomialDeviance

# Loss function.
abstract type LossFunction end

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

# LeastSquares
struct LeastSquares <: LossFunction end

function loss(lf::LeastSquares, y, y_pred)
  mean((y .- y_pred) .^ 2.0)
end

function negative_gradient(lf::LeastSquares, y, y_pred)
  y .- y_pred
end

function minimizing_scalar(lf::LeastSquares, y)
  mean(y)
end


# LeastAbsoluteDeviation
struct LeastAbsoluteDeviation <: LossFunction end

function loss(lf::LeastAbsoluteDeviation, y, y_pred)
  mean(abs.(y .- y_pred))
end

function negative_gradient(lf::LeastAbsoluteDeviation, y, y_pred)
  sign.(y .- y_pred)
end

function minimizing_scalar(lf::LeastAbsoluteDeviation, y)
  median(y)
end


# Binomial Deviance (Two Classes {0,1})
struct BinomialDeviance <: LossFunction end

"""
    loss(lf, y, y_pred)

Calculate ``L(y, p(x)) = -\\sum_{k=1}^K I(y=G_k)\\log p_k(x)``, where
``p_k(x) = \\exp(f_k(x)) / \\sum_{i=1}^K \\exp(f_i(x))``.
"""
function loss(lf::BinomialDeviance, y, y_pred)
  -2.0 .* mean(y .* y_pred .- log.(1.0 .+ exp.(y_pred)))
end

function negative_gradient(lf::BinomialDeviance, y, y_pred)
  y .- 1.0 ./ (1.0 .+ exp.(-y_pred))
end

function minimizing_scalar(lf::BinomialDeviance, y)
  y_sum = sum(y)
  y_length = length(y)
  log.(y_sum / (y_length - y_sum))
end

end # module
