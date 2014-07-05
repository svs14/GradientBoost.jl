# Util functions.
module Util

export weighted_median,
       err_must_be_overriden

# Weighted median.
#
# @param weights Weights of values.
# @param values Values.
# @return Weighted median.
function weighted_median{U,V<:Real}(
  weights::AbstractVector{U}, values::AbstractVector{V})

  k = 1
  sorted_ind = sortperm(values)
  weight_sum = sum(weights)

  remaining_sum = weight_sum - weights[sorted_ind[k]]
  while remaining_sum > weight_sum / 2.0
    k += 1
    remaining_sum -= weights[sorted_ind[k]]
  end

  return values[sorted_ind[k]]
end

function err_must_be_overriden()
  error("This function must be overriden.")
end

end # module
