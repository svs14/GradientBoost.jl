module TestUtil

using FactCheck
importall GradientBoost.Util

facts("Util functions") do
  context("err_must_be_overriden throws an error") do
    @fact_throws err_must_be_overriden()
  end

  context("weighted_median works") do
    weights = [1, 1, 1, 1, 1]
    values = [1, 2, 3, 4, 5]
    @fact weighted_median(weights, values) => 3

    weights = [1, 1, 1, 1, 1]
    values = [5, 4, 3, 2, 1]
    @fact weighted_median(weights, values) => 3

    weights = [1, 1, 1, 1, 1]
    values = [4, 5, 2, 3, 1]
    @fact weighted_median(weights, values) => 3 

    weights = [1, 3, 1, 1, 1]
    values = [1, 2, 3, 4, 5]
    @fact weighted_median(weights, values) => 2 

    weights = [1, 1, 1, 1, 100]
    values = [1, 2, 3, 4, 5]
    @fact weighted_median(weights, values) => 5

    weights = [100, 1, 1, 1, 1]
    values = [1, 2, 3, 4, 5]
    @fact weighted_median(weights, values) => 1

    weights = [0.5, 0.5, 1, 1, 1]
    values = [1, 2, 3, 4, 5]
    @fact weighted_median(weights, values) => 3
  end

  context("holdout returns proportional partitions") do
    n = 10
    right_prop = 0.3
    (left, right) = holdout(n, right_prop)

    @fact size(left, 1) => n - (n * right_prop)
    @fact size(right, 1) => n * right_prop
    @fact intersect(left, right) => isempty
    @fact size(union(left, right), 1) => n
  end
end

end # module
