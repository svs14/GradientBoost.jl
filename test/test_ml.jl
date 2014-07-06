module TestML

using FactCheck
importall GradientBoost.ML
importall GradientBoost.LossFunctions

instances = [
  1.0 1.0;
  1.0 8.0;
  1.0 10.0
]
labels = [
  0.0;
  1.0;
  1.0;
]

facts("Machine Learning API") do
  context("fit! on Float64 arrays works") do
    gbp = GBProblem(GBDT(), :regression)
    @fact gbp.model => nothing
    fit!(gbp, instances, labels)
    @fact gbp.model => not(nothing)
  end
  context("predict! on Float64 arrays works") do
    gbp = GBProblem(GBDT(BernoulliLoss()), :class)
    fit!(gbp, instances, labels)
    predictions = predict!(gbp, instances)
    @fact eltype(predictions) => Float64
  end

  context("postprocess_pred works") do
    predictions = [-Inf, 0.0, Inf]
    expected = [0.0, 1.0, 1.0]
    actual = ML.postprocess_pred(:class, BernoulliLoss(), predictions)
    @fact actual => expected

    predictions = [-Inf, 0.0, Inf]
    actual = ML.postprocess_pred(:class_prob, BernoulliLoss(), predictions)
    @fact all(i -> (0 <= i <= 1), actual) => true

    predictions = [-Inf, 0.0, Inf]
    expected = predictions
    actual = ML.postprocess_pred(:regression, GaussianLoss(), predictions)
    @fact actual => expected

    predictions = [-Inf, 0.0, Inf]
    @fact_throws ML.postprocess_pred(:class, GaussianLoss(), predictions)
  end

  context("logistic works") do
    x = [-Inf, -1, 0, 1, Inf]
    expected = [0.0, 0.0, 1.0, 1.0, 1.0]

    actual = round(ML.logistic(x))
    @fact actual => expected
  end
end

end # module
