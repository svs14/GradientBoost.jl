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
  context("not implemented functions throw an error") do
    gbl = GBLearner(GBDT(), :regression)
    instances = 1
    labels = 1

    @fact_throws fit!(gbl, instances, labels)
    @fact_throws predict!(gbl, instances)
  end

  context("fit! on Float64 arrays works") do
    gbl = GBLearner(GBDT(), :regression)
    @fact gbl.model => nothing
    fit!(gbl, instances, labels)
    @fact gbl.model => not(nothing)
  end

  context("predict! on Float64 arrays works") do
    gbl = GBLearner(GBDT(;loss_function=BinomialDeviance()), :class)
    fit!(gbl, instances, labels)
    predictions = predict!(gbl, instances)
    @fact eltype(predictions) => Float64
  end

  context("postprocess_pred works") do
    predictions = [-Inf, 0.0, Inf]
    expected = [0.0, 1.0, 1.0]
    actual = ML.postprocess_pred(:class, BinomialDeviance(), predictions)
    @fact actual => expected

    predictions = [-Inf, 0.0, Inf]
    actual = ML.postprocess_pred(:class_prob, BinomialDeviance(), predictions)
    @fact all(i -> (0 <= i <= 1), actual) => true

    predictions = [-Inf, 0.0, Inf]
    expected = predictions
    actual = ML.postprocess_pred(:regression, LeastSquares(), predictions)
    @fact actual => expected

    predictions = [-Inf, 0.0, Inf]
    @fact_throws ML.postprocess_pred(:class, LeastSquares(), predictions)
  end

  context("logistic works") do
    x = [-Inf, -1, 0, 1, Inf]
    expected = [0.0, 0.0, 1.0, 1.0, 1.0]

    actual = round(ML.logistic(x))
    @fact actual => expected
  end
end

end # module
