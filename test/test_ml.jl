module TestML

using Test
using GradientBoost.ML
using GradientBoost.LossFunctions

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

@testset "Machine Learning API" begin
  @testset "not implemented functions throw an error" begin
    gbl = GBLearner(GBDT(), :regression)
    instances = 1
    labels = 1

    @test_throws ErrorException fit!(gbl, instances, labels)
    @test_throws ErrorException predict!(gbl, instances)
  end

  @testset "fit! on Float64 arrays works" begin
    gbl = GBLearner(GBDT(), :regression)
    @test gbl.model == nothing
    fit!(gbl, instances, labels)
    @test gbl.model != nothing
  end

  @testset "predict! on Float64 arrays works" begin
    gbl = GBLearner(GBDT(;loss_function=BinomialDeviance()), :class)
    fit!(gbl, instances, labels)
    predictions = predict!(gbl, instances)
    @test eltype(predictions) == Float64
  end

  @testset "postprocess_pred works" begin
    predictions = [-Inf, 0.0, Inf]
    expected = [0.0, 1.0, 1.0]
    actual = ML.postprocess_pred(:class, BinomialDeviance(), predictions)
    @test actual == expected

    predictions = [-Inf, 0.0, Inf]
    actual = ML.postprocess_pred(:class_prob, BinomialDeviance(), predictions)
    @test all(i -> (0 <= i <= 1), actual)

    predictions = [-Inf, 0.0, Inf]
    expected = predictions
    actual = ML.postprocess_pred(:regression, LeastSquares(), predictions)
    @test actual == expected

    predictions = [-Inf, 0.0, Inf]
    @test_throws ErrorException ML.postprocess_pred(:class, LeastSquares(), predictions)
  end

  @testset "logistic works" begin
    x = [-Inf, -1, 0, 1, Inf]
    expected = [0.0, 0.0, 1.0, 1.0, 1.0]

    actual = round.(ML.logistic(x), RoundNearestTiesAway)
    @test actual == expected
  end
end

end # module
