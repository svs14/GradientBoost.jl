module TestGBBaseLearner

using Test
using GradientBoost.GBBaseLearner
using GradientBoost.LossFunctions

using Statistics

struct DummyLearner end
struct StubLearner end

function GBBaseLearner.learner_fit(lf::LossFunction, learner::StubLearner,
  instances, labels)

  return (instance) -> mean(instance)
end

function GBBaseLearner.learner_predict(lf::LossFunction, learner::StubLearner,
  model, instances)

  pred_func = model
  num_instances = size(instances, 1)

  predictions = [pred_func(instances[i,:]) for i = 1:num_instances]
end

@testset "GB Learner" begin
  @testset "not implemented functions throw an error" begin
    dl = DummyLearner()
    emp_mat = []
    emp_vec = []
    dummy_model = emp_vec

    @test_throws ErrorException learner_fit(
      LeastSquares(),
      dl,
      emp_mat,
      emp_vec
    )
    @test_throws ErrorException learner_predict(
      LeastSquares(),
      dl,
      dummy_model,
      emp_mat
    )
  end

  @testset "build_base_func works" begin
    sl = StubLearner()
    gb = GBBL(sl)
    instances = [
      1 1;
      2 4;
    ]
    labels = [
      2;
      6;
    ]

    prev_func_pred =
      fill(minimizing_scalar(gb.loss_function, labels), size(instances, 1))
    psuedo = negative_gradient(
      gb.loss_function,
      labels,
      gb.learning_rate .* prev_func_pred
    )

    base_func = build_base_func(
      gb,
      instances,
      labels,
      prev_func_pred,
      psuedo
    )

    predictions = base_func(instances)
    expected = [ 1.0, 3.0 ]
    @test predictions ≈ expected
  end

  @testset "LeastSquares fit_best_constant works" begin
    lf = LeastSquares()
    dummy_vec = [0.0,0.0,0.0,0.0]
    expected = 1.0

    actual = GBBaseLearner.fit_best_constant(
      lf, dummy_vec, dummy_vec, dummy_vec, dummy_vec
    )
    @test actual == expected
  end
  @testset "LeastAbsoluteDeviation fit_best_constant works" begin
    lf = LeastAbsoluteDeviation()
    dummy_vec = [0.0,0.0,0.0,0.0]
    labels = [0.0,1.0,2.0,3.0]
    psuedo_pred = [3.0,2.0,1.0,0.0]
    prev_func_pred = [1.0,0.0,1.0,0.0]
    expected = -0.333333

    actual = GBBaseLearner.fit_best_constant(
      lf, labels, dummy_vec, psuedo_pred, prev_func_pred
    )
    @test actual ≈ expected atol=1e-6
  end
  @testset "BinomialDeviance fit_best_constant throws error" begin
    lf = BinomialDeviance()
    dummy_vec = [0.0,0.0,0.0,0.0]

    @test_throws ErrorException GBBaseLearner.fit_best_constant(
      lf, dummy_vec, dummy_vec, dummy_vec, dummy_vec
    )
  end
end

end # module
