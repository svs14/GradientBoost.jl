module TestGBBaseLearner

using FactCheck
importall GradientBoost.GBBaseLearner
importall GradientBoost.LossFunctions

type DummyLearner; end
type StubLearner; end

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

facts("GB Learner") do
  context("not implemented functions throw an error") do
    dl = DummyLearner()
    emp_mat = Array(Any, 1, 1)
    emp_vec = Array[]
    dummy_model = emp_vec

    @fact_throws learner_fit(
      LeastSquares(),
      dgb,
      emp_mat,
      emp_vec
    )
    @fact_throws learner_predict(
      LeastSquares(),
      dgb,
      dummy_model,
      emp_mat
    )
  end

  context("build_base_func works") do
    sl = StubLearner()
    gb = GBL(sl)
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
    expected = { 1.0, 3.0 }
    @fact predictions => roughly(expected)
  end

  context("LeastSquares fit_best_constant works") do
    lf = LeastSquares()
    dummy_vec = [0.0,0.0,0.0,0.0]
    expected = 1.0

    actual = GBBaseLearner.fit_best_constant(
      lf, dummy_vec, dummy_vec, dummy_vec, dummy_vec
    )
    @fact actual => expected
  end
  context("LeastAbsoluteDeviation fit_best_constant works") do
    lf = LeastAbsoluteDeviation()
    dummy_vec = [0.0,0.0,0.0,0.0]
    labels = [0.0,1.0,2.0,3.0]
    psuedo_pred = [3.0,2.0,1.0,0.0]
    prev_func_pred = [1.0,0.0,1.0,0.0]
    expected = -0.333333

    actual = GBBaseLearner.fit_best_constant(
      lf, labels, dummy_vec, psuedo_pred, prev_func_pred
    )
    @fact actual => roughly(expected)
  end
  context("BinomialDeviance fit_best_constant throws error") do
    lf = BinomialDeviance()
    dummy_vec = [0.0,0.0,0.0,0.0]

    @fact_throws GBBaseLearner.fit_best_constant(
      lf, dummy_vec, dummy_vec, dummy_vec, dummy_vec
    )
  end
end

end # module
