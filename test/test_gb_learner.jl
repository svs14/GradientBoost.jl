module TestGBLearner

using FactCheck
importall GradientBoost.GBLearner
importall GradientBoost.LossFunctions

type DummyLearner; end
type StubLearner; end

function GBLearner.learner_fit(lf::LossFunction, learner::StubLearner, 
  instances, labels)

  return (instance) -> mean(instance)
end

function GBLearner.learner_predict(lf::LossFunction, learner::StubLearner, 
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
      GaussianLoss(),
      dgb,
      emp_mat,
      emp_vec
    )
    @fact_throws learner_predict(
      GaussianLoss(),
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
end

end # module
