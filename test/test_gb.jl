module TestGB

using FactCheck
importall GradientBoost.GB
importall GradientBoost.LossFunctions

type DummyGradientBoost <: GradientBoost; end

type StubGradientBoost <: GradientBoost 
  loss_function::LossFunction
  sampling_rate::FloatingPoint
  learning_rate::FloatingPoint
  num_iterations::Int
end

function build_base_func(
  gb::StubGradientBoost,
  instances,
  labels,
  prev_func_pred,
  psuedo)

  function pred(instances)
    num_instances = size(instances, 1)
    predictions = Array(Float64, num_instances)
    for i = 1:num_instances
      predictions[i] = sum(instances[i,:])
    end
    predictions
  end

  model_const = 0.5
  return (instances) -> model_const .* pred(instances)
end

sgb_instances = [
  2 2;
  2 4
]
sgb_labels = [
  1.0;
  3.0
]

facts("Gradient Boost") do
  context("not implemented functions throw an error") do
    dgb = DummyGradientBoost()
    emp_mat = Array(Any, 1, 1)
    emp_vec = Array[]
    @fact_throws build_base_func(
      dgb,
      emp_mat,
      emp_vec,
      emp_vec,
      emp_vec
    )
  end

  context("stochastic_gradient_boost works") do
    # Sanity check
    sgb = StubGradientBoost(GaussianLoss(), 1.0, 0.5, 1)
    model = stochastic_gradient_boost(sgb, sgb_instances, sgb_labels)
    @fact 1 => 1
  end

  context("fit returns model") do
    sgb = StubGradientBoost(GaussianLoss(), 1.0, 0.5, 1)
    model = stochastic_gradient_boost(sgb, sgb_instances, sgb_labels)
    @fact typeof(model) <: GBModel => true
  end

  context("predict works") do
    expected = [
      3.0
      3.5
    ]
    sgb = StubGradientBoost(GaussianLoss(), 1.0, 0.5, 1)
    model = stochastic_gradient_boost(sgb, sgb_instances, sgb_labels)
    predictions = predict(model, sgb_instances)
    @fact predictions => expected
  end

  context("create_sample_indices works") do
    instances = [1:5 6:10]
    labels = [1:5]

    sgb = StubGradientBoost(GaussianLoss(), 1, 1, 1)
    indices = create_sample_indices(sgb, instances, labels)
    @fact length(indices) => 5
    @fact length(unique(indices)) => 5
    @fact minimum(indices) >= 1 => true
    @fact maximum(indices) <= 5 => true

    sgb = StubGradientBoost(GaussianLoss(), 0.5, 1, 1)
    indices = create_sample_indices(sgb, instances, labels)
    @fact length(indices) => 3
    @fact length(unique(indices)) => 3
    @fact minimum(indices) >= 1 => true
    @fact maximum(indices) <= 5 => true
  end
end

end # module
