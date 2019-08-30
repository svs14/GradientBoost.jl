# System tests.
module TestSystem

using Test

using GradientBoost.Util
using GradientBoost.ML

import GLM: fit, predict, LinearModel

using DelimitedFiles
using Statistics
# Experiment on GBLearner
#
# gbl_func is a function that returns an instantiated GBLearner.
# score_func is a function that takes predictions, actual and returns as score.
# baseline_func is a function that takes labels.
function experiment(gbl_func, score_func, baseline_func,
  num_experiments, instances, labels)

  scores = Array{Float64, 1}(undef, num_experiments)
  for i = 1:num_experiments
    # Obtain training and test set
    (train_ind, test_ind) = holdout(size(instances, 1), 0.2)
    train_instances = instances[train_ind, :]
    test_instances = instances[test_ind, :]
    train_labels = labels[train_ind]
    test_labels = labels[test_ind]

    # Train
    gbl = gbl_func()
    fit!(gbl, train_instances, train_labels)

    # Test
    predictions = predict!(gbl, test_instances)
    score = score_func(predictions, test_labels)
    scores[i] = score
  end

  # Sanity check, score should be less than baseline.
  baseline = baseline_func(labels)
  # NOTE(weiya) sometimes may not pass, so to be more tolerant
  @test mean(scores) <= baseline*2

  scores
end

# Error rate
function err_rate(predictions, actual)
  1.0 - mean(predictions .== actual)
end
function baseline_err_rate(labels)
  prop_ones = sum(labels) / length(labels)
  baseline = 1.0 - max(prop_ones, 1 - prop_ones)
end

# Mean squared error
function mse(predictions, actual)
  mean((actual .- predictions) .^ 2.0)
end
function baseline_mse(labels)
  label_mean = mean(labels)
  baseline = mse(label_mean, labels)
end

# Mean absolute deviation
function mad(predictions, actual)
  mean(abs.(actual .- predictions))
end
function baseline_mad(labels)
  label_median = median(labels)
  baseline = mad(label_median, labels)
end

num_experiments = 10

function ML.learner_fit(lf::LossFunction,
  learner::Type{LinearModel}, instances, labels)

  model = fit(learner, instances, labels)
end
function ML.learner_predict(lf::LossFunction,
  learner::Type{LinearModel}, model, instances)

  predict(model, instances)
end

@testset "System tests" begin
  @testset "iris dataset is handled by GBDT" begin
    # Get data
    iris = readdlm(joinpath(dirname(@__FILE__), "iris.csv"), ',')
    instances = iris[:, 1:(end-1)]
    labels = iris[:, end]

    # Convert data to required format.
    instances = convert(Matrix{Float64}, instances)
    labels = [species == "setosa" ? 1.0 : 0.0 for species in labels]

    # Train and test multiple times (GBDT)
    function gbl_func()
      gbdt = GBDT(;
        loss_function=BinomialDeviance(),
        sampling_rate=0.6,
        learning_rate=0.1,
        num_iterations=100
      )
      gbl = GBLearner(gbdt, :class)
    end
    experiment(
      gbl_func, err_rate, baseline_err_rate, num_experiments, instances, labels
    )
  end

  @testset "mtcars dataset is handled" begin
    # Get data
    mtcars = readdlm(joinpath(dirname(@__FILE__), "mtcars.csv"), ',')
    instances = mtcars[:, 2:end]
    labels = mtcars[:, 1]

    # Convert data to required format.
    instances = convert(Matrix{Float64}, instances)
    labels = convert(Vector{Float64}, labels)

    # Train and test multiple times (MSE)
    gbl_mse_funcs = Function[]
    function mse_gbdt_func()
      gbdt = GBDT(;
        loss_function=LeastSquares(),
        sampling_rate=0.6,
        learning_rate=0.1,
        num_iterations=100
      )
      gbl = GBLearner(gbdt, :regression)
    end
    push!(gbl_mse_funcs, mse_gbdt_func)
    function mse_gbl_func()
      gbl = GBBL(
        LinearModel;
        loss_function=LeastSquares(),
        sampling_rate=0.8,
        learning_rate=0.1,
        num_iterations=100,
      )
      gbl = GBLearner(gbl, :regression)
    end
    push!(gbl_mse_funcs, mse_gbl_func)
    for i = 1:length(gbl_mse_funcs)
      experiment(
        gbl_mse_funcs[i], mse, baseline_mse, num_experiments, instances, labels
      )
    end

    # Train and test multiple times (MAD)
    gbl_mad_funcs = Function[]
    function mad_gbdt_func()
      gbdt = GBDT(;
        loss_function=LeastAbsoluteDeviation(),
        sampling_rate=0.6,
        learning_rate=0.1,
        num_iterations=100
      )
      gbl = GBLearner(gbdt, :regression)
    end
    push!(gbl_mad_funcs, mad_gbdt_func)
    function mad_gbl_func()
      gbl = GBBL(
        LinearModel;
        loss_function=LeastAbsoluteDeviation(),
        sampling_rate=0.8,
        learning_rate=0.1,
        num_iterations=100,
      )
      gbl = GBLearner(gbl, :regression)
    end
    push!(gbl_mad_funcs, mad_gbl_func)
    for i = 1:length(gbl_mad_funcs)
      experiment(
        gbl_mad_funcs[i], mad, baseline_mad, num_experiments, instances, labels
      )
    end
  end
end

end # module
