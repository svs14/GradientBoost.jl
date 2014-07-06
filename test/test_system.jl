# System tests.
module TestSystem

using FactCheck
importall GradientBoost.Util
importall GradientBoost.ML

# Experiment on GBProblem
#
# gbp_func is a function that returns an instantiated GBProblem.
# score_func is a function that takes predictions, actual and returns as score.
function experiment(gbp_func, score_func, num_experiments, instances, labels)
  scores = Array(Float64, num_experiments)

  for i = 1:num_experiments
    # Obtain training and test set
    (train_ind, test_ind) = holdout(size(instances, 1), 0.3)
    train_instances = instances[train_ind, :]
    test_instances = instances[test_ind, :]
    train_labels = labels[train_ind]
    test_labels = labels[test_ind]

    # Train
    gbp = gbp_func()
    fit!(gbp, train_instances, train_labels)

    # Test
    predictions = predict!(gbp, test_instances)
    score = score_func(predictions, test_labels)
    scores[i] = score
  end

  scores
end

facts("System tests") do
  context("iris dataset is handled by GBDT") do
    # Get data
    dataset = readcsv(joinpath(dirname(@__FILE__), "iris.csv"))
    instances = dataset[:, 1:(end-1)]
    labels = dataset[:, end]

    # Convert data to required format.
    instances = convert(Matrix{Float64}, instances)
    labels = [species == "setosa" ? 1.0 : 0.0 for species in labels]

    # Train and test multiple times
    num_experiments = 5
    function gbp_func()
      gbdt = GBDT(
        BernoulliLoss(),
        0.6,
        0.1,
        100
      )
      gbp = GBProblem(gbdt, :class)
    end
    function score_func(predictions, actual)
      mean(predictions .== actual) * 100.0
    end
    scores = experiment(
      gbp_func, score_func, num_experiments, instances, labels
    )

    # Sanity check, accuracy should be greater or equal to baseline
    prop_ones = sum(labels) / length(labels)
    baseline = max(prop_ones, 1 - prop_ones) * 100.0
    @fact mean(scores) >= baseline => true
  end

  context("mtcars dataset is handled by GBDT") do
    # Get data
    dataset = readcsv(joinpath(dirname(@__FILE__), "mtcars.csv"))
    instances = dataset[:, 2:end]
    labels = dataset[:, 1]

    # Convert data to required format.
    instances = convert(Matrix{Float64}, instances)
    labels = convert(Vector{Float64}, labels)

    # Train and test multiple times (MSE)
    num_experiments = 5
    function gbp_mse_func()
      gbdt = GBDT(
        GaussianLoss(),
        0.6,
        0.1,
        100
      )
      gbp = GBProblem(gbdt, :regression)
    end
    function mse(predictions, actual)
      mean((actual .- predictions) .^ 2.0)
    end
    scores = experiment(
      gbp_mse_func, mse, num_experiments, instances, labels
    )

    # Sanity check, MSE should be smaller or equal to baseline
    label_mean = mean(labels)
    baseline = mse(label_mean, labels)
    @fact mean(scores) <= baseline => true

    # Train and test multiple times (MAD)
    num_experiments = 5
    function gbp_mad_func()
      gbdt = GBDT(
        LaplaceLoss(),
        0.6,
        0.1,
        100
      )
      gbp = GBProblem(gbdt, :regression)
    end
    function mad(predictions, actual)
      mean(abs(actual .- predictions))
    end
    scores = experiment(
      gbp_mad_func, mad, num_experiments, instances, labels
    )

    # Sanity check, MAD should be smaller or equal to baseline
    label_median = median(labels)
    baseline = mad(label_median, labels)

    @fact mean(scores) <= baseline => true
  end
end

end # module
