# GradientBoost

[![Build Status](https://travis-ci.org/svs14/GradientBoost.jl.svg?branch=master)](https://travis-ci.org/svs14/GradientBoost.jl)
[![Coverage Status](https://coveralls.io/repos/svs14/GradientBoost.jl/badge.png?branch=master)](https://coveralls.io/r/svs14/GradientBoost.jl?branch=master)

This package covers the gradient boosting paradigm: a framework that builds
additive expansions based on any fitting criteria.

In machine learning parlance, this is typically referred to as
gradient boosting machines, generalized boosted models and stochastic gradient
boosting.

Normally, gradient boosting implementations cover a specific algorithm: gradient
boosted decision trees. This package covers the framework itself, including such
implementations.

References:

- <cite> Friedman, Jerome H. "Greedy function approximation: a gradient boosting
machine." Annals of Statistics (2001): 1189-1232. </cite>
- <cite> Friedman, Jerome H. "Stochastic gradient boosting." 
Computational Statistics & Data Analysis 38.4 (2002): 367-378. </cite>
- <cite> Hastie, Trevor, et al. The elements of statistical learning.
Vol. 2. No. 1. New York: Springer, 2009. </cite>
- <cite> Ridgeway, Greg. "Generalized Boosted Models: A guide to the gbm package."
Update 1.1 (2007). </cite>
- <cite> Pedregosa, Fabian, et al. "Scikit-learn: Machine learning in Python." 
The Journal of Machine Learning Research 12 (2011): 2825-2830. </cite>
- <cite> Natekin, Alexey, and Alois Knoll. 
"Gradient boosting machines, a tutorial." 
Frontiers in neurorobotics 7 (2013). </cite>

## Machine Learning API

Module `GradientBoost.ML` is provided for users who are only interested in 
using existing gradient boosting algorithms for prediction. 
To get a feel for the API, 
we will run a demonstration 
of gradient boosted decision trees on the iris dataset.

### Obtain Data

At the moment only two-class classification is handled, 
so our learner will attempt to separate "setosa" from the other species.
```julia
using GradientBoost.ML
using RDatasets

# Obtain iris dataset
iris = dataset("datasets", "iris")
instances = array(iris[:, 1:end-1])
labels = [species == "setosa" ? 1.0 : 0.0 for species in array(iris[:, end])]

# Obtain training and test set (20% test)
num_instances = size(instances, 1)
train_ind, test_ind = GradientBoost.Util.holdout(num_instances, 0.2)
```

### Build Learner

The gradient boosting (GB) learner comprises of a GB algorithm 
and what output it must produce. 
In this case, we shall assign a gradient boosted decision tree to output classes.
```julia
# Build GBLearner
gbdt = GBDT(;
  loss_function=BinomialDeviance(),
  sampling_rate=0.6,
  learning_rate=0.1,
  num_iterations=100
)
gbl = GBLearner(
  gbdt,  # Gradient boosting algorithm
  :class # Output (:class, :class_prob, :regression)
)
```

### Train and Predict

Currently `Matrix{Float64}` instances and `Vector{Float64}` labels are 
the only handled types for training and prediction. 
In this case, it is not an issue.

```julia
# Train
ML.fit!(gbl, instances[train_ind, :], labels[train_ind])

# Predict
predictions = ML.predict!(gbl, instances[test_ind, :])
```

### Evaluate

If all is well, we should obtain better than baseline accuracy (67%).
```julia
# Obtain accuracy
accuracy = mean(predictions .== labels[test_ind]) * 100.0
println("GBDT accuracy: $(accuracy)")
```

That concludes the demonstration. Detailed below are the available GB learners.

## Algorithms

Documented below are the currently implemented gradient boosting algorithms.

### GB Decision Tree

Gradient Boosted Decision Tree algorithm backed by 
[DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl#regression-example) 
regression trees. 
Current loss functions covered are: 
`LeastSquares`, `LeastAbsoluteDeviation` and `BinomialDeviance`.

```julia
gbdt = GBDT(;
  loss_function=BinomialDeviance(), # Loss function
  sampling_rate=0.6,                # Sampling rate
  learning_rate=0.1,                # Learning rate
  num_iterations=100,               # Number of iterations
  tree_options={                    # Tree options (DecisionTree.jl regressor)
    :maxlabels => 5,
    :nsubfeatures => 0
  }
)
```

### GB Base Learner

Gradient boosting with a given base learner. 
Current loss functions covered are: `LeastSquares` and `LeastAbsoluteDeviation`. 
In order to use this, 
`ML.learner_fit` and `ML.learner_predict` functions must be extended.
Example provided below for linear regression found in 
[GLM.jl](https://github.com/JuliaStats/GLM.jl).
```julia
import GLM: fit, predict, LinearModel

# Extend functions
function ML.learner_fit(lf::LossFunction, 
  learner::Type{LinearModel}, instances, labels)
  
  model = fit(learner, instances, labels)
end
function ML.learner_predict(lf::LossFunction,
  learner::Type{LinearModel}, model, instances)
  
  predict(model, instances)
end
```

Once this is done, 
the algorithm can be instantiated with the respective base learner.
```julia
gbl = GBBL(
  LinearModel;                  # Base Learner
  loss_function=LeastSquares(), # Loss function
  sampling_rate=0.8,            # Sampling rate
  learning_rate=0.1,            # Learning rate
  num_iterations=100            # Number of iterations
)
gbl = GBLearner(gbl, :regression)
```

## Gradient Boosting Framework

All previously developed algorithms follow the framework 
provided by `GradientBoost.GB`. 
As this package is in its preliminary stage, 
major changes may occur in the near future and as such 
we provide minimal README documentation.

The algorithm must be of type `GradientBoost`, with fields 
`loss_function`,`learning_rate`, `sampling_rate` and `num_iterations` accessible. 
The bare minimum an algorithm must implement is 
`build_base_func`. Optionally, `create_sample_indices` can be extended. 
Loss functions can be found in `GradientBoost.LossFunctions`.

A relatively light algorithm 
that implements this is `GBBL`, found in `src/gb_bl.jl`.

## Misc

The links provided below will only work if you are viewing this in the GitHub repository.

### Changes

See [CHANGELOG.yml](CHANGELOG.yml).

### Future Work

See [FUTUREWORK.md](FUTUREWORK.md).

### Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

### License

MIT "Expat" License. See [LICENSE.md](LICENSE.md).
