# GradientBoost

[![Build Status](https://travis-ci.org/svs14/GradientBoost.jl.svg?branch=master)](https://travis-ci.org/svs14/GradientBoost.jl)

This package covers the gradient boosting paradigm: a framework that builds
additive expansions based on any fitting criteria.

In machine learning parlance, this is typically referred to as
gradient boosting machines, generalized boosted models and stochastic gradient
boosting.

Normally, gradient boosting implementations cover a specific algorithm: gradient
boosted decision trees. The design of this library is to cover this, along with
more general gradient boosting algorithms.

This library is currently in development. ETA for the initial release is
mid-July.

References:

- <cite> Friedman, Jerome H. "Greedy function approximation: a gradient boosting
machine." Annals of Statistics (2001): 1189-1232. </cite>
- <cite> Friedman, Jerome H. "Stochastic gradient boosting." 
Computational Statistics & Data Analysis 38.4 (2002): 367-378. </cite>
- <cite> Ridgeway, Greg. "Generalized Boosted Models: A guide to the gbm package."
Update 1.1 (2007). </cite>
- <cite> Pedregosa, Fabian, et al. "Scikit-learn: Machine learning in Python." 
The Journal of Machine Learning Research 12 (2011): 2825-2830. </cite>
- <cite> Natekin, Alexey, and Alois Knoll. 
"Gradient boosting machines, a tutorial." 
Frontiers in neurorobotics 7 (2013). </cite>
