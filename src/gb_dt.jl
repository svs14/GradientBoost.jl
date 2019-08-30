# Gradient Boosted Decision Trees
module GBDecisionTree

using DecisionTree
using DataStructures

using GradientBoost.GB
using GradientBoost.LossFunctions

using Statistics

export GBDT,
       build_base_func

# Gradient boosted decision tree algorithm.
mutable struct GBDT{F <: AbstractFloat, D <: AbstractDict} <: GBAlgorithm
  loss_function::LossFunction
  sampling_rate::F
  learning_rate::F
  num_iterations::Int
  tree_options::D

  function GBDT(;loss_function=LeastSquares(),
    sampling_rate=0.6, learning_rate=0.1,
    num_iterations=100, tree_options=Dict())

    default_options = Dict(
      :maxdepth => 5,
      :nsubfeatures => 0
    )
    options = merge(default_options, tree_options)
    # refer to https://discourse.julialang.org/t/how-to-resolve-syntax-too-few-type-parameters-specified-in-new/15766
    new{typeof(sampling_rate), typeof(options)}(loss_function, sampling_rate, learning_rate, num_iterations, options)
  end
end

function GB.build_base_func(
  gb::GBDT,
  instances,
  labels,
  prev_func_pred,
  psuedo)

  # Train learner
  model = build_tree(
    psuedo, instances,
    # refer to https://github.com/bensadeghi/DecisionTree.jl
    gb.tree_options[:nsubfeatures],
    gb.tree_options[:maxdepth]
  )
  psuedo_pred = apply_tree(model, instances)

  # Update regions (leaves)
  # NOTE(svs14): Trees are immutable,
  #              override leaves by having node-to-val mapping.
  inst_node_index = InstanceNodeIndex(model, instances)
  function val_func(node)
    inst_ind = inst_node_index.n2i[node]

    # If loss function is LeastSquares, we don't need need to change values.
    if typeof(gb.loss_function) <: LeastSquares
      val = node.majority
    else
      val = fit_best_constant(gb.loss_function,
        labels[inst_ind],
        psuedo[inst_ind],
        psuedo_pred[inst_ind],
        prev_func_pred[inst_ind]
      )
    end

    val
  end
  val_type = eltype(prev_func_pred)
  n2v = Dict{Leaf, val_type}()
  update_regions!(n2v, model, val_func)

  # Prediction function
  function pred(instances)
    num_instances = size(instances, 1)
    predictions = [
      n2v[instance_to_node(model, instances[i,:])]
      for i in 1:num_instances
    ]
    predictions
  end

  # Produce function that delegates prediction to model
  return (instances) -> pred(instances)
end

# DT Helper Functions

mutable struct InstanceNodeIndex
  i2n::Vector{Leaf}
  n2i::DefaultDict{Leaf, Vector{Int}}

  function InstanceNodeIndex(tree::Union{Leaf,Node}, instances)
    num_instances = size(instances, 1)
    i2n = Array{Leaf, 1}(undef, num_instances)
    n2i = DefaultDict{Leaf, Vector{Int}}(Int[])

    for i = 1:num_instances
      node = instance_to_node(tree, instances[i,:])
      i2n[i] = node
      push!(n2i[node], i)
    end

    new(i2n, n2i)
  end
end

# Returns respective node of instance.
function instance_to_node(tree::Node, instance)
  # Code adapted from DecisionTree.jl
  features = instance
  if tree.featval == nothing || features[tree.featid] < tree.featval
    return instance_to_node(tree.left, features)
  else
    return instance_to_node(tree.right, features)
  end
end
function instance_to_node(leaf::Leaf, instance)
  return leaf
end

# Update region by having updated leaf value encoded
# in a leaf-to-value mapping.
function update_regions!(n2v::Dict{Leaf, T}, node::Node, val_func::Function) where T
  update_regions!(n2v, node.left, val_func)
  update_regions!(n2v, node.right, val_func)
end
function update_regions!(n2v::Dict{Leaf, T}, leaf::Leaf, val_func::Function) where T
  n2v[leaf] = val_func(leaf)
end

# Loss function fits
function fit_best_constant(lf::LeastAbsoluteDeviation,
  labels, psuedo, psuedo_pred, prev_func_pred)

  values = labels .- prev_func_pred
  median(values)
end
"""
    fit_best_constant(lf::BinomialDeviance, labels, psuedo, psuedo_pred, prev_func_pred)

Solve ``\\arg\\min -( y(\\hat y+ γ) - \\log (1+\\exp(\\hat y+γ)))``, since no closed form, approximate
it by a single Newton-Raphson step.
"""
function fit_best_constant(lf::BinomialDeviance,
  labels, psuedo, psuedo_pred, prev_func_pred)

  num = sum(psuedo)
  denom = sum((labels .- psuedo) .* (1 .- labels .+ psuedo))
  if denom == 0.0
    return 0.0
  else
    return num / denom
  end
end

end # module
