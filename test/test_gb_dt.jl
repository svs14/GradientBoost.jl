module TestGBDecisionTree

using Test

using GradientBoost.GBDecisionTree
using GradientBoost.LossFunctions

using DecisionTree

instances = [
  1 1;
  1 8;
  1 10
]
labels = [
  1;
  9;
  9
]

# Create DT.
function create_tree(instances, labels)
  model = build_tree(labels, instances)
  model
end

@testset "GB Decision Tree" begin
  @testset "build_base_func works" begin
    gb = GBDT(;
      loss_function=LeastSquares(),
      sampling_rate=0.5,
      learning_rate=0.01,
      num_iterations=100
    )
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
    expected = [ 6.27; 6.27; 6.27 ]
    @test predictions ≈ expected atol=1e-2
  end

  @testset "instance_to_node indexes" begin
    model = create_tree(instances, labels)
    inst_node_index = GBDecisionTree.InstanceNodeIndex(model, instances)
    num_instances = size(instances, 1)

    @test length(inst_node_index.n2i) == num_instances - 1

    n2i_values = reduce(union, collect(values(inst_node_index.n2i)))
    @test length(n2i_values) == num_instances

    for i = 1:num_instances
      @test in(i, inst_node_index.n2i[inst_node_index.i2n[i]])
    end
  end

  @testset "update_regions! updates terminal regions on tree" begin
    model = create_tree(instances, labels)
    function val_func(leaf::Leaf)
      leaf.majority / 2.0
    end
    n2v = Dict{Leaf, Any}()
    GBDecisionTree.update_regions!(n2v, model, val_func)

    num_instances = size(instances, 1)
    for i = 1:num_instances
      old_pred = apply_tree(model, instances[i,:])
      pred_node = GBDecisionTree.instance_to_node(model, instances[i,:])
      new_pred = n2v[pred_node]

      @test old_pred ./ 2.0 == new_pred
    end
  end

  @testset "LeastAbsoluteDeviation fit_best_constant works" begin
    lf = LeastAbsoluteDeviation()
    dummy_vec = [0.0,0.0,0.0,0.0]
    labels = [0.0,1.0,2.0,3.0]
    prev_func_pred = [3.0,2.0,1.0,0.0]
    expected = 0.0

    actual = GBDecisionTree.fit_best_constant(
      lf, labels, dummy_vec, dummy_vec, prev_func_pred
    )
    @test actual == expected
  end

  @testset "BinomialDeviance fit_best_constant works" begin
    lf = BinomialDeviance()
    dummy_vec = [0.0,0.0,0.0,0.0]

    labels = [0.0,0.0,1.0,1.0]
    psuedo = [0.0,0.5,0.0,0.5]
    expected = -2.0
    actual = GBDecisionTree.fit_best_constant(
      lf, labels, psuedo, dummy_vec, dummy_vec
    )
    @test actual == expected

    labels = [1.0,1.0,1.0,1.0]
    psuedo = [1.0,1.0,1.0,1.0]
    expected = 0.0
    actual = GBDecisionTree.fit_best_constant(
      lf, labels, psuedo, dummy_vec, dummy_vec
    )
    @test actual == expected
  end
end

end # module
