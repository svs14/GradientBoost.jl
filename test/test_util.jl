module TestUtil

using GradientBoost.Util

using Test

@testset "Util functions" begin
  @testset "err_must_be_overriden throws an error" begin
    @test_throws ErrorException err_must_be_overriden()
  end;

  @testset "weighted_median works" begin
    weights = [1, 1, 1, 1, 1]
    values = [1, 2, 3, 4, 5]
    @test weighted_median(weights, values) == 3

    weights = [1, 1, 1, 1, 1]
    values = [5, 4, 3, 2, 1]
    @test weighted_median(weights, values) == 3

    weights = [1, 1, 1, 1, 1]
    values = [4, 5, 2, 3, 1]
    @test weighted_median(weights, values) == 3

    weights = [1, 3, 1, 1, 1]
    values = [1, 2, 3, 4, 5]
    @test weighted_median(weights, values) == 2

    weights = [1, 1, 1, 1, 100]
    values = [1, 2, 3, 4, 5]
    @test weighted_median(weights, values) == 5

    weights = [100, 1, 1, 1, 1]
    values = [1, 2, 3, 4, 5]
    @test weighted_median(weights, values) == 1

    weights = [0.5, 0.5, 1, 1, 1]
    values = [1, 2, 3, 4, 5]
    @test weighted_median(weights, values) == 3
  end;

  @testset "holdout returns proportional partitions" begin
    n = 10
    right_prop = 0.3
    (left, right) = holdout(n, right_prop)

    @test size(left, 1) == n - (n * right_prop)
    @test size(right, 1) == n * right_prop
    @test isempty(intersect(left, right))
    @test size(union(left, right), 1) == n
  end;
end

end # module
