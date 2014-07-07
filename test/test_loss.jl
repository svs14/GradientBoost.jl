module TestLossFunctions

using FactCheck
importall GradientBoost.LossFunctions

type DummyLossFunction <: LossFunction; end

y_examples = {
  [1,1],
  [1,0],
  [0,0],
  [-1,1],
  [-1,-1]
}
y_pred_examples = {
  [1,1],
  [1,1],
  [1,1],
  [1,1],
  [1,1]
}
bern_y_examples = {
  [1,1],
  [1,0],
  [0,1],
  [0,0],
  [0,0]
}

function test_loss(lf::LossFunction, 
  y_examples, y_pred_examples, expected)

  for i = 1:size(y_examples, 1)
    @fact loss(lf, y_examples[i], y_pred_examples[i]) => roughly(expected[i])
  end
end
function test_negative_gradient(lf::LossFunction, 
  y_examples, y_pred_examples, expected)

  for i = 1:size(y_examples, 1)
    @fact negative_gradient(
      lf, y_examples[i], y_pred_examples[i]
    ) => roughly(expected[i])
  end
end
function test_minimizing_scalar(lf::LossFunction, 
  y_examples, expected)

  for i = 1:size(y_examples, 1)
    @fact minimizing_scalar(lf, y_examples[i]) => roughly(expected[i])
  end
end

facts("Loss functions") do
  context("not implemented functions throw an error") do
    emp_vec = Array[]
    dlf = DummyLossFunction()

    @fact_throws loss(dlf, emp_vec, emp_vec)
    @fact_throws negative_gradient(dlf, emp_vec, emp_vec)
    @fact_throws minimizing_scalar(dlf, emp_vec)
  end

  context("LeastSquares loss works") do
    lf = LeastSquares()
    expected = { 0.0, 0.5, 1.0, 2.0, 4.0 }
    test_loss(lf, y_examples, y_pred_examples, expected)
  end
  context("LeastSquares negative_gradient works") do
    lf = LeastSquares()
    expected = { [0,0], [0,-1], [-1,-1], [-2,0], [-2,-2] }
    test_negative_gradient(lf, y_examples, y_pred_examples, expected)
  end
  context("LeastSquares minimizing_scalar works") do
    lf = LeastSquares()
    expected = { 1.0, 0.5, 0.0, 0.0, -1.0 }
    test_minimizing_scalar(lf, y_examples, expected)
  end

  context("LeastAbsoluteDeviation loss works") do
    lf = LeastAbsoluteDeviation()
    expected = { 0.0, 0.5, 1.0, 1.0, 2.0 }
    test_loss(lf, y_examples, y_pred_examples, expected)
  end
  context("LeastAbsoluteDeviation negative_gradient works") do
    lf = LeastAbsoluteDeviation()
    expected = { [0,0], [0,-1], [-1,-1], [-1,0], [-1,-1] }
    test_negative_gradient(lf, y_examples, y_pred_examples, expected)
  end
  context("LeastAbsoluteDeviation minimizing_scalar works") do
    lf = LeastAbsoluteDeviation()
    expected = { 1.0, 0.5, 0.0, 0.0, -1.0 }
    test_minimizing_scalar(lf, y_examples, expected)
  end

  context("BinomialDeviance loss works") do
    lf = BinomialDeviance()
    expected = { 
      0.626523, 1.626523, 1.626523, 2.626523, 2.626523,
    }
    test_loss(lf, bern_y_examples, y_pred_examples, expected)
  end
  context("BinomialDeviance negative_gradient works") do
    lf = BinomialDeviance()
    expected = { 
      [0.268941, 0.268941], 
      [0.268941, -0.731059], 
      [-0.731059, 0.268941], 
      [-0.731059, -0.731059], 
      [-0.731059, -0.731059] 
    }
    test_negative_gradient(lf, bern_y_examples, y_pred_examples, expected)
  end
  context("BinomialDeviance minimizing_scalar works") do
    lf = BinomialDeviance()
    expected = { Inf, 0.0, 0.0, -Inf, -Inf}
    test_minimizing_scalar(lf, bern_y_examples, expected)
  end
end

end # module
