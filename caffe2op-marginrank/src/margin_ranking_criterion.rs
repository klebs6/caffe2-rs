crate::ix!();

/**
  | MarginRankingCriterion takes two
  | input data X1 (Tensor),
  | 
  | X2 (Tensor), and label Y (Tensor) to
  | produce the loss (Tensor) where the
  | loss function, loss(X1, X2, Y) = max(0,
  | -Y * (X1 - X2) + margin), is applied to
  | the tensor elementwise.
  | 
  | If y == 1 then it assumed the first input
  | should be ranked higher (have a larger
  | value) than the second input, and vice-versa
  | for y == -1.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct MarginRankingCriterionOp<Context> {
    storage: OperatorStorage,
    context: Context,
    margin: f32,
}

num_inputs!{MarginRankingCriterion, 3}

num_outputs!{MarginRankingCriterion, 1}

inputs!{MarginRankingCriterion, 
    0 => ("X1", "The left input vector as a 1-dim TensorCPU."),
    1 => ("X2", "The right input vector as a 1-dim TensorCPU."),
    2 => ("Y",  "The label as a 1-dim TensorCPU with int value of 1 or -1.")
}

outputs!{MarginRankingCriterion, 
    0 => ("loss", "The output loss with the same dimensionality as X1.")
}

args!{MarginRankingCriterion, 
    0 => ("margin", "The margin value as a float. Default is 1.0.")
}

impl<Context> MarginRankingCriterionOp<Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(float, "margin", margin_, 1.0)
        */
    }
}
