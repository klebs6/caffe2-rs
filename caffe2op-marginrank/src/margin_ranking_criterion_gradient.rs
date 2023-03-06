crate::ix!();

/**
  | MarginRankingCriterionGradient
  | takes both X1, X2, Y and dY and uses them
  | to update dX1, and dX2 according to the
  | chain rule and derivatives of the loss
  | function.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct MarginRankingCriterionGradientOp<Context> {
    storage: OperatorStorage,
    context: Context,
    margin:  f32,
}

num_inputs!{MarginRankingCriterionGradient, 4}

num_outputs!{MarginRankingCriterionGradient, 2}

impl<Context> MarginRankingCriterionGradientOp<Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(float, "margin", margin_, 1.0)
        */
    }
}
