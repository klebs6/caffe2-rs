crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct LabelCrossEntropyGradientOp<T, Context> {
    storage: OperatorStorage,
    context: Context,

    /**
      | Input: X, label, dY
      |
      | Ouptut: dX. There is no gradient with
      | respect to the label.
      */
    phantom: PhantomData<T>,
}

num_inputs!{LabelCrossEntropyGradient, 3}

num_outputs!{LabelCrossEntropyGradient, 1}

impl<T,Context> LabelCrossEntropyGradientOp<T, Context> {

    pub const fn k_log_threshold() -> T {
        todo!();
        //return static_cast<T>(1e-20);
    }
}
