crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct CrossEntropyGradientOp<T, Context> {
    storage: OperatorStorage,
    context: Context,

    /**
      | Input: X, label, dY
      | 
      | Ouptut: dX. There is no gradient with
      | respect to the label.
      |
      */
    phantom: PhantomData<T>,
}

num_inputs!{CrossEntropyGradient, 3}

num_outputs!{CrossEntropyGradient, 1}

impl<T, Context> CrossEntropyGradientOp<T, Context> {

    pub const fn k_log_threshold() -> T {
        todo!();
        //return static_cast<T>(1e-20);
    }
}
