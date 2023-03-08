crate::ix!();

/**
  | PReluGradient takes both Y and dY and
  | uses this to update dX and dW according
  | to the chain rule and derivatives of
  | the rectified linear function.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct PReluGradientOp<T,Context> {
    storage: OperatorStorage,
    context: Context,
    order:   StorageOrder,

    /**
      | Input: Y, dY,
      | 
      | output: dX
      |
      */
    phantom: PhantomData<T>,
}

num_inputs!{PReluGradient, 4}

num_outputs!{PReluGradient, 2}

identical_type_and_shape_of_multiple_inputs!{
    PReluGradient, 
    vec![(2, 3)]
}

impl<T,Context> PReluGradientOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            order_(StringToStorageOrder(
                this->template GetSingleArgument<string>("order", "NCHW")))
        */
    }
}
