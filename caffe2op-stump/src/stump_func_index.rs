crate::ix!();

/**
  | Split the elements and return the indices
  | based on the given threshold.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct StumpFuncIndexOp<TIN,TOUT,Context> {

    storage: OperatorStorage,
    context: Context,

    threshold:  TIN,

    /**
      | Input: label
      | 
      | output: indices
      |
      */
    phantom: PhantomData<TOUT>,
}

register_cpu_operator!{
    StumpFuncIndex,
    StumpFuncIndexOp<f32, i64, CPUContext>
}

num_inputs!{StumpFuncIndex, 1}

num_outputs!{StumpFuncIndex, 2}

inputs!{StumpFuncIndex, 
    0 => ("X", "tensor of float")
}

outputs!{StumpFuncIndex, 
    0 => ("Index_Low",  "tensor of int64 indices for elements below/equal threshold"),
    1 => ("Index_High", "tensor of int64 indices for elements above threshold")
}

no_gradient!{StumpFuncIndex}

impl<TIN,TOUT,Context> StumpFuncIndexOp<TIN,TOUT,Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            threshold_(this->template GetSingleArgument<TIN>("threshold", 0))
        */
    }
}
