crate::ix!();

/**
  | Identity operator, but checks all values
  | for nan or inf
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct NanCheckOp<Context, W: Write> {
    storage:        OperatorStorage,
    context:        Context,
    tensor_printer: TensorPrinter<W>,
    scratch:        Tensor,
}

register_cpu_operator!{NanCheck, NanCheckOp<CPUContext>}

register_gradient!{NanCheck,     GetNanCheckGradient}

num_inputs!{NanCheck, (1,INT_MAX)}

num_outputs!{NanCheck, 1}

inputs!{NanCheck, 
    0 => ("tensor", "Tensor to check for nan/inf")
}

outputs!{NanCheck, 
    0 => ("output", "Tensor to copy input into if no NaNs or inf. Can be in-place")
}

identical_type_and_shape_of_input!{NanCheck, 0}

allow_inplace!{NanCheck, vec![(0, 0)]}

impl<Context,W: Write> NanCheckOp<Context,W> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
}
