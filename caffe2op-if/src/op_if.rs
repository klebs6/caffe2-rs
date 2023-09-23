crate::ix!();

/**
  | 'If' control operator, first input
  | is a scalar boolean blob that stores
  | condition value.
  | 
  | Accepts 'then_net' (required) and
  | 'else_net' (optional) arguments for
  | 'then' and 'else' subnets respectively.
  | 
  | Subnets are executed in the same workspace
  | as 'If'.
  |
  */
pub struct IfOp<Context> {
    storage:  OperatorStorage,
    context:  Context,
    then_net: Box<NetBase>,
    else_net: Box<NetBase>,
}

num_inputs!{If, (1,INT_MAX)}

num_outputs!{If, (0,INT_MAX)}

inputs!{If, 
    0 => ("condition", "Scalar boolean condition")
}

args!{If, 
    0 => ("then_net", "Net executed when condition is true"),
    1 => ("else_net", "Net executed when condition is false (optional)")
}

allow_inplace!{
    If,
    |input: i32, output: i32| -> bool {
        true
    }
}

register_cpu_operator!{
    If, 
    IfOp<CPUContext>
}

register_cuda_operator!{
    If, 
    IfOp<CUDAContext>
}
