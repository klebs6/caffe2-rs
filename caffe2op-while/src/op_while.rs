crate::ix!();

/**
  | 'While' control operator, first input
  | is a scalar boolean blob that stores
  | loop's condition value.
  | 
  | Accepts 'loop_net' (required) and
  | 'cond_net' (optional) arguments for
  | loop's body and condition subnets respectively.
  | 
  | If condition subnet is specified, it
  | is executed before the first and after
  | each iteration.
  | 
  | Subnets are executed in the same workspace
  | as 'While'.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct WhileOp<Context> {
    storage:      OperatorStorage,
    context:      Context,
    loop_net_def: NetDef,
    loop_net:     Box<NetBase>,
    cond_net_def: NetDef,
    cond_net:     Box<NetBase>,
}

num_inputs!{While, (1,INT_MAX)}

num_outputs!{While, (0,INT_MAX)}

inputs!{While, 
    0 => ("condition", "Scalar boolean condition")
}

args!{While, 
    0 => ("loop_net", "Net executed on each iteration"),
    1 => ("cond_net", "Net to (re)compute condition value")
}

allow_inplace!{While, 
    |input: i32, output: i32| -> bool {
        true
    }
}

register_cpu_operator!{While,  WhileOp<CPUContext>}

register_cuda_operator!{While, WhileOp<CUDAContext>}
