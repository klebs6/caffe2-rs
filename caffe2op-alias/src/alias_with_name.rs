crate::ix!();

/**
  | Similar with AliasOp, storing the alias
  | name as operator argument.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct AliasWithNameOp<Context> {
    storage:     OperatorStorage,
    context:     Context,
    name:        String,
    is_backward: bool,
}

num_inputs!{AliasWithName, 1}

num_outputs!{AliasWithName, 1}

inputs!{AliasWithName, 
    0 => ("input", "Input tensor whose storage will be shared.")
}

outputs!{AliasWithName, 
    0 => ("output", "Tensor of same shape as input, sharing its storage.")
}

args!{AliasWithName, 
    0 => ("name", "name of the aliasing"),
    1 => ("is_backward", "weather or not to alias forward or backward")
}

identical_type_and_shape!{AliasWithName}

allow_inplace!{AliasWithName, vec![(0, 0)]}

register_cpu_operator!{
    AliasWithName, 
    AliasWithNameOp<CPUContext>
}
