crate::ix!();

/**
  | Given a 1-D tensor of boolean values,
  | apply conditional operator along the
  | first dimension of DataT and DataF and 
  | return DataO. Note, DataT and
  | DataF must have the exact same shape
  | and type.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct ConditionalOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

register_cpu_operator!{Conditional, ConditionalOp<CPUContext>}

num_inputs!{Conditional, 3}

num_outputs!{Conditional, 1}

inputs!{Conditional, 
    0 => ("Condition", "Boolean tensor to select DataT or DataF"),
    1 => ("DataT", "Data to use when True"),
    2 => ("DataF", "Data to use when False")
}

outputs!{Conditional, 
    0 => ("DataO", "Output data after applying ConditionalOp")
}

identical_type_and_shape_of_input!{Conditional, 1}

no_gradient!{Conditional}

impl<Context> ConditionalOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
}
