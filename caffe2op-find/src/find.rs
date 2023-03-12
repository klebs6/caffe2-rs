crate::ix!();

/**
  | Finds elements of second input from
  | first input, outputting the last (max)
  | index for each query.
  | 
  | If query not find, inserts missing_value.
  | 
  | See IndexGet() for a version that modifies
  | the index when values are not found.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
#[USE_DISPATCH_HELPER]
pub struct FindOp<Context> {
    storage:       OperatorStorage,
    context:       Context,
    missing_value: i32,
}

num_inputs!{Find, 2}

num_outputs!{Find, 1}

inputs!{Find, 
    0 => ("index", "Index (integers)"),
    1 => ("query", "Needles / query")
}

outputs!{Find, 
    0 => ("query_indices", "Indices of the needles in index or 'missing value'")
}

args!{Find, 
    0 => ("missing_value", "Placeholder for items that are not found")
}

identical_type_and_shape_of_input!{Find, 1}

register_cpu_operator!{Find, FindOp<CPUContext>}
