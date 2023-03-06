crate::ix!();

/**
 | Given input vector LENGTHS, and input n_split,
 | LengthsSplit returns a single output vector. 
 |
 | It "splits" each length into n_split values which
 | add up to the original length. 
 |
 | It will attempt to do equal splits, and if not
 | possible, it orders larger values first. 
 |
 | If the n_split is larger than the length, zero
 | padding will be applied.
 |
 | e.g. LENGTHS = [9 4 5]
 |      n_split = 3
 |      Y = [3 3 3 2 1 1 2 2 1]
 |
 | e.g. LENGTHS = [2, 1, 2]
 |      n_split = 3
 |      Y = [1 1 0 1 0 0 1 1 0]
 */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct LengthsSplitOp<Context> {
    storage: OperatorStorage,
    context: Context,
    n_split: i32,
}

num_inputs!{LengthsSplit, (1,2)}

num_outputs!{LengthsSplit, 1}

inputs!{LengthsSplit, 
    0 => ("LENGTHS", "Mx1 Input tensor denoting INT32 lengths"),
    1 => ("n_split", "(Optional) Number of splits for each element in LENGTHS (overrides argument)")
}

outputs!{LengthsSplit, 
    0 => ("Y", "(M*n_split)x1 Output vector denoting split lengths")
}

args!{LengthsSplit, 
    0 => ("n_split", "Number of splits for each element in LENGTHS")
}

scalar_type!{LengthsSplit, TensorProto::INT32}

register_cpu_operator!{LengthsSplit, LengthsSplitOp<CPUContext>}

// TODO: Write gradient for this when needed
gradient_not_implemented_yet!{LengthsSplit}
