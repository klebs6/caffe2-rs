crate::ix!();

/**
  | Gather items from sparse tensor.
  | 
  | Sparse tensor is described by items
  | and lengths.
  | 
  | This operator gathers items corresponding
  | to lengths at the given indices.
  | 
  | This deliberately doesn't return lengths
  | of OUTPUTS so that both lists and maps
  | can be supported without special cases.
  | 
  | If you need lengths tensor for
  | 
  | OUTPUT, use `Gather`.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct LengthsGatherOp<Context> {
    storage: OperatorStorage,
    context: Context,
    offsets: Vec<i64>,
}

num_inputs!{LengthsGather, 3}

num_outputs!{LengthsGather, 1}

inputs!{LengthsGather, 
    0 => ("ITEMS", "items tensor"),
    1 => ("LENGTHS", "lengths tensor"),
    2 => ("INDICES", "indices into LENGTHS where items should be gathered")
}

outputs!{LengthsGather, 
    0 => ("OUTPUT", "1-D tensor containing gathered items")
}

input_tags!{
    LengthsGatherOp
    {
        Items,
        Lengths,
        Indices
    }
}
