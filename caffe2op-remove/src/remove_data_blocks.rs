crate::ix!();

/**
  | Shrink the data tensor by removing data
  | blocks with given zero-based indices
  | in the outermost dimension of the tensor.
  | 
  | Indices are not assumed in any order
  | or unique but with the range [0, blocks_size).
  | Indices could be empty.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
#[USE_SIMPLE_CTOR_DTOR("RemoveDataBlocksOp")]
#[USE_DISPATCH_HELPER]
pub struct RemoveDataBlocksOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

input_tags!{
    RemoveDataBlocksOp {
        Data,
        Indices
    }
}

register_cpu_operator!{RemoveDataBlocks, RemoveDataBlocksOp<CPUContext>}

num_inputs!{RemoveDataBlocks, 2}

num_outputs!{RemoveDataBlocks, 1}

inputs!{RemoveDataBlocks, 
    0 => ("data", "a N-D data tensor, N >= 1"),
    1 => ("indices", "zero-based indices of blocks to be removed")
}

outputs!{RemoveDataBlocks, 
    0 => ("shrunk data", "data after removing data blocks indexed by 'indices'")
}

should_not_do_gradient!{RemoveDataBlocks}
