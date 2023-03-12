crate::ix!();

/**
  | Performs SparseLengthsSum, but operating
  | on 2-bit rowwise quantized matrices
  | with fused storage (where each row stores
  | quantized values, and then 2-byte fp16
  | scale and 2-byte fp16 bias), and where
  | rows are pruned.
  |
  */
register_cpu_operator!{
    SparseLengthsSum2BitRowwiseSparse,
    SparseLengthsNBitRowwiseSparseOp<2>}

num_inputs!{SparseLengthsSum2BitRowwiseSparse, 4}

num_outputs!{SparseLengthsSum2BitRowwiseSparse, 1}

inputs!{SparseLengthsSum2BitRowwiseSparse, 
    0 => ("DATA", "uint8 tensor obtained with operator FloatToFused2BitRowwiseQuantized"),
    1 => ("INDICES", "Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated"),
    2 => ("LENGTHS", "Vector with the same sum of elements as the first dimension of DATA"),
    3 => ("COMPRESSED_INDICES_MAPPING", "Integer vector mapping uncompressed indices to compressed indices")
}

outputs!{SparseLengthsSum2BitRowwiseSparse, 
    0 => ("output", "output")
}

inherit_onnx_schema!{SparseLengthsSum2BitRowwiseSparse}

value_key_length_input_fillers!{
    /*
    SparseLengthsSum2BitRowwiseSparse, (
        SparseLengthsNBitRowwiseSparseOp<2>::COMPRESSED_INDICES_MAPPING,
        SparseLengthsNBitRowwiseSparseOp<2>::INDICES,
        SparseLengthsNBitRowwiseSparseOp<2>::LENGTHS)

        */
}

no_gradient!{SparseLengthsSum2BitRowwiseSparse}

/**
  | Performs SparseLengthsWeightedSum,
  | but operating on 2-bit rowwise quantized
  | matrices with fused storage (where
  | each row stores quantized values, and
  | then 2-byte fp16 scale and bias), and
  | where rows are pruned.
  |
  */
register_cpu_operator!{
    SparseLengthsWeightedSum2BitRowwiseSparse,
    SparseLengthsNBitRowwiseSparseOp<
        2,
        /*with_weights=*/true>
}

num_inputs!{SparseLengthsWeightedSum2BitRowwiseSparse, 5}

num_outputs!{SparseLengthsWeightedSum2BitRowwiseSparse, 1}

inputs!{SparseLengthsWeightedSum2BitRowwiseSparse, 
    0 => ("DATA", "uint8 tensor obtained with operator FloatToFused2BitRowwiseQuantized"),
    1 => ("WEIGHTS", "Vector of weights to scale rows of DATA with before reduction"),
    2 => ("INDICES", "Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated"),
    3 => ("LENGTHS", "Vector with the same sum of elements as the first dimension of DATA"),
    4 => ("COMPRESSED_INDICES_MAPPING", "Integer vector mapping uncompressed indices to compressed indices")
}

outputs!{SparseLengthsWeightedSum2BitRowwiseSparse, 
    0 => ("output", "output")
}

weighted_value_key_length_input_fillers!{
    /*
    SparseLengthsWeightedSum2BitRowwiseSparse, (
        SparseLengthsNBitRowwiseSparseOp<2, true>::COMPRESSED_INDICES_MAPPING,
        SparseLengthsNBitRowwiseSparseOp<2, true>::INDICES,
        SparseLengthsNBitRowwiseSparseOp<2, true>::LENGTHS,
        SparseLengthsNBitRowwiseSparseOp<2, true>::WEIGHTS)

        */
}

no_gradient!{SparseLengthsWeightedSum2BitRowwiseSparse}
