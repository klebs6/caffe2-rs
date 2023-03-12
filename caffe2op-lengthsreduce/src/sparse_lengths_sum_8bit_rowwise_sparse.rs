crate::ix!();

/**
  | Performs SparseLengthsSum, but operating
  | on 8-bit rowwise quantized matrices
  | with fused storage (where each row stores
  | quantized values, and then 4-byte fp32
  | scale and 4-byte fp32 bias), and where
  | rows are pruned.
  |
  */
register_cpu_operator!{
    SparseLengthsSum8BitRowwiseSparse,
    SparseLengthsNBitRowwiseSparseOp<8>}

num_inputs!{SparseLengthsSum8BitRowwiseSparse, 4}

num_outputs!{SparseLengthsSum8BitRowwiseSparse, 1}

inputs!{SparseLengthsSum8BitRowwiseSparse, 
    0 => ("DATA", "uint8 tensor obtained with operator FloatToFused4BitRowwiseQuantized"),
    1 => ("INDICES", "Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated"),
    2 => ("LENGTHS", "Vector with the same sum of elements as the first dimension of DATA"),
    3 => ("COMPRESSED_INDICES_MAPPING", "Integer vector mapping uncompressed indices to compressed indices")
}

outputs!{SparseLengthsSum8BitRowwiseSparse, 
    0 => ("output", "output")
}

inherit_onnx_schema!{SparseLengthsSum8BitRowwiseSparse}

value_key_length_input_fillers!{
    /*
    SparseLengthsSum8BitRowwiseSparse, (
        SparseLengthsNBitRowwiseSparseOp<8>::COMPRESSED_INDICES_MAPPING,
        SparseLengthsNBitRowwiseSparseOp<8>::INDICES,
        SparseLengthsNBitRowwiseSparseOp<8>::LENGTHS)
    */
}

no_gradient!{SparseLengthsSum8BitRowwiseSparse}

/**
  | Performs SparseLengthsWeightedSum,
  | but operating on 8-bit rowwise quantized
  | matrices with fused storage (where
  | each row stores quantized values, and
  | then 4-byte fp32 scale and bias), and
  | where rows are pruned.
  |
  */
register_cpu_operator!{
    SparseLengthsWeightedSum8BitRowwiseSparse,
    SparseLengthsNBitRowwiseSparseOp<
        8,
        /*with_weights=*/true>}

num_inputs!{SparseLengthsWeightedSum8BitRowwiseSparse, 5}

num_outputs!{SparseLengthsWeightedSum8BitRowwiseSparse, 1}

inputs!{SparseLengthsWeightedSum8BitRowwiseSparse, 
    0 => ("DATA", "uint8 tensor obtained with operator FloatToFused4BitRowwiseQuantized"),
    1 => ("WEIGHTS", "Vector of weights to scale rows of DATA with before reduction"),
    2 => ("INDICES", "Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated"),
    3 => ("LENGTHS", "Vector with the same sum of elements as the first dimension of DATA"),
    4 => ("COMPRESSED_INDICES_MAPPING", "Integer vector mapping uncompressed indices to compressed indices")
}

outputs!{SparseLengthsWeightedSum8BitRowwiseSparse, 
    0 => ("output", "output")
}

weighted_value_key_length_input_fillers!{
    /*
    SparseLengthsWeightedSum8BitRowwiseSparse, (
        SparseLengthsNBitRowwiseSparseOp<8, true>::COMPRESSED_INDICES_MAPPING,
        SparseLengthsNBitRowwiseSparseOp<8, true>::INDICES,
        SparseLengthsNBitRowwiseSparseOp<8, true>::LENGTHS,
        SparseLengthsNBitRowwiseSparseOp<8, true>::WEIGHTS)
    */
}

no_gradient!{SparseLengthsWeightedSum8BitRowwiseSparse}

/**
  | Performs SparseLengthsMean, but operating
  | on 8-bit rowwise quantized matrices
  | with fused storage (where each row stores
  | quantized values, and then 4-byte fp32
  | scale and bias), and where rows are pruned.
  |
  */
register_cpu_operator!{
    SparseLengthsMean8BitRowwiseSparse,
    SparseLengthsNBitRowwiseSparseOp<
        8,
        /*with_weights=*/false,
        /*is_mean=*/true>}

num_inputs!{SparseLengthsMean8BitRowwiseSparse, 4}

num_outputs!{SparseLengthsMean8BitRowwiseSparse, 1}

inputs!{SparseLengthsMean8BitRowwiseSparse, 
    0 => ("DATA", "uint8 tensor obtained with operator FloatToFused4BitRowwiseQuantized"),
    1 => ("INDICES", "Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated"),
    2 => ("LENGTHS", "Vector with the same sum of elements as the first dimension of DATA"),
    3 => ("COMPRESSED_INDICES_MAPPING", "Integer vector mapping uncompressed indices to compressed indices")
}

outputs!{SparseLengthsMean8BitRowwiseSparse, 
    0 => ("output", "output")
}

value_key_length_input_fillers!{
    /*
    SparseLengthsMean8BitRowwiseSparse, (
        SparseLengthsNBitRowwiseSparseOp<8, false, true>::
            COMPRESSED_INDICES_MAPPING,
        SparseLengthsNBitRowwiseSparseOp<8, false, true>::INDICES,
        SparseLengthsNBitRowwiseSparseOp<8, false, true>::LENGTHS)
        */
}

no_gradient!{SparseLengthsMean8BitRowwiseSparse}
