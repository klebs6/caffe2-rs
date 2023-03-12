crate::ix!();

/**
  | Performs SparseLengthsSum, but operating
  | on 4-bit rowwise quantized matrices
  | with fused storage (where each row stores
  | quantized values, and then 2-byte fp16
  | scale and 2-byte fp16 bias), and where
  | rows are pruned.
  |
  */
register_cpu_operator!{
    SparseLengthsSum4BitRowwiseSparse,
    SparseLengthsNBitRowwiseSparseOp<4>}

num_inputs!{SparseLengthsSum4BitRowwiseSparse, 4}

num_outputs!{SparseLengthsSum4BitRowwiseSparse, 1}

inputs!{SparseLengthsSum4BitRowwiseSparse, 
    0 => ("DATA", "uint8 tensor obtained with operator FloatToFused4BitRowwiseQuantized"),
    1 => ("INDICES", "Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated"),
    2 => ("LENGTHS", "Vector with the same sum of elements as the first dimension of DATA"),
    3 => ("COMPRESSED_INDICES_MAPPING", "Integer vector mapping uncompressed indices to compressed indices")
}

outputs!{SparseLengthsSum4BitRowwiseSparse, 
    0 => ("output", "output")
}

inherit_onnx_schema!{SparseLengthsSum4BitRowwiseSparse}

value_key_length_input_fillers!{
    /*
    SparseLengthsSum4BitRowwiseSparse, (
        SparseLengthsNBitRowwiseSparseOp<4>::COMPRESSED_INDICES_MAPPING,
        SparseLengthsNBitRowwiseSparseOp<4>::INDICES,
        SparseLengthsNBitRowwiseSparseOp<4>::LENGTHS)
    */
}

no_gradient!{SparseLengthsSum4BitRowwiseSparse}

/**
  | Performs SparseLengthsWeightedSum,
  | but operating on 4-bit rowwise quantized
  | matrices with fused storage (where
  | each row stores quantized values, and
  | then 2-byte fp16 scale and bias), and
  | where rows are pruned.
  |
  */
register_cpu_operator!{
    SparseLengthsWeightedSum4BitRowwiseSparse,
    SparseLengthsNBitRowwiseSparseOp<
        4,
        /*with_weights=*/true>}

num_inputs!{SparseLengthsWeightedSum4BitRowwiseSparse, 5}

num_outputs!{SparseLengthsWeightedSum4BitRowwiseSparse, 1}

inputs!{SparseLengthsWeightedSum4BitRowwiseSparse, 
    0 => ("DATA", "uint8 tensor obtained with operator FloatToFused4BitRowwiseQuantized"),
    1 => ("WEIGHTS", "Vector of weights to scale rows of DATA with before reduction"),
    2 => ("INDICES", "Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated"),
    3 => ("LENGTHS", "Vector with the same sum of elements as the first dimension of DATA"),
    4 => ("COMPRESSED_INDICES_MAPPING", "Integer vector mapping uncompressed indices to compressed indices")
}

outputs!{SparseLengthsWeightedSum4BitRowwiseSparse, 
    0 => ("output", "output")
}

weighted_value_key_length_input_fillers!{
    /*
    SparseLengthsWeightedSum4BitRowwiseSparse, (
        SparseLengthsNBitRowwiseSparseOp<4, true>::COMPRESSED_INDICES_MAPPING,
        SparseLengthsNBitRowwiseSparseOp<4, true>::INDICES,
        SparseLengthsNBitRowwiseSparseOp<4, true>::LENGTHS,
        SparseLengthsNBitRowwiseSparseOp<4, true>::WEIGHTS)
    */
}

no_gradient!{SparseLengthsWeightedSum4BitRowwiseSparse}

/**
  | Performs SparseLengthsMean, but operating
  | on 4-bit rowwise quantized matrices
  | with fused storage (where each row stores
  | quantized values, and then 2-byte fp16
  | scale and bias), and where rows are pruned.
  |
  */
register_cpu_operator!{
    SparseLengthsMean4BitRowwiseSparse,
    SparseLengthsNBitRowwiseSparseOp<
        4,
        /*with_weights=*/false,
        /*is_mean=*/true>}

num_inputs!{SparseLengthsMean4BitRowwiseSparse, 4}

num_outputs!{SparseLengthsMean4BitRowwiseSparse, 1}

inputs!{SparseLengthsMean4BitRowwiseSparse, 
    0 => ("DATA", "uint8 tensor obtained with operator FloatToFused4BitRowwiseQuantized"),
    1 => ("INDICES", "Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated"),
    2 => ("LENGTHS", "Vector with the same sum of elements as the first dimension of DATA"),
    3 => ("COMPRESSED_INDICES_MAPPING", "Integer vector mapping uncompressed indices to compressed indices")
}

outputs!{SparseLengthsMean4BitRowwiseSparse, 
    0 => ("output", "output")
}

value_key_length_input_fillers!{
    /*
    SparseLengthsMean4BitRowwiseSparse, (
        SparseLengthsNBitRowwiseSparseOp<4, false, true>::
            COMPRESSED_INDICES_MAPPING,
        SparseLengthsNBitRowwiseSparseOp<4, false, true>::INDICES,
        SparseLengthsNBitRowwiseSparseOp<4, false, true>::LENGTHS)
    */
}

no_gradient!{SparseLengthsMean4BitRowwiseSparse}

