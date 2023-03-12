crate::ix!();

/**
  | Performs the same operation as SparseLengthsSum,
  | but operating on 2-bit rowwise quantized
  | matrices with fused storage (where
  | each row stores quantized values, and
  | then 2-byte fp16 scale and bias).
  |
  */
register_cpu_operator!{
    SparseLengthsSumFused2BitRowwise,
    SparseLengthsFusedNBitRowwiseOp<2, CPUContext>}

num_inputs!{SparseLengthsSumFused2BitRowwise, 3}

num_outputs!{SparseLengthsSumFused2BitRowwise, 1}

inputs!{SparseLengthsSumFused2BitRowwise, 
    0 => ("DATA", "uint8 tensor obtained with operator FloatToFused2BitRowwiseQuantized"),
    1 => ("INDICES", "Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated"),
    2 => ("LENGTHS", "Vector with the same sum of elements as the first dimension of DATA")
}

outputs!{SparseLengthsSumFused2BitRowwise, 
    0 => ("output", "output")
}

inherit_onnx_schema!{SparseLengthsSumFused2BitRowwise}

value_key_length_input_fillers!{
    /*
    SparseLengthsSumFused2BitRowwise, (
        SparseLengthsFusedNBitRowwiseOp<2, CPUContext>::DATA,
        SparseLengthsFusedNBitRowwiseOp<2, CPUContext>::INDICES,
        SparseLengthsFusedNBitRowwiseOp<2, CPUContext>::LENGTHS)
    */
}

no_gradient!{SparseLengthsSumFused2BitRowwise}

/**
  | Performs the same operation as
  | SparseLengthsWeightedSum, but operating
  | on 2-bit rowwise quantized matrices
  | with fused storage (where each row stores
  | quantized values, and then 2-byte fp16
  | scale and bias).
  |
  */
register_cpu_operator!{
    SparseLengthsWeightedSumFused2BitRowwise,
    SparseLengthsFusedNBitRowwiseOp<2, CPUContext, /*with_weights=*/true>}

num_inputs!{SparseLengthsWeightedSumFused2BitRowwise, 4}

num_outputs![SparseLengthsWeightedSumFused2BitRowwise, 1];

inputs!{SparseLengthsWeightedSumFused2BitRowwise, 
    0 => ("DATA", "uint8 tensor obtained with operator FloatToFused2BitRowwiseQuantized"),
    1 => ("WEIGHTS", "Vector of weights to scale rows of DATA with before reduction"),
    2 => ("INDICES", "Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated"),
    3 => ("LENGTHS", "Vector with the same sum of elements as the first dimension of DATA")
}

outputs!{SparseLengthsWeightedSumFused2BitRowwise, 
    0 => ("output", "output")
}

weighted_value_key_length_input_fillers!{
    /*
    SparseLengthsWeightedSumFused2BitRowwise, (
        SparseLengthsFusedNBitRowwiseOp<2, CPUContext, true>::DATA,
        SparseLengthsFusedNBitRowwiseOp<2, CPUContext, true>::INDICES,
        SparseLengthsFusedNBitRowwiseOp<2, CPUContext, true>::LENGTHS,
        SparseLengthsFusedNBitRowwiseOp<2, CPUContext, true>::WEIGHTS)
    */
}

no_gradient!{SparseLengthsWeightedSumFused2BitRowwise}

/**
  | Performs the same operation as SparseLengthsMean,
  | but operating on 2-bit rowwise quantized
  | matrices with fused storage (where
  | each row stores quantized values, and
  | then 2-byte fp16 scale and bias).
  |
  */
register_cpu_operator!{
    SparseLengthsMeanFused2BitRowwise,
    SparseLengthsFusedNBitRowwiseOp<
        2,
        CPUContext,
        /*with_weights=*/false,
        /*is_mean=*/true>}

num_inputs!{SparseLengthsMeanFused2BitRowwise, 3}

num_outputs!{SparseLengthsMeanFused2BitRowwise, 1}

inputs!{SparseLengthsMeanFused2BitRowwise, 
    0 => ("DATA", "uint8 tensor obtained with operator FloatToFused2BitRowwiseQuantized"),
    1 => ("INDICES", "Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated"),
    2 => ("LENGTHS", "Vector with the same sum of elements as the first dimension of DATA")
}

outputs!{SparseLengthsMeanFused2BitRowwise, 
    0 => ("output", "output")
}

value_key_length_input_fillers!{
    /*
    SparseLengthsMeanFused2BitRowwise, (
        SparseLengthsFusedNBitRowwiseOp<2, CPUContext, false, true>::DATA,
        SparseLengthsFusedNBitRowwiseOp<2, CPUContext, false, true>::INDICES,
        SparseLengthsFusedNBitRowwiseOp<2, CPUContext, false, true>::LENGTHS)
    */
}


no_gradient!{SparseLengthsMeanFused2BitRowwise}

