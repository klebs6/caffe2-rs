crate::ix!();

/**
  | Performs the same operation as SparseLengthsSum,
  | but operating on 4-bit rowwise quantized
  | matrices with fused storage (where
  | each row stores quantized values, and
  | then 2-byte fp16 scale and bias).
  |
  */
register_cpu_operator!{
    SparseLengthsSumFused4BitRowwise,
    SparseLengthsFusedNBitRowwiseOp<4, CPUContext>
}

num_inputs!{SparseLengthsSumFused4BitRowwise, 3}

num_outputs!{SparseLengthsSumFused4BitRowwise, 1}

inputs!{SparseLengthsSumFused4BitRowwise, 
    0 => ("DATA",    "uint8 tensor obtained with operator FloatToFused4BitRowwiseQuantized"),
    1 => ("INDICES", "Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated"),
    2 => ("LENGTHS", "Vector with the same sum of elements as the first dimension of DATA")
}

outputs!{SparseLengthsSumFused4BitRowwise, 
    0 => ("output",  "output")
}

inherit_onnx_schema!{SparseLengthsSumFused4BitRowwise}

value_key_length_input_fillers!{
    /*
    SparseLengthsSumFused4BitRowwise, (
        SparseLengthsFusedNBitRowwiseOp<4, CPUContext>::DATA,
        SparseLengthsFusedNBitRowwiseOp<4, CPUContext>::INDICES,
        SparseLengthsFusedNBitRowwiseOp<4, CPUContext>::LENGTHS)
    */
}

no_gradient!{SparseLengthsSumFused4BitRowwise}

/**
  | Performs the same operation as
  | SparseLengthsWeightedSum, but operating
  | on 4-bit rowwise quantized matrices
  | with fused storage (where each row stores
  | quantized values, and then 2-byte fp16
  | scale and bias).
  |
  */
register_cpu_operator!{
    SparseLengthsWeightedSumFused4BitRowwise,
    SparseLengthsFusedNBitRowwiseOp<4, CPUContext, WithWeights>}

num_inputs!{SparseLengthsWeightedSumFused4BitRowwise, 4}

num_outputs!{SparseLengthsWeightedSumFused4BitRowwise, 1}

inputs!{SparseLengthsWeightedSumFused4BitRowwise, 
    0 => ("DATA",    "uint8 tensor obtained with operator FloatToFused4BitRowwiseQuantized"),
    1 => ("WEIGHTS", "Vector of weights to scale rows of DATA with before reduction"),
    2 => ("INDICES", "Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated"),
    3 => ("LENGTHS", "Vector with the same sum of elements as the first dimension of DATA")
}

outputs!{SparseLengthsWeightedSumFused4BitRowwise, 
    0 => ("output", "output")
}

weighted_value_key_length_input_fillers!{
    /*
    SparseLengthsWeightedSumFused4BitRowwise, (
        SparseLengthsFusedNBitRowwiseOp<4, CPUContext, true>::DATA,
        SparseLengthsFusedNBitRowwiseOp<4, CPUContext, true>::INDICES,
        SparseLengthsFusedNBitRowwiseOp<4, CPUContext, true>::LENGTHS,
        SparseLengthsFusedNBitRowwiseOp<4, CPUContext, true>::WEIGHTS)
    */
}

no_gradient!{SparseLengthsWeightedSumFused4BitRowwise}

/**
  | Performs the same operation as SparseLengthsMean,
  | but operating on 4-bit rowwise quantized
  | matrices with fused storage (where
  | each row stores quantized values, and
  | then 2-byte fp16 scale and bias).
  |
  */
register_cpu_operator!{
    SparseLengthsMeanFused4BitRowwise,
    SparseLengthsFusedNBitRowwiseOp<
        4,
        CPUContext,
        WithoutWeights,
        IsMean>}

num_inputs!{SparseLengthsMeanFused4BitRowwise, 3}

num_outputs!{SparseLengthsMeanFused4BitRowwise, 1}

inputs!{SparseLengthsMeanFused4BitRowwise, 
    0 => ("DATA",    "uint8 tensor obtained with operator FloatToFused4BitRowwiseQuantized"),
    1 => ("INDICES", "Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated"),
    2 => ("LENGTHS", "Vector with the same sum of elements as the first dimension of DATA")
}

outputs!{SparseLengthsMeanFused4BitRowwise, 
    0 => ("output",  "output")
}

value_key_length_input_fillers!{
    /*
    SparseLengthsMeanFused4BitRowwise, (
        SparseLengthsFusedNBitRowwiseOp<4, CPUContext, false, true>::DATA,
        SparseLengthsFusedNBitRowwiseOp<4, CPUContext, false, true>::INDICES,
        SparseLengthsFusedNBitRowwiseOp<4, CPUContext, false, true>::LENGTHS)
    */
}

no_gradient!{SparseLengthsMeanFused4BitRowwise}

