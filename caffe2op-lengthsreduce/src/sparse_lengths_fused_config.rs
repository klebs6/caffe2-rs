crate::ix!();

/**
  | Performs the same operation as SparseLengthsSum,
  | but operating on 8-bit rowwise quantized
  | matrices with fused storage (where
  | each row stores quantized values, and
  | then 4-byte scale and 4-byte bias).
  |
  */
register_cpu_operator!{
    SparseLengthsSumFused8BitRowwise,
    SparseLengthsFused8BitRowwiseOp<CPUContext>
}

num_inputs!{SparseLengthsSumFused8BitRowwise, 3}

num_outputs!{SparseLengthsSumFused8BitRowwise, 1}

inputs!{SparseLengthsSumFused8BitRowwise, 
    0 => ("DATA", "uint8 tensor obtained with operator FloatToFused8BitRowwiseQuantized"),
    1 => ("INDICES", "Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated"),
    2 => ("LENGTHS", "Vector with the same sum of elements as the first dimension of DATA")
}

outputs!{SparseLengthsSumFused8BitRowwise, 
    0 => ("output", "output")
}

inherit_onnx_schema!{SparseLengthsSumFused8BitRowwise}

value_key_length_input_fillers!{
    /*
    SparseLengthsSumFused8BitRowwise, 

       SparseLengthsFused8BitRowwiseOp<CPUContext>::DATA,
       SparseLengthsFused8BitRowwiseOp<CPUContext>::INDICES,
       SparseLengthsFused8BitRowwiseOp<CPUContext>::LENGTHS
       */
}


no_gradient!{SparseLengthsSumFused8BitRowwise}

/**
  | Performs the same operation as
  | SparseLengthsWeightedSum, but operating
  | on 8-bit rowwise quantized matrices
  | with fused storage (where each row stores
  | quantized values, and then 4-byte scale
  | and 4-byte bias).
  |
  */
register_cpu_operator!{
    SparseLengthsWeightedSumFused8BitRowwise,
    SparseLengthsFused8BitRowwiseOp<CPUContext, /*with_weights=*/true>
}

num_inputs!{SparseLengthsWeightedSumFused8BitRowwise, 4}

num_outputs!{SparseLengthsWeightedSumFused8BitRowwise, 1}

inputs!{SparseLengthsWeightedSumFused8BitRowwise, 
    0 => ("DATA", "uint8 tensor obtained with operator FloatToFused8BitRowwiseQuantized"),
    1 => ("WEIGHTS", "Vector of weights to scale rows of DATA with before reduction"),
    2 => ("INDICES", "Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated"),
    3 => ("LENGTHS", "Vector with the same sum of elements as the first dimension of DATA")
}

outputs!{SparseLengthsWeightedSumFused8BitRowwise, 
    0 => ("output", "output")
}

weighted_value_key_length_input_fillers!{
    /*
    SparseLengthsWeightedSumFused8BitRowwise, 

       SparseLengthsFused8BitRowwiseOp<CPUContext, true>::DATA,
       SparseLengthsFused8BitRowwiseOp<CPUContext, true>::INDICES,
       SparseLengthsFused8BitRowwiseOp<CPUContext, true>::LENGTHS,
       SparseLengthsFused8BitRowwiseOp<CPUContext, true>::WEIGHTS
       */
}

no_gradient!{SparseLengthsWeightedSumFused8BitRowwise}

/**
  | Performs the same operation as SparseLengthsMean,
  | but operating on 8-bit rowwise quantized
  | matrices with fused storage (where
  | each row stores quantized values, and
  | then 4-byte scale and 4-byte bias).
  |
  */
register_cpu_operator!{
    SparseLengthsMeanFused8BitRowwise,
    SparseLengthsFused8BitRowwiseOp<
        CPUContext,
        /*with_weights=*/false,
        /*is_mean=*/true>
}

num_inputs!{SparseLengthsMeanFused8BitRowwise, 3}

num_outputs!{SparseLengthsMeanFused8BitRowwise, 1}

inputs!{SparseLengthsMeanFused8BitRowwise, 
    0 => ("DATA", "uint8 tensor obtained with operator FloatToFused8BitRowwiseQuantized"),
    1 => ("INDICES", "Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated"),
    2 => ("LENGTHS", "Vector with the same sum of elements as the first dimension of DATA")
}

outputs!{SparseLengthsMeanFused8BitRowwise, 
    0 => ("output", "output")
}

value_key_length_input_fillers!{
    /* 
    SparseLengthsMeanFused8BitRowwise, 

     SparseLengthsFused8BitRowwiseOp<CPUContext, false, true>::DATA,
       SparseLengthsFused8BitRowwiseOp<CPUContext, false, true>::INDICES,
       SparseLengthsFused8BitRowwiseOp<CPUContext, false, true>::LENGTHS*/
}

no_gradient!{SparseLengthsMeanFused8BitRowwise}
