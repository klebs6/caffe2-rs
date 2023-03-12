crate::ix!();

/**
  | Variation of SparseLengthsWeightedSum
  | operator, where
  | 
  | DATA is stored using 8bits. DATA was
  | quantized with 8Bit row-wise quantization
  | (see doc to FloatToRowwiseQuantized8Bits
  | operator). To restore DATA from 8Bit,
  | we use additional input that stores
  | scales and biases.
  |
  */
register_cpu_operator!{
    SparseLengthsWeightedSum8BitsRowwise, 
    SparseLengths8BitsRowwiseOp<CPUContext, 1>
}

num_inputs!{SparseLengthsWeightedSum8BitsRowwise, 5}

num_outputs!{SparseLengthsWeightedSum8BitsRowwise, 1}

inputs!{SparseLengthsWeightedSum8BitsRowwise, 
    0 => ("DATA",          "uint8 tensor obtained with operator FloatToRowwiseQuantized8Bits"),
    1 => ("SCALARS",       "Scalar multipliers for the input slices. Must be a vector with the length matching the length of INDICES"),
    2 => ("INDICES",       "Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated"),
    3 => ("LENGTHS",       "Vector with the same sum of elements as the first dimension of DATA"),
    4 => ("scale_bias",    "Matrix of floats, each row r_i of which stores a pair s_i, b_i -- scale and bias for i-th row")
}

outputs!{SparseLengthsWeightedSum8BitsRowwise, 
    0 => ("output", "output")
}

no_gradient!{SparseLengthsWeightedSum8BitsRowwise}

