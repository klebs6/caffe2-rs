crate::ix!();

/**
  | Variation of SparseLengthsSum operator,
  | where DATA is stored using 8bits.
  | 
  | DATA was quantized with 8Bit row-wise
  | quantization (see doc to FloatToRowwiseQuantized8Bits
  | operator).
  | 
  | To restore DATA from 8Bit, we use additional
  | input that stores scales and biases.
  |
  */
register_cpu_operator!{SparseLengthsSum8BitsRowwise, SparseLengths8BitsRowwiseOp<CPUContext>}

num_inputs!{SparseLengthsSum8BitsRowwise, 4}

num_outputs!{SparseLengthsSum8BitsRowwise, 1}

inputs!{SparseLengthsSum8BitsRowwise, 
    0 => ("DATA",         "uint8 tensor obtained with operator FloatToRowwiseQuantized8Bits"),
    1 => ("INDICES",      "Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated"),
    2 => ("LENGTHS",      "Vector with the same sum of elements as the first dimension of DATA"),
    3 => ("scale_bias",   "Matrix of floats, each row r_i of which stores a pair s_i, b_i -- scale and bias for i-th row")
}

outputs!{SparseLengthsSum8BitsRowwise, 
    0 => ("output", "output")
}

no_gradient!{SparseLengthsSum8BitsRowwise}

