crate::ix!();

/**
  | Performs SparseLengthsMean, but operating
  | on 2-bit rowwise quantized matrices
  | with fused storage (where each row stores
  | quantized values, and then 2-byte fp16
  | scale and bias), and where rows are pruned.
  |
  */
register_cpu_operator!{
    SparseLengthsMean2BitRowwiseSparse,
    SparseLengthsNBitRowwiseSparseOp<
        2,
        /*with_weights=*/false,
        /*is_mean=*/true>
}

num_inputs!{SparseLengthsMean2BitRowwiseSparse, 4}

num_outputs!{SparseLengthsMean2BitRowwiseSparse, 1}

inputs!{SparseLengthsMean2BitRowwiseSparse, 
    0 => ("DATA", "uint8 tensor obtained with operator FloatToFused2BitRowwiseQuantized"),
    1 => ("INDICES", "Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated"),
    2 => ("LENGTHS", "Vector with the same sum of elements as the first dimension of DATA"),
    3 => ("COMPRESSED_INDICES_MAPPING", "Integer vector mapping uncompressed indices to compressed indices")
}

outputs!{SparseLengthsMean2BitRowwiseSparse, 
    0 => ("output", "output")
}

value_key_length_input_fillers!{
    /*
    SparseLengthsMean2BitRowwiseSparse, (
        SparseLengthsNBitRowwiseSparseOp<2, false, true>::COMPRESSED_INDICES_MAPPING,
        SparseLengthsNBitRowwiseSparseOp<2, false, true>::INDICES,
        SparseLengthsNBitRowwiseSparseOp<2, false, true>::LENGTHS)

        */
}

no_gradient!{SparseLengthsMean2BitRowwiseSparse}

export_caffe2_op_to_c10_cpu!{
    SparseLengthsSum8BitRowwiseSparse,
    "_caffe2::SparseLengthsSum8BitRowwiseSparse(
        Tensor data, 
        Tensor indices, 
        Tensor lengths, 
        Tensor compressed_indices_mapping) -> Tensor output",
        SparseLengthsNBitRowwiseSparseOp::<8>
}
