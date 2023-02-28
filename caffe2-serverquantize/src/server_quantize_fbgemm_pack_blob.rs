crate::ix!();

use crate::{
    Tensor,
    PackedDepthWiseConvMatrix,
    PackWeightMatrixForGConv,
    PackWeightMatrixForGConvB,
    PackBMatrix,
    TensorQuantizationParams,
    CompressedSparseColumn,
};

/**
  | Packed weight matrix for DNNLOWP Int8FC
  | operator
  |
  */
pub struct Int8FCDNNLowPPackedWeightBlob {

    qparams:               Vec<TensorQuantizationParams>,
    column_offsets:        Arc<Vec<i32>>,

    /**
      | The original tensor before packing
      | but only with meta information
      |
      */
    original_tensor:       Tensor, // default = CPU

    bias:                  Arc<Vec<i32>>,

    /// Only for 32-bit accumulation
    w:                     Arc<PackBMatrix<i8>>,

    /**
      | Only for 16-bit accumulation
      | 
      | Dense matrix holding common values
      |
      */
    w_acc16:               Arc<PackBMatrix<i8,i16>>,

    /// Sparse matrix holding outliers
    w_outlier:             Arc<CompressedSparseColumn>,

    nbits_in_non_outlier:  i32,
}

/**
  | Packed weight matrix for DNNLOWP Int8Conv
  | operator
  |
  */
pub struct Int8ConvDNNLowPPackedWeightBlob {

    base: Int8FCDNNLowPPackedWeightBlob,

    // Only for 32-bit accumulation
    w_depthwise:  Arc<PackedDepthWiseConvMatrix>,
    w_gconv:      Arc<PackWeightMatrixForGConv<i8>>,
    w_gconv3d:    Arc<PackWeightMatrixForGConvB<i8, i32, 3>>,
}
