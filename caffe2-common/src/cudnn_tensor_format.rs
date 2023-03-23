crate::ix!();

/**
  | A wrapper function to convert the Caffe
  | storage order to cudnn storage order
  | enum values.
  |
  */
#[inline] pub fn get_cudnn_tensor_format(order: &StorageOrder) -> CudnnTensorFormat {
    
    todo!();
    /*
        switch (order) {
        case StorageOrder::NHWC:
          return CUDNN_TENSOR_NHWC;
        case StorageOrder::NCHW:
          return CUDNN_TENSOR_NCHW;
        default:
          LOG(FATAL) << "Unknown cudnn equivalent for order: " << order;
      }
      // Just to suppress compiler warnings
      return CUDNN_TENSOR_NCHW;
    */
}
