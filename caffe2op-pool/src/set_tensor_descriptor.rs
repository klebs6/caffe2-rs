crate::ix!();

#[inline] pub fn set_tensor_descriptor(
    data_type: CudnnDataType,
    order:     StorageOrder,
    dims:      &Vec<i64>,
    desc:      *mut CudnnTensorDescriptor)  
{
    
    todo!();
    /*
        const int ndim = dims.size();
      const int N = dims[0];
      const int C = order == StorageOrder::NCHW ? dims[1] : dims[ndim - 1];
      switch (ndim) {
        case 4: {
          const int H = order == StorageOrder::NCHW ? dims[2] : dims[1];
          const int W = order == StorageOrder::NCHW ? dims[3] : dims[2];
          CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
              *desc, GetCudnnTensorFormat(order), data_type, N, C, H, W));
          break;
        }
        case 5: {
          const int D = order == StorageOrder::NCHW ? dims[2] : dims[1];
          const int H = order == StorageOrder::NCHW ? dims[3] : dims[2];
          const int W = order == StorageOrder::NCHW ? dims[4] : dims[3];
          const std::array<int, 5> dims_arr = {N, C, D, H, W};
          const std::array<int, 5> strides_arr = order == StorageOrder::NCHW
              ? std::array<int, 5>{C * D * H * W, D * H * W, H * W, W, 1}
              : std::array<int, 5>{D * H * W * C, 1, H * W * C, W * C, C};
          CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(
              *desc, data_type, 5, dims_arr.data(), strides_arr.data()));
          break;
        }
        default: {
          CAFFE_THROW("Unsupported tensor dim: ", ndim);
          break;
        }
      }
    */
}
