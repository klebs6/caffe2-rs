crate::ix!();

#[USE_OPERATOR_FUNCTIONS("CUDAContext")]
pub struct CudnnOrderSwithOpBase {
    storage:       OperatorStorage,
    context:       CUDAContext,
    cudnn_wrapper: CudnnWrapper,
    x_desc:        CudnnTensorDescriptor,
    y_desc:        CudnnTensorDescriptor,
    cached_X_dims: Vec<i32>,
}

impl CudnnOrderSwithOpBase {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<CUDAContext>(std::forward<Args>(args)...),
            cudnn_wrapper_(&context_) 

        CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&X_desc_));
        CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&Y_desc_));
        */
    }
}

impl Drop for CudnnOrderSwithOpBase {

    fn drop(&mut self) {
        todo!();
        /* 
        CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(X_desc_));
        CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(Y_desc_));
       */
    }
}

impl CudnnOrderSwithOpBase {
    
    /// TODO: std::vector<int> -> std::vector<int64_t>
    #[inline] pub fn set_tensor_descriptor(&self, 
        data_type: CudnnDataType,
        order:     StorageOrder,
        data_dims: &Vec<i32>,
        data_desc: CudnnTensorDescriptor)
    {
        
        todo!();
        /*
            const int ndim = data_dims.size();
        const int N = data_dims[0];
        const int C = order == StorageOrder::NCHW ? data_dims[1] : data_dims.back();
        if (ndim == 3) {
          const int H = 1;
          const int W = order == StorageOrder::NCHW ? data_dims[2] : data_dims[1];
          CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
              data_desc, GetCudnnTensorFormat(order), data_type, N, C, H, W));
        } else if (ndim == 4) {
          const int H = order == StorageOrder::NCHW ? data_dims[2] : data_dims[1];
          const int W = order == StorageOrder::NCHW ? data_dims[3] : data_dims[2];
          CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
              data_desc, GetCudnnTensorFormat(order), data_type, N, C, H, W));
        } else {
          const int H = order == StorageOrder::NCHW ? data_dims[2] : data_dims[1];
          const int W = order == StorageOrder::NCHW ? data_dims[3] : data_dims[2];
          const auto l_iter = order == StorageOrder::NCHW ? data_dims.cbegin() + 4
                                                          : data_dims.cbegin() + 3;
          const auto r_iter =
              order == StorageOrder::NCHW ? data_dims.cend() : data_dims.cend() - 1;
          const int D = std::accumulate(l_iter, r_iter, 1, std::multiplies<int>());
          const std::array<int, 5> dims = {N, C, H, W, D};
          const std::array<int, 5> strides = order == StorageOrder::NCHW
              ? std::array<int, 5>{C * H * W * D, H * W * D, W * D, D, 1}
              : std::array<int, 5>{C * H * W * D, 1, W * D * C, D * C, C};
          CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(
              data_desc, data_type, 5, dims.data(), strides.data()));
        }
        */
    }
}
