crate::ix!();

pub struct CudnnConvTransposeOpBase {
    base:                      ConvTransposeUnpoolBase<CUDAContext>,
    cudnn_input_dims:          Vec<i64>,
    cudnn_filter_dims:         Vec<i64>,
    cudnn_wrapper:             CudnnWrapper,
    bottom_desc:               CudnnTensorDescriptor,
    filter_desc:               CudnnFilterDescriptor,
    bias_desc:                 CudnnTensorDescriptor,
    top_desc:                  CudnnTensorDescriptor,
    top_desc_for_bias:         CudnnTensorDescriptor,
    conv_desc:                 CudnnConvolutionDescriptor,
    cudnn_ws_nbytes_limit:     usize,
    cudnn_ws_nbytes:           usize,
    exhaustive_search:         bool,
    deterministic:             bool,
    cudnn_state:               usize,

    // stored as FWD, dFILTER, dDATA
    force_algo:                Vec<i32>,
    enable_tensor_core:        bool,
}

impl Drop for CudnnConvTransposeOpBase {

    fn drop(&mut self) {
        todo!();
        /*
        CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(bottom_desc_));
        CUDNN_ENFORCE(cudnnDestroyFilterDescriptor(filter_desc_));
        if (InputSize() == 3) {
          CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(bias_desc_));
          CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(top_desc_for_bias_));
        }
        CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(top_desc_));
        CUDNN_ENFORCE(cudnnDestroyConvolutionDescriptor(conv_desc_));
        */
    }
}

impl CudnnConvTransposeOpBase {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : ConvTransposeUnpoolBase<CUDAContext>(std::forward<Args>(args)...),
            cudnn_wrapper_(&context_),
            cudnn_ws_nbytes_limit_(OperatorStorage::GetSingleArgument<size_t>(
                "ws_nbytes_limit",
                kCONV_CUDNN_WORKSPACE_LIMIT_BYTES)),
            exhaustive_search_(
                OperatorStorage::GetSingleArgument<int>("exhaustive_search", 0)),
            deterministic_(
                OperatorStorage::GetSingleArgument<int>("deterministic", 0)),
            cudnn_state_(OperatorStorage::GetSingleArgument<int>("cudnn_state", 0)),
            force_algo_(OperatorStorage::GetRepeatedArgument<int>(
                "force_algo",
                vector<int>{-1, -1, -1})),
            enable_tensor_core_(
                OperatorStorage::GetSingleArgument<bool>("enable_tensor_core", 1)) 

        CAFFE_ENFORCE(!deterministic_ || !exhaustive_search_);

        bool individual_force_algo = OperatorStorage::HasArgument("force_algo_fwd") ||
            OperatorStorage::HasArgument("force_algo_dgrad") ||
            OperatorStorage::HasArgument("force_algo_wgrad");
        if (OperatorStorage::HasArgument("force_algo")) {
          CAFFE_ENFORCE(
              !individual_force_algo,
              "Cannot specify both force_algo and any of",
              "force_algo_fwd, force_algo_dgrad, force_algo_wgrad");
        } else {
          force_algo_ = std::vector<int>{-1, -1, -1};
          force_algo_[ALGO_FWD] =
              OperatorStorage::GetSingleArgument<int>("force_algo_fwd", -1);
          force_algo_[ALGO_DGRAD] =
              OperatorStorage::GetSingleArgument<int>("force_algo_dgrad", -1);
          force_algo_[ALGO_WGRAD] =
              OperatorStorage::GetSingleArgument<int>("force_algo_wgrad", -1);
        }

        CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&bottom_desc_));
        CUDNN_ENFORCE(cudnnCreateFilterDescriptor(&filter_desc_));
        if (InputSize() == 3) {
          CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&bias_desc_));
          CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&top_desc_for_bias_));
        }
        CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&top_desc_));
        CUDNN_ENFORCE(cudnnCreateConvolutionDescriptor(&conv_desc_));
        */
    }
    
    #[inline] pub fn set_tensor_4ddescriptor_with_group(
        &self, 
        data_type:  CudnnDataType,
        n:          i32,
        c:          i32,
        h:          i32,
        w:          i32,
        desc:       *mut CudnnTensorDescriptor)  
    {
        
        todo!();
        /*
            #if CUDNN_VERSION_MIN(7, 0, 0)
        const int CC = C;
    #else
        const int CC = C / group_;
    #endif
        switch (order_) {
          case StorageOrder::NCHW: {
            CUDNN_ENFORCE(cudnnSetTensor4dDescriptorEx(
                *desc, data_type, N, CC, H, W, C * H * W, H * W, W, 1));
            break;
          }
          case StorageOrder::NHWC: {
            CUDNN_ENFORCE(cudnnSetTensor4dDescriptorEx(
                *desc, data_type, N, CC, H, W, H * W * C, 1, W * C, C));
            break;
          }
          default: {
            LOG(FATAL) << "Unknown storage order: " << order_;
          }
        }
        */
    }
}
