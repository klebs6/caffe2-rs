crate::ix!();

pub struct CudnnConvOpBase {

    base: ConvPoolOpBase<CUDAContext>,

    cudnn_input_dims:       Vec<i64>,
    cudnn_filter_dims:      Vec<i64>,
    cudnn_wrapper:          CudnnWrapper,
    bottom_desc:            CudnnTensorDescriptor,
    filter_desc:            CudnnFilterDescriptor,
    bias_desc:              CudnnTensorDescriptor,
    top_desc:               CudnnTensorDescriptor,

    /**
      | top desc for bias add in case we do group
      | convolution
      |
      */
    top_desc_for_bias:      CudnnTensorDescriptor,

    conv_desc:              CudnnConvolutionDescriptor,
    cudnn_ws_nbytes_limit:  usize,
    cudnn_ws_nbytes:        usize,
    exhaustive_search:      bool,
    deterministic:          bool,
    cudnn_state:            usize,

    /// stored as FWD, dFILTER, dDATA
    force_algo:             Vec<i32>,

    enable_tensor_core:     bool,
    compute_type:           CudnnDataType,
}

impl CudnnConvOpBase {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : ConvPoolOpBase<CUDAContext>(operator_def, ws),
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

        CHECK(!deterministic_ || !exhaustive_search_);
        CAFFE_ENFORCE(group_ > 0);
        CAFFE_ENFORCE(!deterministic_ || !exhaustive_search_);
        for (int i = 0; i < kernel_.size(); ++i) {
          OPERATOR_NEEDS_FEATURE(
              pads_[i] == pads_[kernel_.size() + i],
              "The current padding scheme leads to unequal padding on the left "
              "and right, which is not supported by cudnn.");
        }
        // dilated convolution supported by some algorithms in cuDNN v6
    #if !(CUDNN_VERSION_MIN(6, 0, 0))
        OPERATOR_NEEDS_FEATURE(
            dilation_h() == 1 && dilation_w() == 1,
            "The cudnn convolution does not support dilation yet.");
    #endif
        // dilated grouped convolution supported in cuDNN v7.1
    #if !(CUDNN_VERSION_MIN(7, 1, 0))
        if (group_ != 1) {
          for (int dim = 0; dim < kernel_.size(); ++dim) {
            OPERATOR_NEEDS_FEATURE(
                dilation_[dim] == 1,
                "When group is used, dilation should not be set at the same time.");
          }
        }
    #endif

    #if CUDNN_VERSION_MIN(7, 0, 0)
        // verify TensorCore math is supported
        enable_tensor_core_ &= TensorCoreAvailable();
    #else
        enable_tensor_core_ = false;
    #endif

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
        CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&bias_desc_));
        CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&top_desc_));
        CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&top_desc_for_bias_));
        CUDNN_ENFORCE(cudnnCreateConvolutionDescriptor(&conv_desc_));
        */
    }

    /**
      | A helper function to set up the tensor
      | Nd descriptor, depending on the order
      | the group and the type given.
      |
      */
    #[inline] pub fn set_tensor_nd_descriptor_with_group<T>(
        &mut self,
        size:          i32,
        tensor_desc:   CudnnTensorDescriptor,
        n:             i32,
        c:             i32,
        h:             i32,
        w:             i32,
        d:             i32) 
    {
        todo!();
        /*
            #if CUDNN_VERSION_MIN(7, 0, 0)
            const int CC = C;
        #else
            const int CC = C / group_;
        #endif
            switch (order_) {
              case StorageOrder::NHWC:
                if (size == 4) {
                  CUDNN_ENFORCE(cudnnSetTensor4dDescriptorEx(
                      tensorDesc,
                      cudnnTypeWrapper<T>::type,
                      N,
                      CC,
                      H,
                      W,
                      H * W * C,
                      1,
                      W * C,
                      C));
                } else {
                  vector<int> dims = {N, H, W, D, CC};
                  vector<int> strides = {H * W * D * CC, W * D * CC, D * CC, CC, 1};
                  CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(
                      tensorDesc,
                      cudnnTypeWrapper<T>::type,
                      size > 3 ? size : 4,
                      dims.data(),
                      strides.data()));
                }
                break;
              case StorageOrder::NCHW:
                if (size == 4) {
                  CUDNN_ENFORCE(cudnnSetTensor4dDescriptorEx(
                      tensorDesc,
                      cudnnTypeWrapper<T>::type,
                      N,
                      CC,
                      H,
                      W,
                      C * H * W,
                      H * W,
                      W,
                      1));
                } else {
                  vector<int> dims = {N, CC, H, W, D};
                  vector<int> strides = {CC * H * W * D, H * W * D, W * D, D, 1};
                  CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(
                      tensorDesc,
                      cudnnTypeWrapper<T>::type,
                      size > 3 ? size : 4,
                      dims.data(),
                      strides.data()));
                }
                break;
              default:
                LOG(FATAL) << "Unknown storage order: " << order_;
            }
        */
    }
    
    #[inline] pub fn duplicate_conv_desc(
        &mut self, 
        input:          CudnnConvolutionDescriptor,
        kernel_dims:    usize,
        dilation_dims:  usize,
        copy:           CudnnConvolutionDescriptor)  
    {
        todo!();
        /*
            if (kernelDims == 1 || kernelDims == 2) {
          cudnnConvolutionMode_t mode;
          cudnnDataType_t dataType;
          int pad_height = 0;
          int pad_width = 0;
          int stride_height = 0;
          int stride_width = 0;
          int dilation_height = 0;
          int dilation_width = 0;

    #if CUDNN_VERSION_MIN(6, 0, 0)
          CUDNN_ENFORCE(cudnnGetConvolution2dDescriptor(
              input,
              &pad_height,
              &pad_width,
              &stride_height,
              &stride_width,
              &dilation_height,
              &dilation_width,
              &mode,
              &dataType));
    #else
          CUDNN_ENFORCE(cudnnGetConvolution2dDescriptor(
              input,
              &pad_height,
              &pad_width,
              &stride_height,
              &stride_width,
              &dilation_height,
              &dilation_width,
              &mode));
    #endif

    #if CUDNN_VERSION_MIN(6, 0, 0)
          CUDNN_ENFORCE(cudnnSetConvolution2dDescriptor(
              copy,
              pad_height,
              pad_width,
              stride_height,
              stride_width,
              dilation_height,
              dilation_width,
              mode,
              dataType));
    #else
          CUDNN_ENFORCE(cudnnSetConvolution2dDescriptor(
              copy,
              pad_height,
              pad_width,
              stride_height,
              stride_width,
              dilation_height,
              dilation_width,
              mode));
    #endif
        } else {
          cudnnConvolutionMode_t mode;
          cudnnDataType_t dataType;
          int arrayLength = 0;
          vector<int> ones(dilationDims, 1);
          CUDNN_ENFORCE(cudnnGetConvolutionNdDescriptor(
              input,
              kernel_.size(),
              &arrayLength,
              pads_.data(),
              stride_.data(),
              ones.data(),
              &mode,
              &dataType));

          CUDNN_ENFORCE(cudnnSetConvolutionNdDescriptor(
              copy,
              kernel_.size(),
              pads_.data(),
              stride_.data(),
              ones.data(),
              mode,
              dataType));
        }
        */
    }

    #[inline] pub fn determine_compute_type_from_input<T>(&mut self, x: &T) -> CudnnDataType {
        todo!();
        /*
            const cudaDeviceProp& prop = GetDeviceProperty(0);
            cudnnDataType_t computeType = CUDNN_DATA_FLOAT;
            if (X.template IsType<at::Half>()) {
              if (float16_compute_ && prop.major >= 6) {
                VLOG(1) << "CUDNN Convolution: float16_compute specified and "
                        << "supported, input data is Half - using Half "
                        << "compute.";
                computeType = CUDNN_DATA_HALF;
              } else if (float16_compute_) {
                VLOG(1) << "CUDNN Convolution: float16_compute specified but"
                        << "not supported, input data is Half - using float32 "
                        << "compute.";
              } else {
                VLOG(1) << "CUDNN Convolution: float16_compute not specified but "
                        << "input data is Half - using float32 compute.";
              }
            } else {
              VLOG(1) << "CUDNN Convolution: using float32 compute.";
            }
            return computeType;
        */
    }
    
    #[inline] pub fn set_conv_desc_from_arguments(&mut self)  {
        
        todo!();
        /*
            #if CUDNN_VERSION_MIN(6, 0, 0)
        if (kernel_.size() == 1 || kernel_.size() == 2) {
          CUDNN_ENFORCE(cudnnSetConvolution2dDescriptor(
              conv_desc_,
              pad_t(),
              kernel_.size() == 1 ? 0 : pad_l(),
              stride_h(),
              kernel_.size() == 1 ? 1 : stride_w(),
              dilation_h(),
              kernel_.size() == 1 ? 1 : dilation_w(),
              CUDNN_CROSS_CORRELATION,
              compute_type_));
        } else {
          CUDNN_ENFORCE(cudnnSetConvolutionNdDescriptor(
              conv_desc_,
              kernel_.size(),
              pads_.data(),
              stride_.data(),
              dilation_.data(),
              CUDNN_CROSS_CORRELATION,
              compute_type_));
        }
    #else
        if (kernel_.size() == 2) {
          CUDNN_ENFORCE(cudnnSetConvolution2dDescriptor(
              conv_desc_,
              pad_t(),
              pad_l(),
              stride_h(),
              stride_w(),
              1,
              1,
              CUDNN_CROSS_CORRELATION));
        } else {
          vector<int> ones(dilation_.size(), 1);
          CUDNN_ENFORCE(cudnnSetConvolutionNdDescriptor(
              conv_desc_,
              kernel_.size(),
              pads_.data(),
              stride_.data(),
              ones.data(),
              CUDNN_CROSS_CORRELATION,
              compute_type_));
        }
    #endif
        */
    }
    
    #[inline] pub fn set_conv_desc_compute_type(
        &mut self, 
        conv_desc: CudnnConvolutionDescriptor,
        math: CudnnDataType)  
    {
        
        todo!();
        /*
            if (kernel_.size() == 2) {
          cudnnConvolutionMode_t mode;
          cudnnDataType_t dataType;
          int pad_height = 0;
          int pad_width = 0;
          int stride_height = 0;
          int stride_width = 0;
          int dilation_height = 0;
          int dilation_width = 0;

    #if CUDNN_VERSION_MIN(6, 0, 0)
          CUDNN_ENFORCE(cudnnGetConvolution2dDescriptor(
              conv_desc,
              &pad_height,
              &pad_width,
              &stride_height,
              &stride_width,
              &dilation_height,
              &dilation_width,
              &mode,
              &dataType));
    #else
          CUDNN_ENFORCE(cudnnGetConvolution2dDescriptor(
              conv_desc,
              &pad_height,
              &pad_width,
              &stride_height,
              &stride_width,
              &dilation_height,
              &dilation_width,
              &mode));
    #endif

    #if CUDNN_VERSION_MIN(6, 0, 0)
          CUDNN_ENFORCE(cudnnSetConvolution2dDescriptor(
              conv_desc,
              pad_height,
              pad_width,
              stride_height,
              stride_width,
              dilation_height,
              dilation_width,
              mode,
              math));
    #else
          CUDNN_ENFORCE(cudnnSetConvolution2dDescriptor(
              conv_desc,
              pad_height,
              pad_width,
              stride_height,
              stride_width,
              dilation_height,
              dilation_width,
              mode));
    #endif
        } else {
          cudnnConvolutionMode_t mode;
          cudnnDataType_t dataType;
          int arrayLength = 0;
          vector<int> ones(dilation_.size(), 1);
          CUDNN_ENFORCE(cudnnGetConvolutionNdDescriptor(
              conv_desc,
              kernel_.size(),
              &arrayLength,
              pads_.data(),
              stride_.data(),
              ones.data(),
              &mode,
              &dataType));

          CUDNN_ENFORCE(cudnnSetConvolutionNdDescriptor(
              conv_desc,
              kernel_.size(),
              pads_.data(),
              stride_.data(),
              ones.data(),
              mode,
              math));
        }
        */
    }
}

impl Drop for CudnnConvOpBase {
    fn drop(&mut self) {
        todo!();
        /*
           CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(bottom_desc_));
           CUDNN_ENFORCE(cudnnDestroyFilterDescriptor(filter_desc_));
           CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(bias_desc_));
           CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(top_desc_));
           CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(top_desc_for_bias_));
           CUDNN_ENFORCE(cudnnDestroyConvolutionDescriptor(conv_desc_));
           */
    }
}

