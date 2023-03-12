crate::ix!();

pub struct CudnnConvTransposeOp<T> {
    base:            CudnnConvTransposeOpBase,
    data_algo_cache: AlgorithmsCache<CudnnConvolutionBwdDataAlgo>,
    bwd_data_algo:   CudnnConvolutionBwdDataAlgo,

    // Input: X, W, b
    //
    // Output: Y
    //
    phantom: PhantomData<T>,
}

impl<T> CudnnConvTransposeOp<T> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : CudnnConvTransposeOpBase(std::forward<Args>(args)...)
        */
    }

    #[inline] pub fn run_on_device(&mut self) -> bool {
        todo!();
        /*
            auto& X = Input(INPUT);
          auto& filter = Input(FILTER);
          int C = 0;
          switch (order_) {
            case StorageOrder::NHWC:
              C = filter.dim32(3) * group_;
              break;
            case StorageOrder::NCHW:
              C = filter.dim32(1) * group_;
              break;
            default:
              LOG(FATAL) << "Unknown storage order: " << order_;
          }

          auto sizes = ConvTransposeUnpoolBase<CUDAContext>::GetOutputSize(X, C);
          auto* Y = Output(0, sizes, at::dtype<T>());

          if (X.numel() == 0) {
            VLOG(2) << "Number on elements is 0 in CudnnConvTransposeOp";
            return true;
          }

          int N = 0, M = 0, H = 0, W = 0, H_out = 0, W_out = 0;
          switch (order_) {
            case StorageOrder::NHWC:
              N = X.dim32(0);
              H = X.dim32(1);
              W = X.dim32(2);
              M = X.dim32(3);
              H_out = Y->dim32(1);
              W_out = Y->dim32(2);
              CAFFE_ENFORCE_EQ(filter.dim32(1), kernel_h());
              CAFFE_ENFORCE_EQ(filter.dim32(2), kernel_w());
              CAFFE_ENFORCE_EQ(filter.dim32(3), C / group_);
              break;
            case StorageOrder::NCHW:
              N = X.dim32(0);
              M = X.dim32(1);
              H = X.dim32(2);
              W = X.dim32(3);
              H_out = Y->dim32(2);
              W_out = Y->dim32(3);
              CAFFE_ENFORCE_EQ(filter.dim32(1), C / group_);
              CAFFE_ENFORCE_EQ(filter.dim32(2), kernel_h());
              CAFFE_ENFORCE_EQ(filter.dim32(3), kernel_w());
              break;
            default:
              LOG(FATAL) << "Unknown storage order: " << order_;
          }
          CAFFE_ENFORCE_EQ(M % group_, 0);

          if (InputSize() == 3) {
            auto& bias = Input(BIAS);
            CAFFE_ENFORCE_EQ(bias.dim(), 1);
            CAFFE_ENFORCE_EQ(bias.dim32(0), C);
          }

          // Set up the cudnn algorithms & workspace if necessary
          const bool input_changed = (X.sizes() != cudnn_input_dims_);
          const bool filter_changed = (filter.sizes() != cudnn_filter_dims_);

          if (input_changed || filter_changed) {
            VLOG(1) << "Changing the cudnn descriptor configurations.";
            if (input_changed) {
              cudnn_input_dims_ = X.sizes().vec();
              SetTensor4DDescriptorWithGroup(
                  cudnnTypeWrapper<T>::type, N, M, H, W, &bottom_desc_);
            }
            if (filter_changed) {
              cudnn_filter_dims_ = filter.sizes().vec();
        #if CUDNN_VERSION_MIN(7, 0, 0)
              const int MM = M;
        #else
              const int MM = M / group_;
        #endif
              CUDNN_ENFORCE(cudnnSetFilter4dDescriptor(
                  filter_desc_,
                  cudnnTypeWrapper<T>::type,
                  GetCudnnTensorFormat(order_),
                  MM,
                  C / group_,
                  kernel_h(),
                  kernel_w()));
              if (InputSize() == 3) {
                CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
                    bias_desc_,
                    GetCudnnTensorFormat(order_),
                    cudnnTypeWrapper<T>::type,
                    1,
                    C,
                    1,
                    1));
              }
            }
            // Set the output
            SetTensor4DDescriptorWithGroup(
                cudnnTypeWrapper<T>::type, N, C, H_out, W_out, &top_desc_);
            if (InputSize() == 3) {
              CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
                  top_desc_for_bias_,
                  GetCudnnTensorFormat(order_),
                  cudnnTypeWrapper<T>::type,
                  N,
                  C,
                  H_out,
                  W_out));
            }

            // Set the convolution descriptor
            CAFFE_ENFORCE_EQ(
                pad_t(),
                pad_b(),
                "The current padding scheme leads to unequal padding on the top and "
                "bottom, which is not supported by cudnn.");
            CAFFE_ENFORCE_EQ(
                pad_l(),
                pad_r(),
                "The current padding scheme leads to unequal padding on the left "
                "and right, which is not supported by cudnn.");
            // Set the convolution descriptor
        #if CUDNN_VERSION_MIN(6, 0, 0)
            CUDNN_ENFORCE(cudnnSetConvolution2dDescriptor(
                conv_desc_,
                pad_t(),
                pad_l(),
                stride_h(),
                stride_w(),
                1,
                1,
                CUDNN_CROSS_CORRELATION,
                cudnnTypeWrapper<T>::type));
        #else
            CUDNN_ENFORCE(cudnnSetConvolution2dDescriptor(
                conv_desc_,
                pad_t(),
                pad_l(),
                stride_h(),
                stride_w(),
                1,
                1,
                CUDNN_CROSS_CORRELATION));
        #endif

        #if CUDNN_VERSION_MIN(7, 0, 0)
            // enable TensorCore math if desired
            enable_tensor_core_ &= TensorCoreAvailable();
            if (enable_tensor_core_) {
              CUDNN_ENFORCE(
                  cudnnSetConvolutionMathType(conv_desc_, CUDNN_TENSOR_OP_MATH));
            }
            // set cuDNN groups if appropriate
            CUDNN_ENFORCE(cudnnSetConvolutionGroupCount(conv_desc_, group_));
        #endif

            if (force_algo_[ALGO_DGRAD] >= 0) {
              bwd_data_algo_ = (cudnnConvolutionBwdDataAlgo_t)force_algo_[ALGO_DGRAD];
            } else if (deterministic_) {
              bwd_data_algo_ = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
            } else if (exhaustive_search_) {
              bwd_data_algo_ =
                  data_algo_cache_.getAlgorithm(X.sizes(), filter.sizes(), 0, [&]() {
                    int returned_algo_count;
                    std::array<
                        cudnnConvolutionBwdDataAlgoPerf_t,
                        kNUM_CUDNN_BWD_DATA_ALGS>
                        data_perf_stat;
                    cudnn_wrapper_.with_cudnn_state(
                        cudnn_state_, [&](CudnnState* state) {
                          state->workspace().reset();
                          CUDNN_ENFORCE(cudnnFindConvolutionBackwardDataAlgorithm(
                              state->cudnn_handle(),
                              filter_desc_,
                              bottom_desc_,
                              conv_desc_,
                              top_desc_,
                              kNUM_CUDNN_BWD_DATA_ALGS,
                              &returned_algo_count,
                              data_perf_stat.data()));
                        });

                    LogCudnnPerfStats(data_perf_stat, returned_algo_count);
                    return data_perf_stat[0].algo;
                  });
            } else {
              constexpr int nalgo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT;
              int valid_algos;
              cudnnConvolutionBwdDataAlgoPerf_t algos[nalgo];
              CUDNN_ENFORCE(cudnnGetConvolutionBackwardDataAlgorithm_v7(
                  cudnn_wrapper_.inline_cudnn_handle(),
                  filter_desc_,
                  bottom_desc_,
                  conv_desc_,
                  top_desc_,
                  nalgo,
                  &valid_algos,
                  algos));
              bool found = false;
              for (int i = 0; i < valid_algos; i++) {
                auto a = algos[i];
                if (a.memory <= cudnn_ws_nbytes_limit_) {
                  bwd_data_algo_ = a.algo;
                  found = true;
                  break;
                }
              }
              CAFFE_ENFORCE(found, "Unable to find algorithms for cuDNN backward data");
            }

            size_t bwd_data_ws_size;
            CUDNN_ENFORCE(cudnnGetConvolutionBackwardDataWorkspaceSize(
                cudnn_wrapper_.inline_cudnn_handle(),
                filter_desc_,
                bottom_desc_,
                conv_desc_,
                top_desc_,
                bwd_data_algo_,
                &bwd_data_ws_size));
            cudnn_ws_nbytes_ = bwd_data_ws_size;
            VLOG(1) << "Cudnn algorithm: " << bwd_data_algo_;
            VLOG(1) << "Cudnn workspace size: " << bwd_data_ws_size;
          }

          const T* X_data = X.template data<T>();
          const T* filter_data = filter.template data<T>();
          T* Y_data = Y->template mutable_data<T>();

          // Now, actually run the computation.
          // Filter
        #if CUDNN_VERSION_MIN(7, 0, 0)
          cudnn_wrapper_.with_cudnn_state(cudnn_state_, [&](CudnnState* state) {
            CUDNN_ENFORCE(cudnnConvolutionBackwardData(
                state->cudnn_handle(),
                cudnnTypeWrapper<T>::kOne(),
                filter_desc_,
                filter_data,
                bottom_desc_,
                X_data,
                conv_desc_,
                bwd_data_algo_,
                state->workspace().get(cudnn_ws_nbytes_),
                cudnn_ws_nbytes_,
                cudnnTypeWrapper<T>::kZero(),
                top_desc_,
                Y_data));
          });
        #else
          const int X_HxW = H * W;
          const int Y_HxW = H_out * W_out;
          const int group_offset_X =
              order_ == StorageOrder::NCHW ? M / group_ * X_HxW : M / group_;
          const int group_offset_Y =
              order_ == StorageOrder::NCHW ? C / group_ * Y_HxW : C / group_;
          const int group_offset_filter = filter.numel() / group_;
          for (int i = 0; i < group_; ++i) {
            cudnn_wrapper_.with_cudnn_state(cudnn_state_, [&](CudnnState* state) {
              CUDNN_ENFORCE(
                  cudnnConvolutionBackwardData(state->cudnn_handle(),
                                               cudnnTypeWrapper<T>::kOne(),
                                               filter_desc_,
                                               filter_data + i * group_offset_filter,
                                               bottom_desc_,
                                               X_data + i * group_offset_X;
                                               conv_desc_,
                                               bwd_data_algo_,
                                               state->workspace().get(cudnn_ws_nbytes_),
                                               cudnn_ws_nbytes_,
                                               cudnnTypeWrapper<T_DX>::kZero(),
                                               top_desc_,
                                               Y_data + i * group_offset_Y));
            });
          }
        #endif
          // Bias
          if (InputSize() == 3) {
            CUDNN_ENFORCE(cudnnAddTensor(
                cudnn_wrapper_.inline_cudnn_handle(),
                cudnnTypeWrapper<T>::kOne(),
                bias_desc_,
                Input(BIAS).template data<T>(),
                cudnnTypeWrapper<T>::kOne(),
                top_desc_for_bias_,
                Y->template mutable_data<T>()));
          }
          // Done.
          return true;
        */
    }
}

input_tags!{
    CudnnConvTransposeOp {
        Input,
        Filter,
        Bias
    }
}
