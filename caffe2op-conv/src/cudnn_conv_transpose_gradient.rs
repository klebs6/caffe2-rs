crate::ix!();

pub struct CudnnConvTransposeGradientOp<T> {
    base:               CudnnConvTransposeOpBase,

    // input: X, W, dY
    // output: dW, optionally db and dX

    algo:               CudnnConvolutionFwdAlgo,
    bwd_filter_algo:    CudnnConvolutionBwdFilterAlgo,
    no_bias:            bool,

    forward_algo_cache: AlgorithmsCache<CudnnConvolutionFwdAlgo>,
    filter_algo_cache:  AlgorithmsCache<CudnnConvolutionBwdFilterAlgo>,
    phantom: PhantomData<T>,
}

input_tags!{
    CudnnConvTransposeGradientOp {
        Input,
        Filter,
        OutputGrad
    }
}

output_tags!{
    CudnnConvTransposeGradientOp {
        FilterGrad,
        BiasOrInputGrad,
        InputGrad
    }
}

impl<T> CudnnConvTransposeGradientOp<T> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : CudnnConvTransposeOpBase(std::forward<Args>(args)...),
            no_bias_(OperatorStorage::GetSingleArgument<bool>("no_bias", false)) 

        CAFFE_ENFORCE(
            !(no_bias_ && OutputSize() == 3),
            "If bias is not present, you should not have 3 grad output.");
        */
    }

    /// TODO(Yangqing): a lot of the function contents are very similar. Consider
    /// consolidating them.
    #[inline] pub fn run_on_device(&mut self) -> bool {
        todo!();
        /*
            const auto& X = Input(INPUT);
          const auto& filter = Input(FILTER);
          const auto& dY = Input(OUTPUT_GRAD);

          CAFFE_ENFORCE_EQ(X.dim(), 4);
          CAFFE_ENFORCE_EQ(filter.dim(), 4);
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

          int N = 0, M = 0, H = 0, W = 0, H_out = 0, W_out = 0;
          switch (order_) {
            case StorageOrder::NHWC:
              N = X.dim32(0);
              H = X.dim32(1);
              W = X.dim32(2);
              M = X.dim32(3);
              H_out = dY.dim32(1);
              W_out = dY.dim32(2);
              CAFFE_ENFORCE_EQ(filter.dim32(1), kernel_h());
              CAFFE_ENFORCE_EQ(filter.dim32(1), kernel_h());
              CAFFE_ENFORCE_EQ(filter.dim32(2), kernel_w());
              CAFFE_ENFORCE_EQ(filter.dim32(3), C / group_);
              break;
            case StorageOrder::NCHW:
              N = X.dim32(0);
              M = X.dim32(1);
              H = X.dim32(2);
              W = X.dim32(3);
              H_out = dY.dim32(2);
              W_out = dY.dim32(3);
              CAFFE_ENFORCE_EQ(filter.dim32(1), C / group_);
              CAFFE_ENFORCE_EQ(filter.dim32(2), kernel_h());
              CAFFE_ENFORCE_EQ(filter.dim32(3), kernel_w());
              break;
            default:
              LOG(FATAL) << "Unknown storage order: " << order_;
          }
          CAFFE_ENFORCE_EQ(M % group_, 0);

          // Since we only handle LegacyPadding::NOTSET, we don't need to
          // compute padding.
          auto* dfilter = Output(FILTER_GRAD, filter.sizes(), at::dtype<T>());

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
              if (!no_bias_) {
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
            if (!no_bias_) {
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
            CUDNN_CHECK(cudnnSetConvolutionGroupCount(conv_desc_, group_));
        #endif
            if (force_algo_[ALGO_WGRAD] >= 0) {
              bwd_filter_algo_ =
                  (cudnnConvolutionBwdFilterAlgo_t)force_algo_[ALGO_WGRAD];
            } else if (deterministic_) {
              algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
              bwd_filter_algo_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
            } else if (exhaustive_search_) {
              bwd_filter_algo_ =
                  filter_algo_cache_.getAlgorithm(X.sizes(), filter.sizes(), 0, [&]() {
                    LOG(INFO) << "CUDNN Convolution bwd: doing exhaustive search.";
                    // When we do an exhaustive search, we will ignore the workspace
                    // size
                    // limit and simply go for the fastest algorithm. If you happen to
                    // run
                    // out of memory later, you will be on your own...
                    int returned_algo_count;
                    // We clean up the current workspace memory so that the forward
                    // algorithm
                    // is free to allocate memory.
                    // Actually run the search.
                    std::array<
                        cudnnConvolutionBwdFilterAlgoPerf_t,
                        kNUM_CUDNN_BWD_FILTER_ALGS>
                        filter_perf_stat;

                    cudnn_wrapper_.with_cudnn_state(
                        cudnn_state_, [&](CudnnState* state) {
                          state->workspace().reset();
                          CUDNN_ENFORCE(cudnnFindConvolutionBackwardFilterAlgorithm(
                              state->cudnn_handle(),
                              top_desc_,
                              bottom_desc_,
                              conv_desc_,
                              filter_desc_,
                              kNUM_CUDNN_BWD_FILTER_ALGS,
                              &returned_algo_count,
                              filter_perf_stat.data()));
                        });
                    LogCudnnPerfStats(filter_perf_stat, returned_algo_count);
                    return filter_perf_stat[0].algo;
                  });

              algo_ =
                  forward_algo_cache_.getAlgorithm(X.sizes(), filter.sizes(), 0, [&]() {
                    int returned_algo_count;
                    std::array<cudnnConvolutionFwdAlgoPerf_t, kNUM_CUDNN_FWD_ALGS>
                        fwd_perf_stat;
                    cudnn_wrapper_.with_cudnn_state(
                        cudnn_state_, [&](CudnnState* state) {
                          state->workspace().reset();
                          CUDNN_ENFORCE(cudnnFindConvolutionForwardAlgorithm(
                              state->cudnn_handle(),
                              top_desc_,
                              filter_desc_,
                              conv_desc_,
                              bottom_desc_,
                              kNUM_CUDNN_BWD_DATA_ALGS,
                              &returned_algo_count,
                              fwd_perf_stat.data()));
                        });

                    LogCudnnPerfStats(fwd_perf_stat, returned_algo_count);
                    return fwd_perf_stat[0].algo;
                  });
            } else {
              // choose backward algorithm for filter
              {
              constexpr int nalgo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT;
              int valid_algos;
              cudnnConvolutionBwdFilterAlgoPerf_t algos[nalgo];
              CUDNN_ENFORCE(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
                  cudnn_wrapper_.inline_cudnn_handle(),
                  top_desc_,
                  bottom_desc_,
                  conv_desc_,
                  filter_desc_,
                  nalgo,
                  &valid_algos,
                  algos));
              bool found = false;
              for (int i = 0; i < valid_algos; i++) {
                auto a = algos[i];
                if (a.memory <= cudnn_ws_nbytes_limit_) {
                  bwd_filter_algo_ = a.algo;
                  found = true;
                  break;
                }
              }
              CAFFE_ENFORCE(found, "Unable to find algorithms for cuDNN backward filter");
              }
              // choose backward algo for data
              {
              constexpr int nalgo = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
              int valid_algos;
              cudnnConvolutionFwdAlgoPerf_t algos[nalgo];
              CUDNN_ENFORCE(cudnnGetConvolutionForwardAlgorithm_v7(
                  cudnn_wrapper_.inline_cudnn_handle(),
                  top_desc_,
                  filter_desc_,
                  conv_desc_,
                  bottom_desc_,
                  nalgo,
                  &valid_algos,
                  algos));
              bool found = false;
              for (int i = 0; i < valid_algos; i++) {
                auto a = algos[i];
                if (a.memory <= cudnn_ws_nbytes_limit_) {
                  algo_ = a.algo;
                  found = true;
                  break;
                }
              }
              CAFFE_ENFORCE(found, "Unable to find algorithms for cuDNN forward");
              }
            }
            // get workspace for backwards filter algorithm
            size_t bwd_filter_ws_size, fwd_ws_size;
            CUDNN_ENFORCE(cudnnGetConvolutionBackwardFilterWorkspaceSize(
                cudnn_wrapper_.inline_cudnn_handle(),
                top_desc_,
                bottom_desc_,
                conv_desc_,
                filter_desc_,
                bwd_filter_algo_,
                &bwd_filter_ws_size));
            // get workspace for backwards data algorithm
            CUDNN_ENFORCE(cudnnGetConvolutionForwardWorkspaceSize(
                cudnn_wrapper_.inline_cudnn_handle(),
                top_desc_,
                filter_desc_,
                conv_desc_,
                bottom_desc_,
                algo_,
                &fwd_ws_size));
            cudnn_ws_nbytes_ = std::max(bwd_filter_ws_size, fwd_ws_size);

            VLOG(1) << "Cudnn bwd algorithm: " << bwd_filter_algo_ << ", " << algo_;
            VLOG(1) << "Cudnn workspace size: " << cudnn_ws_nbytes_;
          }

          // Now, actually run the computation.
          if (!no_bias_) {
            auto* dbias = Output(BIAS_OR_INPUT_GRAD, {C}, at::dtype<T>());
            CUDNN_ENFORCE(cudnnConvolutionBackwardBias(
                cudnn_wrapper_.inline_cudnn_handle(),
                cudnnTypeWrapper<T>::kOne(),
                top_desc_for_bias_,
                dY.template data<T>(),
                cudnnTypeWrapper<T>::kZero(),
                bias_desc_,
                dbias->template mutable_data<T>()));
          }

        #if CUDNN_VERSION_MIN(7, 0, 0)
          cudnn_wrapper_.with_cudnn_state(cudnn_state_, [&](CudnnState* state) {
            CUDNN_ENFORCE(cudnnConvolutionBackwardFilter(
                state->cudnn_handle(),
                cudnnTypeWrapper<T>::kOne(),
                top_desc_,
                dY.template data<T>(),
                bottom_desc_,
                X.template data<T>(),
                conv_desc_,
                bwd_filter_algo_,
                state->workspace().get(cudnn_ws_nbytes_),
                cudnn_ws_nbytes_,
                cudnnTypeWrapper<T>::kZero(),
                filter_desc_,
                dfilter->template mutable_data<T>()));

            if (OutputSize() == 3 || (no_bias_ && (OutputSize() == 2))) {
              // Compute the gradient w.r.t. the input.
              auto* dX = Output(
                  no_bias_ ? BIAS_OR_INPUT_GRAD : INPUT_GRAD,
                  X.sizes(),
                  at::dtype<T>());
              CUDNN_ENFORCE(cudnnConvolutionForward(
                  state->cudnn_handle(),
                  cudnnTypeWrapper<T>::kOne(),
                  top_desc_,
                  dY.template data<T>(),
                  filter_desc_,
                  filter.template data<T>(),
                  conv_desc_,
                  algo_,
                  state->workspace().get(cudnn_ws_nbytes_),
                  cudnn_ws_nbytes_,
                  cudnnTypeWrapper<T>::kZero(),
                  bottom_desc_,
                  dX->template mutable_data<T>()));
            }
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
              CUDNN_ENFORCE(cudnnConvolutionBackwardFilter(
                  state->cudnn_handle(),
                  cudnnTypeWrapper<T>::kOne(),
                  top_desc_,
                  dY.template data<T>() + i * group_offset_Y,
                  bottom_desc_,
                  X.template data<T>() + i * group_offset_X,
                  conv_desc_,
                  bwd_filter_algo_,
                  state->workspace().get(cudnn_ws_nbytes_),
                  cudnn_ws_nbytes_,
                  cudnnTypeWrapper<T>::kZero(),
                  filter_desc_,
                  dfilter->template mutable_data<T>() + i * group_offset_filter));
              if (OutputSize() == 3 || (no_bias_ && (OutputSize() == 2))) {
                // Compute the gradient w.r.t. the input.
                auto* dX = Output(
                    no_bias_ ? BIAS_OR_INPUT_GRAD : INPUT_GRAD,
                    X.sizes(),
                    at::dtype<T>());
                cudnn_wrapper_.with_cudnn_state(cudnn_state_, [&](CudnnState* state) {
                  CUDNN_ENFORCE(cudnnConvolutionForward(
                      state->cudnn_handle(),
                      cudnnTypeWrapper<T>::kOne(),
                      top_desc_,
                      dY.template data<T>() + i * group_offset_Y,
                      filter_desc_,
                      filter.template data<T>() + i * group_offset_filter,
                      conv_desc_,
                      algo_,
                      state->workspace().get(cudnn_ws_nbytes_),
                      cudnn_ws_nbytes_,
                      cudnnTypeWrapper<T>::kZero(),
                      bottom_desc_,
                      dX->template mutable_data<T>() + i * group_offset_X));
                });
              }
          }
        #endif
          return true;
        }

        REGISTER_CUDNN_OPERATOR(ConvTranspose, CudnnConvTransposeOp<float>);
        REGISTER_CUDNN_OPERATOR(
            ConvTransposeGradient,
            CudnnConvTransposeGradientOp<float>);
        */
    }
}

register_cuda_operator!{
    ConvTranspose, 
    ConvTransposeOp<f32, CUDAContext>
}

register_cuda_operator!{
    ConvTransposeGradient,
    ConvTransposeGradientOp<f32, CUDAContext>
}
