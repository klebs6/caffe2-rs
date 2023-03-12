crate::ix!();

pub struct CudnnConvOp {

    base: CudnnConvOpBase,

    // Input: X, W, b
    // Output: Y

    algo:       CudnnConvolutionFwdAlgo,
    algo_cache: AlgorithmsCache<ConvFwdAlgorithmWithCost>,
}

input_tags!{
    CudnnConvOp {
        Input,
        Filter,
        Bias
    }
}

impl CudnnConvOp {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : CudnnConvOpBase(operator_def, ws)
        */
    }
}

type ConvFwdAlgorithmWithCost = (CudnnConvolutionFwdAlgo, f32);

/* ----------------- Implementations  ----------------- */

pub const kComputeTypesToTry: [CudnnDataType; 2] = [
    CudnnDataType::CUDNN_DATA_FLOAT,
    CudnnDataType::CUDNN_DATA_HALF
];

pub const kComputePassNames: [&'static str; 2] = [
    "fp32 compute",
    "fp16 compute"
];

impl CudnnConvOp {

    #[inline] pub fn do_run_with_type<T_X, T_W, T_B, T_Y>(&mut self) -> bool {
        todo!();
        /*
            auto& X = Input(INPUT);
          auto& filter = Input(FILTER);

          // Figure out the output shape
          CAFFE_ENFORCE(X.dim() >= 3 && X.dim() <= 5);
          CAFFE_ENFORCE(filter.dim() >= 3 && filter.dim() <= 5);
          const int M = filter.dim32(0);
          auto output_sizes = ConvPoolOpBase<CUDAContext>::GetOutputSize(X, M);
          auto* Y = Output(0, output_sizes, at::dtype<T_Y>());

          int N = 0, C = 0, H = 0, W = 0, D = 0, H_out = 0, W_out = 0, D_out = 0;
          int group_offset_X = 0, group_offset_Y = 0;

          switch (order_) {
            case StorageOrder::NHWC:
              N = X.dim32(0);
              H = X.dim32(1);
              W = X.dim() > 3 ? X.dim32(2) : 1;
              D = X.dim() > 4 ? X.dim32(3) : 1;
              C = X.dim32(X.dim() - 1);
              H_out = Y->dim32(1);
              W_out = Y->dim() > 3 ? Y->dim32(2) : 1;
              D_out = Y->dim() > 4 ? Y->dim32(3) : 1;
              for (int i = 0; i < kernel_.size(); ++i) {
                CAFFE_ENFORCE_EQ(filter.dim32(i + 1), kernel_[i]);
              }
              CAFFE_ENFORCE_EQ(filter.dim32(filter.dim() - 1), C / group_);
              group_offset_X = C / group_;
              group_offset_Y = M / group_;
              break;
            case StorageOrder::NCHW:
              N = X.dim32(0);
              C = X.dim32(1);
              H = X.dim32(2);
              W = X.dim() > 3 ? X.dim32(3) : 1;
              D = X.dim() > 4 ? X.dim32(4) : 1;
              H_out = Y->dim32(2);
              W_out = Y->dim() > 3 ? Y->dim32(3) : 1;
              D_out = Y->dim() > 4 ? Y->dim32(4) : 1;
              CAFFE_ENFORCE_EQ(filter.dim32(1), C / group_);
              for (int i = 0; i < kernel_.size(); ++i) {
                CAFFE_ENFORCE_EQ(filter.dim32(i + 2), kernel_[i]);
              }
              group_offset_X = C / group_ * H * W * D;
              group_offset_Y = M / group_ * H_out * W_out * D_out;
              break;
            default:
              LOG(FATAL) << "Unknown storage order: " << order_;
          }

          CAFFE_ENFORCE(
              C % group_ == 0,
              "If you set group, the number of input channels should be divisible "
              "by group.");
          CAFFE_ENFORCE(
              M % group_ == 0,
              "If you set group, the number of output channels should be divisible "
              "by group.");

          if (N == 0) {
            Y->template mutable_data<T_Y>();
            return true;
          }

          int group_offset_filter = filter.numel() / group_;

          // Set up the cudnn algorithms & workspace if necessary
          bool input_changed = (X.sizes() != cudnn_input_dims_);
          bool filter_changed = (filter.sizes() != cudnn_filter_dims_);
          if (input_changed || filter_changed) {
            VLOG(1) << "Changing the cudnn descriptor configurations.";
            if (input_changed) {
              cudnn_input_dims_ = X.sizes().vec();
              SetTensorNdDescriptorWithGroup<T_X>(X.dim(), bottom_desc_, N, C, H, W, D);
            }
            if (filter_changed) {
              cudnn_filter_dims_ = filter.sizes().vec();
              if (kernel_.size() == 1 || kernel_.size() == 2) {
        #if CUDNN_VERSION_MIN(7, 0, 0)
                const int MM = M;
        #else
                const int MM = M / group_;
        #endif
                CUDNN_ENFORCE(cudnnSetFilter4dDescriptor(
                    filter_desc_,
                    cudnnTypeWrapper<T_W>::type,
                    GetCudnnTensorFormat(order_),
                    MM,
                    C / group_,
                    kernel_h(),
                    kernel_.size() == 1 ? 1 : kernel_w()));
              } else {
                vector<int> dims(filter.sizes().begin(), filter.sizes().end());
        #if !CUDNN_VERSION_MIN(7, 0, 0)
                // We only need to divide dims by group_ when CUDNN version < 7.0
                // see CUDA group convolution doc: https://fburl.com/dgj6dvpd
                order_ == StorageOrder::NCHW ? dims[1] /= group_
                                             : dims[filter.ndim() - 1] /= group_;
        #endif
                CUDNN_ENFORCE(cudnnSetFilterNdDescriptor(
                    filter_desc_,
                    cudnnTypeWrapper<T_W>::type,
                    GetCudnnTensorFormat(order_),
                    dims.size(),
                    dims.data()));
              }
              if (InputSize() == 3) {
                if (kernel_.size() == 1 || kernel_.size() == 2) {
                  CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
                      bias_desc_,
                      GetCudnnTensorFormat(order_),
                      cudnnTypeWrapper<T_B>::type,
                      1,
                      M,
                      1,
                      1));
                } else {
                  std::vector<int> bias_dims(X.dim(), 1);
                  bias_dims[1] = M;
                  std::vector<int> strides = {M, 1, 1, 1, 1, 1};
                  CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(
                      bias_desc_,
                      cudnnTypeWrapper<T_B>::type,
                      X.dim() > 3 ? X.dim() : 4,
                      bias_dims.data(),
                      strides.data()));
                }
              }
            }
            // Set the output
            SetTensorNdDescriptorWithGroup<T_Y>(
                X.dim(), top_desc_, N, M, H_out, W_out, D_out);
            // Set the output with descriptor useful for bias addition in one run.
            if (kernel_.size() == 1 || kernel_.size() == 2) {
              CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
                  top_desc_for_bias_,
                  GetCudnnTensorFormat(order_),
                  cudnnTypeWrapper<T_B>::type,
                  N,
                  M,
                  H_out,
                  W_out));
            } else {
              vector<int> dims = {N, M, H_out, W_out, D_out};
              vector<int> strides = {M * H_out * W_out * D_out,
                                     H_out * W_out * D_out,
                                     W_out * D_out,
                                     D_out,
                                     1};
              CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(
                  top_desc_for_bias_,
                  cudnnTypeWrapper<T_B>::type,
                  X.dim() > 3 ? X.dim() : 4,
                  dims.data(),
                  strides.data()));
            }

            compute_type_ = DetermineComputeTypeFromInput(X);
            SetConvDescFromArguments();

        #if CUDNN_VERSION_MIN(7, 0, 0)
            if (enable_tensor_core_) {
              CUDNN_ENFORCE(
                  cudnnSetConvolutionMathType(conv_desc_, CUDNN_TENSOR_OP_MATH));
            }

            // enable cuDNN conv groups
            CUDNN_CHECK(cudnnSetConvolutionGroupCount(conv_desc_, group_));
        #endif

            if (force_algo_[ALGO_FWD] >= 0) {
              algo_ = (cudnnConvolutionFwdAlgo_t)force_algo_[ALGO_FWD];
            } else if (deterministic_) {
              algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
            } else if (exhaustive_search_) {
              // Even when FP16 compute is supported and requested, try FP32
              // because it may be faster. However, if FP32 compute is specified,
              // FP16 is not a suitable alternative - early out from the loop.
              std::array<ConvFwdAlgorithmWithCost, 2> algosToCompare;
              for (int i = 0; i < 2; i++) {
                SetConvDescComputeType(conv_desc_, kComputeTypesToTry[i]);

                algosToCompare[i] = algo_cache_.getAlgorithm(
                    X.sizes(), filter.sizes(), kComputeTypesToTry[i], [&]() {
                      VLOG(1) << "CUDNN Convolution fwd: doing exhaustive "
                              << "search for " << kComputePassNames[i];
                      // When we do an exhaustive search, we will ignore the workspace
                      // size limit and simply go for the fastest algorithm. If you
                      // happen to run out of memory later, you will be on your own...
                      int returned_algo_count;
                      std::array<cudnnConvolutionFwdAlgoPerf_t, kNUM_CUDNN_FWD_ALGS>
                          fwd_perf_stat;

                      // no need to clean up workspace,
                      cudnn_wrapper_.with_cudnn_state(
                          cudnn_state_, [&](CudnnState* state) {
                            // Actually run the search.
                            CUDNN_ENFORCE(cudnnFindConvolutionForwardAlgorithmEx(
                                state->cudnn_handle(),
                                bottom_desc_,
                                X.template data<T_X>(),
                                filter_desc_,
                                filter.template data<T_W>(),
                                conv_desc_,
                                top_desc_,
                                Y->template mutable_data<T_Y>(),
                                kNUM_CUDNN_FWD_ALGS,
                                &returned_algo_count,
                                fwd_perf_stat.data(),
                                state->workspace().get(cudnn_ws_nbytes_limit_),
                                cudnn_ws_nbytes_limit_));
                          });
                      LogCudnnPerfStats(fwd_perf_stat, returned_algo_count);
                      float algo_time = fwd_perf_stat[0].status == CUDNN_STATUS_SUCCESS
                          ? fwd_perf_stat[0].time
                          : 1e10;
                      return ConvFwdAlgorithmWithCost(fwd_perf_stat[0].algo, algo_time);
                    });

                // When set to fp32 compute, don't try fp16
                if (compute_type_ == CUDNN_DATA_FLOAT) {
                  break;
                }
              }

              if (compute_type_ == CUDNN_DATA_FLOAT) {
                // For FP32 compute, just use the best FP32 algorithm
                algo_ = std::get<0>(algosToCompare[0]);
              } else {
                // For FP16 compute, choose algo with fastest execution
                int bestAlgoIndex =
                    (std::get<1>(algosToCompare[0]) < std::get<1>(algosToCompare[1]))
                    ? 0
                    : 1;
                algo_ = std::get<0>(algosToCompare[bestAlgoIndex]);
                SetConvDescComputeType(conv_desc_, kComputeTypesToTry[bestAlgoIndex]);
              }
            } else {
              // Get the convolution algorithm based on the workspace limit.
              constexpr int nalgo = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
              int valid_algos;
              cudnnConvolutionFwdAlgoPerf_t algos[nalgo];
              CUDNN_ENFORCE(cudnnGetConvolutionForwardAlgorithm_v7(
                  cudnn_wrapper_.inline_cudnn_handle(),
                  bottom_desc_,
                  filter_desc_,
                  conv_desc_,
                  top_desc_,
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
            for (int step = 0; step < 2; ++step) {
              cudnnStatus_t _status = cudnnGetConvolutionForwardWorkspaceSize(
                  cudnn_wrapper_.inline_cudnn_handle(),
                  bottom_desc_,
                  filter_desc_,
                  conv_desc_,
                  top_desc_,
                  algo_,
                  &cudnn_ws_nbytes_);
              if (step == 0) {
                if (_status == CUDNN_STATUS_SUCCESS) {
                  break;
                }
                if (_status == CUDNN_STATUS_NOT_SUPPORTED) {
                  cudnnConvolutionFwdAlgo_t new_algo = deterministic_
                      ? CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
                      : CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
                  VLOG(1) << "Forward algorithm " << (int)algo_
                          << " is not currently supported for given parameters."
                          << " Trying the default algorithm " << (int)new_algo;
                  algo_ = new_algo;
                  continue;
                }
              }
              CUDNN_ENFORCE(_status);
            }
            VLOG(1) << "Cudnn algorithm: " << algo_;
            VLOG(1) << "Cudnn workspace size: " << cudnn_ws_nbytes_;
          }

          // Now, actually run the computation.
          // Run directly through cuDNN if possible
        #if CUDNN_VERSION_MIN(7, 0, 0)
          cudnn_wrapper_.with_cudnn_state(cudnn_state_, [&](CudnnState* state) {
            CUDNN_ENFORCE(cudnnConvolutionForward(
                state->cudnn_handle(),
                cudnnTypeWrapper<T_X>::kOne(),
                bottom_desc_,
                X.template data<T_X>(),
                filter_desc_,
                filter.template data<T_W>(),
                conv_desc_,
                algo_,
                state->workspace().get(cudnn_ws_nbytes_),
                cudnn_ws_nbytes_,
                cudnnTypeWrapper<T_Y>::kZero(),
                top_desc_,
                Y->template mutable_data<T_Y>()));
          });
        #else
          // otherwise manually run through groups
          for (int i = 0; i < group_; ++i) {
            cudnn_wrapper_.with_cudnn_state(cudnn_state_, [&](CudnnState* state) {
              CUDNN_ENFORCE(cudnnConvolutionForward(
                  state->cudnn_handle(),
                  cudnnTypeWrapper<T_X>::kOne(),
                  bottom_desc_,
                  X.template data<T_X>() + i * group_offset_X,
                  filter_desc_,
                  filter.template data<T_W>() + i * group_offset_filter,
                  conv_desc_,
                  algo_,
                  state->workspace().get(cudnn_ws_nbytes_),
                  cudnn_ws_nbytes_,
                  cudnnTypeWrapper<T_Y>::kZero(),
                  top_desc_,
                  Y->template mutable_data<T_Y>() + i * group_offset_Y));
            });
          }
        #endif
          // Bias
          if (InputSize() == 3) {
            auto& bias = Input(BIAS);

            CAFFE_ENFORCE_EQ(bias.dim(), 1);
            CAFFE_ENFORCE_EQ(bias.dim32(0), M);

            CUDNN_ENFORCE(cudnnAddTensor(
                cudnn_wrapper_.inline_cudnn_handle(),
                cudnnTypeWrapper<T_B>::kOne(),
                bias_desc_,
                bias.template data<T_B>(),
                cudnnTypeWrapper<T_Y>::kOne(),
                top_desc_for_bias_,
                Y->template mutable_data<T_Y>()));
          }
          // Done.
          return true;
        */
    }

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            if (Input(0).IsType<float>()) {
        return DoRunWithType<
            float, // X
            float, // W
            float, // B
            float>(); // Y
      } else if (Input(0).IsType<at::Half>()) {
        return DoRunWithType<
            at::Half, // X
            at::Half, // W
            at::Half, // B
            at::Half>(); // Y
      } else {
        LOG(FATAL) << "Only float (32bit) and Half are supported by "
                   << "cudnn convolution, but input " << debug_def().input(0)
                   << " has [" << Input(0).dtype().name() << "]";
      }
      return true;
        */
    }
}
