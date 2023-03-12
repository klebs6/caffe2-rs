crate::ix!();

pub struct CudnnConvGradientOp {
    base:                 CudnnConvOpBase,
    bwd_filter_conv_desc: CudnnConvolutionDescriptor,
    bwd_data_conv_desc:   CudnnConvolutionDescriptor,
    bwd_filter_algo:      CudnnConvolutionBwdFilterAlgo,
    bwd_data_algo:        CudnnConvolutionBwdDataAlgo,

    filter_algo_cache:    AlgorithmsCache<ConvBwdFilterAlgorithmWithCost>,
    data_algo_cache:      AlgorithmsCache<ConvBwdFilterAlgorithmWithCost>,

    no_bias: bool,

    // input: X, W, dY
    // output: dW, db, and optionally dX
}

input_tags!{
    CudnnConvGradientOp {
        Input,
        Filter,
        OutputGrad
    }
}

output_tags!{
    CudnnConvGradientOp {
        FilterGrad,
        BiasOrInputGrad,
        InputGrad
    }
}

impl CudnnConvGradientOp {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : CudnnConvOpBase(operator_def, ws),
            no_bias_(OperatorStorage::GetSingleArgument<int>("no_bias", 0)) 

        CAFFE_ENFORCE(
            !(no_bias_ && OutputSize() == 3),
            "If bias is not present, you should not have 3 grad output.");

        CUDNN_ENFORCE(cudnnCreateConvolutionDescriptor(&bwd_data_conv_desc_));
        CUDNN_ENFORCE(cudnnCreateConvolutionDescriptor(&bwd_filter_conv_desc_));
        */
    }
}

impl Drop for CudnnConvGradientOp {
    fn drop(&mut self) {
        todo!();
        /*
           CUDNN_ENFORCE(cudnnDestroyConvolutionDescriptor(bwd_data_conv_desc_));
           CUDNN_ENFORCE(cudnnDestroyConvolutionDescriptor(bwd_filter_conv_desc_));
        */
    }
}

pub type ConvBwdFilterAlgorithmWithCost = (CudnnConvolutionBwdFilterAlgo, f32);
pub type ConvBwdDataAlgorithmWithCost   = (CudnnConvolutionBwdDataAlgo,   f32);

impl CudnnConvGradientOp {

    #[inline] pub fn do_run_with_type<T_X, T_DY, T_W, T_B, T_DX, T_DW, T_DB>(&mut self) -> bool {
        todo!();
        /*
            auto& X = Input(INPUT);
          auto& filter = Input(FILTER);
          auto& dY = Input(OUTPUT_GRAD);

          CAFFE_ENFORCE(X.dim() >= 3 && X.dim() <= 5);
          CAFFE_ENFORCE(filter.dim() >= 3 && filter.dim() <= 5);

          const int M = filter.dim32(0);
          int N = 0, C = 0, H = 0, W = 0, D = 0, H_out = 0, W_out = 0, D_out = 0;
          int group_offset_X = 0, group_offset_Y = 0;

          switch (order_) {
            case StorageOrder::NHWC:
              N = X.dim32(0);
              H = X.dim32(1);
              W = X.dim() > 3 ? X.dim32(2) : 1;
              D = X.dim() > 4 ? X.dim32(3) : 1;
              C = X.dim32(X.dim() - 1);
              H_out = dY.dim32(1);
              W_out = dY.dim() > 3 ? dY.dim32(2) : 1;
              D_out = dY.dim() > 4 ? dY.dim32(3) : 1;
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
              H_out = dY.dim32(2);
              W_out = dY.dim() > 3 ? dY.dim32(3) : 1;
              D_out = dY.dim() > 4 ? dY.dim32(4) : 1;
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

          int group_offset_filter = filter.numel() / group_;
          if (kernel_.size() == 1) {
            ConvPoolOpBase<CUDAContext>::ComputePads({H});
          } else if (kernel_.size() == 2) {
            ConvPoolOpBase<CUDAContext>::ComputePads({H, W});
          } else if (kernel_.size() == 3) {
            ConvPoolOpBase<CUDAContext>::ComputePads({H, W, D});
          } else {
            CAFFE_THROW("Unsupported kernel size:", kernel_.size());
          }
          auto* dfilter = Output(FILTER_GRAD, filter.sizes(), at::dtype<T_DW>());

          if (N == 0) {
            math::Set<T_DW, CUDAContext>(
                dfilter->numel(),
                T_DW(0),
                dfilter->template mutable_data<T_DW>(),
                &context_);
            if (!no_bias_) {
              auto* dbias = Output(BIAS_OR_INPUT_GRAD, {M}, at::dtype<T_DB>());
              math::Set<T_DB, CUDAContext>(
                  dbias->numel(),
                  T_DB(0),
                  dbias->template mutable_data<T_DB>(),
                  &context_);
            }
            if (OutputSize() == 3 || (no_bias_ && (OutputSize() == 2))) {
              auto* dX = Output(
                  no_bias_ ? BIAS_OR_INPUT_GRAD : INPUT_GRAD,
                  X.sizes(),
                  at::dtype<T_DX>());
              dX->template mutable_data<T_DX>();
            }
            return true;
          }

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
              if (!no_bias_) {
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
            SetTensorNdDescriptorWithGroup<T_DX>(
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

            DuplicateConvDesc(
                conv_desc_, kernel_.size(), dilation_.size(), bwd_filter_conv_desc_);
            DuplicateConvDesc(
                conv_desc_, kernel_.size(), dilation_.size(), bwd_data_conv_desc_);

        #if CUDNN_VERSION_MIN(7, 0, 0)
            if (enable_tensor_core_) {
              CUDNN_ENFORCE(cudnnSetConvolutionMathType(
                  bwd_filter_conv_desc_, CUDNN_TENSOR_OP_MATH));
              CUDNN_ENFORCE(cudnnSetConvolutionMathType(
                  bwd_data_conv_desc_, CUDNN_TENSOR_OP_MATH));
            }

            // set cuDNN groups if appropriate
            CUDNN_CHECK(cudnnSetConvolutionGroupCount(bwd_filter_conv_desc_, group_));
            CUDNN_CHECK(cudnnSetConvolutionGroupCount(bwd_data_conv_desc_, group_));
        #endif

            // Choose dW algorithm
            if (force_algo_[ALGO_WGRAD] >= 0) {
              bwd_filter_algo_ =
                  (cudnnConvolutionBwdFilterAlgo_t)force_algo_[ALGO_WGRAD];
            } else if (deterministic_) {
              bwd_filter_algo_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
            } else if (exhaustive_search_) {
              // Even when FP16 compute is supported and requested, try FP32
              // because it may be faster. However, if FP32 compute is specified,
              // FP16 is not a suitable alternative - early out from the loop.
              std::array<ConvBwdFilterAlgorithmWithCost, 2> algosToCompare;
              for (int i = 0; i < 2; i++) {
                SetConvDescComputeType(bwd_filter_conv_desc_, kComputeTypesToTry[i]);

                algosToCompare[i] = filter_algo_cache_.getAlgorithm(
                    X.sizes(), filter.sizes(), kComputeTypesToTry[i], [&]() {
                      VLOG(1) << "CUDNN Convolution bwd: doing filter exhaustive"
                              << "search for " << kComputePassNames[i];
                      // When we do an exhaustive search, we will ignore the workspace
                      // size limit and simply go for the fastest algorithm. If you
                      // happen to run out of memory later, you will be on your own...
                      int returned_algo_count;
                      // We clean up the current workspace memory so that the forward
                      // algorithm is free to allocate memory.
                      // Actually run the search.
                      std::array<
                          cudnnConvolutionBwdFilterAlgoPerf_t,
                          kNUM_CUDNN_BWD_FILTER_ALGS>
                          filter_perf_stat;

                      cudnn_wrapper_.with_cudnn_state(
                          cudnn_state_, [&](CudnnState* state) {
                            CUDNN_ENFORCE(cudnnFindConvolutionBackwardFilterAlgorithmEx(
                                state->cudnn_handle(),
                                bottom_desc_,
                                X.template data<T_X>(),
                                top_desc_,
                                dY.template data<T_DY>(),
                                bwd_filter_conv_desc_,
                                filter_desc_,
                                dfilter->template mutable_data<T_DW>(),
                                kNUM_CUDNN_BWD_FILTER_ALGS,
                                &returned_algo_count,
                                filter_perf_stat.data(),
                                state->workspace().get(cudnn_ws_nbytes_limit_),
                                cudnn_ws_nbytes_limit_));
                          });
                      LogCudnnPerfStats(filter_perf_stat, returned_algo_count);
                      float algo_time =
                          filter_perf_stat[0].status == CUDNN_STATUS_SUCCESS
                          ? filter_perf_stat[0].time
                          : 1e10;
                      return ConvBwdFilterAlgorithmWithCost(
                          filter_perf_stat[0].algo, algo_time);
                    });

                // When set to fp32 compute, don't try fp16
                if (compute_type_ == CUDNN_DATA_FLOAT) {
                  break;
                }
              }

              if (compute_type_ == CUDNN_DATA_FLOAT) {
                // For FP32 compute, just use the best FP32 algorithm
                bwd_filter_algo_ = std::get<0>(algosToCompare[0]);
              } else {
                // For FP16 compute, choose algo with fastest execution
                int bestAlgoIndex =
                    (std::get<1>(algosToCompare[0]) < std::get<1>(algosToCompare[1]))
                    ? 0
                    : 1;
                bwd_filter_algo_ = std::get<0>(algosToCompare[bestAlgoIndex]);
                SetConvDescComputeType(
                    bwd_filter_conv_desc_, kComputeTypesToTry[bestAlgoIndex]);
              }
            } else {
              // choose backward algorithm for filter
              constexpr int nalgo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT;
              int valid_algos;
              cudnnConvolutionBwdFilterAlgoPerf_t algos[nalgo];
              CUDNN_ENFORCE(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
                  cudnn_wrapper_.inline_cudnn_handle(),
                  bottom_desc_,
                  top_desc_,
                  bwd_filter_conv_desc_,
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
            // Pick dX algo if needed
            if (OutputSize() == 3 || (no_bias_ && (OutputSize() == 2))) {
              if (force_algo_[ALGO_DGRAD] >= 0) {
                bwd_data_algo_ = (cudnnConvolutionBwdDataAlgo_t)force_algo_[ALGO_DGRAD];
              } else if (deterministic_) {
                bwd_data_algo_ = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
              } else if (exhaustive_search_) {
                // Even when FP16 compute is supported and requested, try FP32
                // because it may be faster. However, if FP32 compute is specified,
                // FP16 is not a suitable alternative - early out from the loop.
                std::array<ConvBwdDataAlgorithmWithCost, 2> algosToCompare;
                for (int i = 0; i < 2; i++) {
                  SetConvDescComputeType(bwd_data_conv_desc_, kComputeTypesToTry[i]);

                  algosToCompare[i] = data_algo_cache_.getAlgorithm(
                      X.sizes(), filter.sizes(), kComputeTypesToTry[i], [&]() {
                        VLOG(1) << "CUDNN Convolution bwd: doing data exhaustive"
                                << "search for " << kComputePassNames[i];
                        int returned_algo_count;

                        std::array<
                            cudnnConvolutionBwdDataAlgoPerf_t,
                            kNUM_CUDNN_BWD_DATA_ALGS>
                            data_perf_stat;
                        cudnn_wrapper_.with_cudnn_state(
                            cudnn_state_, [&](CudnnState* state) {
                              auto* dX = Output(
                                  no_bias_ ? BIAS_OR_INPUT_GRAD : INPUT_GRAD,
                                  X.sizes(),
                                  at::dtype<T_DX>());
                              const T_W* filter_data = filter.template data<T_W>();
                              const T_DY* dYdata = dY.template data<T_DY>();
                              T_DX* dXdata = dX->template mutable_data<T_DX>();
                              CUDNN_ENFORCE(cudnnFindConvolutionBackwardDataAlgorithmEx(
                                  state->cudnn_handle(),
                                  filter_desc_,
                                  filter_data,
                                  top_desc_,
                                  dYdata,
                                  bwd_data_conv_desc_,
                                  bottom_desc_,
                                  dXdata,
                                  kNUM_CUDNN_BWD_DATA_ALGS,
                                  &returned_algo_count,
                                  data_perf_stat.data(),
                                  state->workspace().get(cudnn_ws_nbytes_limit_),
                                  cudnn_ws_nbytes_limit_));
                            });

                        LogCudnnPerfStats(data_perf_stat, returned_algo_count);
                        float algo_time =
                            data_perf_stat[0].status == CUDNN_STATUS_SUCCESS
                            ? data_perf_stat[0].time
                            : 1e10;
                        return ConvBwdDataAlgorithmWithCost(
                            data_perf_stat[0].algo, algo_time);
                      });

                  // When set to fp32 compute, don't try fp16
                  if (compute_type_ == CUDNN_DATA_FLOAT) {
                    break;
                  }
                }

                if (compute_type_ == CUDNN_DATA_FLOAT) {
                  // For FP32 compute, just use the best FP32 algorithm
                  bwd_data_algo_ = std::get<0>(algosToCompare[0]);
                } else {
                  // For FP16 compute, choose algo with fastest execution
                  int bestAlgoIndex =
                      (std::get<1>(algosToCompare[0]) < std::get<1>(algosToCompare[1]))
                      ? 0
                      : 1;
                  bwd_data_algo_ = std::get<0>(algosToCompare[bestAlgoIndex]);
                  SetConvDescComputeType(
                      bwd_data_conv_desc_, kComputeTypesToTry[bestAlgoIndex]);
                }
              } else {
                constexpr int nalgo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT;
                int valid_algos;
                cudnnConvolutionBwdDataAlgoPerf_t algos[nalgo];
                CUDNN_ENFORCE(cudnnGetConvolutionBackwardDataAlgorithm_v7(
                    cudnn_wrapper_.inline_cudnn_handle(),
                    filter_desc_,
                    top_desc_,
                    bwd_data_conv_desc_,
                    bottom_desc_,
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
            }

            // get workspace size for backwards filter algorithm
            size_t bwd_filter_ws_size, bwd_data_ws_size;

            for (int step = 0; step < 2; ++step) {
              cudnnStatus_t _status = cudnnGetConvolutionBackwardFilterWorkspaceSize(
                  cudnn_wrapper_.inline_cudnn_handle(),
                  bottom_desc_,
                  top_desc_,
                  bwd_filter_conv_desc_,
                  filter_desc_,
                  bwd_filter_algo_,
                  &bwd_filter_ws_size);
              if (step == 0) {
                if (_status == CUDNN_STATUS_SUCCESS) {
                  break;
                }
                if (_status == CUDNN_STATUS_NOT_SUPPORTED) {
                  cudnnConvolutionBwdFilterAlgo_t new_algo = deterministic_
                      ? CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1
                      : CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
                  VLOG(1) << "Backward Filter algorithm " << (int)bwd_filter_algo_
                          << " is not currently supported for given parameters."
                          << " Trying the default algorithm " << (int)new_algo;
                  bwd_filter_algo_ = new_algo;
                  continue;
                }
              }
              CUDNN_ENFORCE(_status);
            }

            if (OutputSize() == 3 || (no_bias_ && (OutputSize() == 2))) {
              // get workspace size for backwards data algorithm
              for (int step = 0; step < 2; ++step) {
                cudnnStatus_t _status = cudnnGetConvolutionBackwardDataWorkspaceSize(
                    cudnn_wrapper_.inline_cudnn_handle(),
                    filter_desc_,
                    top_desc_,
                    bwd_data_conv_desc_,
                    bottom_desc_,
                    bwd_data_algo_,
                    &bwd_data_ws_size);
                if (step == 0) {
                  if (_status == CUDNN_STATUS_SUCCESS) {
                    break;
                  }
                  if (_status == CUDNN_STATUS_NOT_SUPPORTED) {
                    cudnnConvolutionBwdDataAlgo_t new_algo = deterministic_
                        ? CUDNN_CONVOLUTION_BWD_DATA_ALGO_1
                        : CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
                    VLOG(1) << "Backward Data algorithm " << (int)bwd_data_algo_
                            << " is not currently supported for given parameters."
                            << " Trying the default algorithm " << (int)new_algo;
                    bwd_data_algo_ = new_algo;
                    continue;
                  }
                }
                CUDNN_ENFORCE(_status);
              }
            } else {
              bwd_data_ws_size = 0;
            }
            cudnn_ws_nbytes_ = std::max(bwd_filter_ws_size, bwd_data_ws_size);

            VLOG(1) << "Cudnn bwd data & filter algorithm: " << bwd_data_algo_ << ", "
                    << bwd_filter_algo_;
            VLOG(1) << "Cudnn workspace size: " << cudnn_ws_nbytes_;
          }

          // Now, actually run the computation.
          if (!no_bias_) {
            auto* dbias = Output(BIAS_OR_INPUT_GRAD, {M}, at::dtype<T_DB>());
            CUDNN_ENFORCE(cudnnConvolutionBackwardBias(
                cudnn_wrapper_.inline_cudnn_handle(),
                cudnnTypeWrapper<T_DY>::kOne(),
                top_desc_for_bias_,
                dY.template data<T_DY>(),
                cudnnTypeWrapper<T_DB>::kZero(),
                bias_desc_,
                dbias->template mutable_data<T_DB>()));
          }

        #if CUDNN_VERSION_MIN(7, 0, 0)
          cudnn_wrapper_.with_cudnn_state(cudnn_state_, [&](CudnnState* state) {
            CUDNN_ENFORCE(cudnnConvolutionBackwardFilter(
                state->cudnn_handle(),
                cudnnTypeWrapper<T_X>::kOne(),
                bottom_desc_,
                X.template data<T_X>(),
                top_desc_,
                dY.template data<T_DY>(),
                bwd_filter_conv_desc_,
                bwd_filter_algo_,
                state->workspace().get(cudnn_ws_nbytes_),
                cudnn_ws_nbytes_,
                cudnnTypeWrapper<T_DW>::kZero(),
                filter_desc_,
                dfilter->template mutable_data<T_DW>()));
            if (OutputSize() == 3 || (no_bias_ && (OutputSize() == 2))) {
              // Compute the gradient w.r.t. the input.

              auto* dX = Output(
                  no_bias_ ? BIAS_OR_INPUT_GRAD : INPUT_GRAD,
                  X.sizes(),
                  at::dtype<T_DX>());
              CUDNN_ENFORCE(cudnnConvolutionBackwardData(
                  state->cudnn_handle(),
                  cudnnTypeWrapper<T_W>::kOne(),
                  filter_desc_,
                  filter.template data<T_W>(),
                  top_desc_,
                  dY.template data<T_DY>(),
                  bwd_data_conv_desc_,
                  bwd_data_algo_,
                  state->workspace().get(cudnn_ws_nbytes_),
                  cudnn_ws_nbytes_,
                  cudnnTypeWrapper<T_DX>::kZero(),
                  bottom_desc_,
                  dX->template mutable_data<T_DX>()));
            }
          });
        #else
          for (int i = 0; i < group_; ++i) {
            cudnn_wrapper_.with_cudnn_state(cudnn_state_, [&](CudnnState* state) {
              CUDNN_ENFORCE(cudnnConvolutionBackwardFilter(
                  state->cudnn_handle(),
                  cudnnTypeWrapper<T_X>::kOne(),
                  bottom_desc_,
                  X.template data<T_X>() + i * group_offset_X,
                  top_desc_,
                  dY.template data<T_DY>() + i * group_offset_Y,
                  bwd_filter_conv_desc_,
                  bwd_filter_algo_,
                  state->workspace().get(cudnn_ws_nbytes_),
                  cudnn_ws_nbytes_,
                  cudnnTypeWrapper<T_DW>::kZero(),
                  filter_desc_,
                  dfilter->template mutable_data<T_DW>() + i * group_offset_filter));
              if (OutputSize() == 3 || (no_bias_ && (OutputSize() == 2))) {
                // Compute the gradient w.r.t. the input.
                auto* dX = Output(no_bias_ ? BIAS_OR_INPUT_GRAD : INPUT_GRAD);
                dX->ResizeLike(X);
                CUDNN_ENFORCE(cudnnConvolutionBackwardData(
                    state->cudnn_handle(),
                    cudnnTypeWrapper<T_W>::kOne(),
                    filter_desc_,
                    filter.template data<T_W>() + i * group_offset_filter,
                    top_desc_,
                    dY.template data<T_DY>() + i * group_offset_Y,
                    bwd_data_conv_desc_,
                    bwd_data_algo_,
                    state->workspace().get(cudnn_ws_nbytes_),
                    cudnn_ws_nbytes_,
                    cudnnTypeWrapper<T_DX>::kZero(),
                    bottom_desc_,
                    dX->template mutable_data<T_DX>() + i * group_offset_X));
              }
            });
          }
        #endif
          return true;
        */
    }

    /**
      | TODO(Yangqing): a lot of the function
      | contents are very similar. Consider
      | consolidating them.
      |
      */
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            if (Input(0).IsType<float>()) {
        return DoRunWithType<
            float, //  X
            float, // dY
            float, //  W
            float, //  b
            float, // dX
            float, // dW
            float>(); // db
      } else if (Input(0).IsType<at::Half>()) {
        return DoRunWithType<
            at::Half, //  X
            at::Half, // dY
            at::Half, //  W
            at::Half, //  b
            at::Half, // dX
            at::Half, // dW
            at::Half>(); // db
      } else {
        LOG(FATAL) << "Unsupported input types";
      }
      return true;
        */
    }
}

