crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/Convolution.cpp]
pub const MIOPEN_DIM_MAX: i32 = 5;

define_dispatch!{convolution_depthwise3x3_winograd_stub}

pub struct ConvParams {
    stride:         Vec<i64>,
    padding:        Vec<i64>,
    dilation:       Vec<i64>,
    transposed:     bool,
    output_padding: Vec<i64>,
    groups:         i32,
    benchmark:      bool,
    deterministic:  bool,
    cudnn_enabled:  bool,
    allow_tf32:     bool,
}

impl ConvParams {

    pub fn is_strided(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn is_dilated(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn is_padded(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn is_output_padding_neg(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn is_output_padding_big(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn is_padding_neg(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn is_stride_nonpos(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn view1d_as_2d(&mut self)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn use_cpu_depthwise3x3_winograd(&self, 
        input:  &Tensor,
        weight: &Tensor,
        bias:   &Tensor) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn needs_64bit_indexing_no_split(&self, 
        input:  &Tensor,
        weight: &Tensor) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn use_cudnn(&self, 
        input:  &Tensor,
        weight: &Tensor) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn use_cudnn_depthwise(&self, 
        input:  &Tensor,
        weight: &Tensor) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn use_miopen(&self, 
        input:        &Tensor,
        weight:       &Tensor,
        bias_defined: bool) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn use_mkldnn(&self, 
        input:  &Tensor,
        weight: &Tensor) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn use_nnpack(&self, 
        input:  &Tensor,
        weight: &Tensor) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn use_xnnpack(&self, 
        input:  &Tensor,
        weight: &Tensor,
        bias:   &Tensor) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn is_depthwise(&self, 
        input:  &Tensor,
        weight: &Tensor) -> bool {
        
        todo!();
        /*
        
        */
    }
}

impl fmt::Display for ConvParams {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            out << "ConvParams {"
          << "  stride = " << IntArrayRef{params.stride}
          << "  padding = " << IntArrayRef{params.padding}
          << "  dilation = " << IntArrayRef{params.dilation}
          << "  transposed = " << params.transposed
          << "  output_padding = " << IntArrayRef{params.output_padding}
          << "  groups = " << params.groups
          << "  benchmark = " << params.benchmark
          << "  deterministic = " << params.deterministic
          << "  cudnn_enabled = " << params.cudnn_enabled
          << "  allow_tf32 = " << params.allow_tf32
          << "}";
      return out;
        */
    }
}

impl ConvParams {
    
    pub fn is_strided(&mut self) -> bool {
        
        todo!();
        /*
            bool is_strided = false;
      for (int s : stride) {
        is_strided |= (s != 1);
      }
      return is_strided;
        */
    }
    
    pub fn is_dilated(&mut self) -> bool {
        
        todo!();
        /*
            bool is_dilated = false;
      for (int d : dilation) {
        is_dilated |= (d != 1);
      }
      return is_dilated;
        */
    }
    
    pub fn is_padded(&mut self) -> bool {
        
        todo!();
        /*
            bool is_padded = false;
      for (int p : padding) {
        is_padded |= (p != 0);
      }
      return is_padded;
        */
    }
    
    pub fn is_output_padding_neg(&mut self) -> bool {
        
        todo!();
        /*
            bool is_non_neg = false;
      for (int p : output_padding) {
        is_non_neg |= (p < 0);
      }
      return is_non_neg;
        */
    }
    
    pub fn is_output_padding_big(&mut self) -> bool {
        
        todo!();
        /*
            bool is_big = false;
      for (usize i = 0; i < output_padding.size(); i++) {
        is_big |= (output_padding[i] >= stride[i]);
      }
      return is_big;
        */
    }
    
    pub fn is_padding_neg(&mut self) -> bool {
        
        todo!();
        /*
            bool is_non_neg = false;
      for (int p : padding) {
        is_non_neg |= (p < 0);
      }
      return is_non_neg;
        */
    }
    
    pub fn is_stride_nonpos(&mut self) -> bool {
        
        todo!();
        /*
            bool is_nonpos = false;
      for (int s : stride) {
        is_nonpos |= (s <= 0);
      }
      return is_nonpos;
        */
    }
    
    pub fn new() -> Self {
    
        todo!();
        /*


            if (stride.size() == 1) {
        stride.insert(stride.begin(), 1);
        padding.insert(padding.begin(), 0);
        dilation.insert(dilation.begin(), 1);
        output_padding.insert(output_padding.begin(), 0);
      }
        */
    }
    
    pub fn use_cpu_depthwise3x3_winograd(&mut self, 
        input:  &Tensor,
        weight: &Tensor,
        bias:   &Tensor) -> bool {
        
        todo!();
        /*
            #if defined(__ARM_NEON__)
      // Currently only 3x3 depthwise convolutions on tensors of float are supported.
      return (input.ndimension() == 4) &&
             (input.size(1) == groups) &&
             (weight.ndimension() == 4 ) &&
             (weight.size(0) % input.size(1) == 0) &&
             (weight.size(1) == 1) &&
             (weight.size(2) == 3) &&
             (weight.size(3) == 3) &&
             (input.device().is_cpu()) &&
             (input.scalar_type() == kFloat) &&
             input.is_contiguous() &&
             (weight.device().is_cpu()) &&
             (weight.scalar_type() == kFloat) &&
             weight.is_contiguous() &&
             (!bias.defined() ||
                ((bias.device().is_cpu()) &&
                 (bias.scalar_type() == kFloat))) &&
             !is_strided() &&
             !is_dilated() &&
             // 3x3 depthwith convolutions implementation is inference only
             !(GradMode::is_enabled() &&
                     (input.requires_grad() ||
                      weight.requires_grad() ||
                     (bias.defined() && bias.requires_grad()))) &&
             !transposed;
    #else
      return false;
    #endif
        */
    }
    
    pub fn needs_64bit_indexing_no_split(&mut self, 
        input:  &Tensor,
        weight: &Tensor) -> bool {
        
        todo!();
        /*
            constexpr i64 int_max = int::max;
      i64 numel_input = input.numel();
      // empty input
      if (numel_input == 0) {
        return false;
      }
      // input size can not be reduced to the range of int by splitting the batch dim
      i64 n = input.size(0);
      if (numel_input / n > int_max) {
        return true;
      }
      // output size can not be reduced to the range of int by splitting the batch dim
      i64 outsize = 1;
      if (transposed) {
        vector<i64> o = conv_input_size(input.sizes(), weight.sizes(), padding, output_padding, stride, dilation, groups);
        outsize = multiply_integers(o.begin() + 1, o.end());
      } else {
        vector<i64> o = conv_output_size(input.sizes(), weight.sizes(), padding, stride, dilation);
        outsize = multiply_integers(o.begin() + 1, o.end());
      }
      return outsize > int_max;
        */
    }
    
    pub fn use_cudnn(&mut self, 
        input:  &Tensor,
        weight: &Tensor) -> bool {
        
        todo!();
        /*
            if (needs_64bit_indexing_no_split(input, weight)) {
        return false;
      }
      if (!getCUDAHooks().compiledWithCuDNN()) {
        return false;
      }
      if (!input.is_cuda() || !cudnn_enabled) {
        return false;
      }
      if (input.scalar_type() == kBFloat16 || weight.scalar_type() == kBFloat16) {
        return false;
      }
      if (!cudnn_conv_use_channels_last(input, weight)) { // bypass dilation checks for channels-last convolution
        if (deterministic && is_dilated()) {
          // cudnn doesn't support deterministic dilated convolution fully yet
          return false;
        }
        if (is_dilated()) {
          return getCUDAHooks().supportsDilatedConvolutionWithCuDNN() && !is_output_padding_big();
        }
      }
      return !is_output_padding_big();
        */
    }
    
    pub fn use_miopen(&mut self, 
        input:        &Tensor,
        weight:       &Tensor,
        bias_defined: bool) -> bool {
        
        todo!();
        /*
            if (needs_64bit_indexing_no_split(input, weight)) {
        return false;
      }
      return ((input.scalar_type() == kFloat) || (input.scalar_type() == kHalf) || (input.scalar_type() == kBFloat16))
             && getCUDAHooks().compiledWithMIOpen()
             && input.is_cuda()
             && input.dim() <= MIOPEN_DIM_MAX
             && !(groups > 1 && is_dilated()) // MIOpen currently does not support dilation with groups of size > 1
             && !(input.scalar_type() == kBFloat16 && bias_defined) // MIOpen currently doesn't support bias with bfloat16
             && cudnn_enabled
             ;
        */
    }
    
    pub fn use_mkldnn(&mut self, 
        input:  &Tensor,
        weight: &Tensor) -> bool {
        
        todo!();
        /*
            #if AT_MKLDNN_ENABLED()
      if (!globalContext().userEnabledMkldnn()) {
        return false;
      }
      return (input.is_mkldnn()) || // input is mkldnn Tensor
        (input.device().is_cpu() &&
         input.scalar_type() == kFloat && // only on CPU Float Tensors
         !transposed && // or transposed tensors
         // For 1x1 filters, MKLDNN is faster than THNN when multi-threaded,
         // but THNN is faster when single-threaded.
         (is_strided() || is_dilated() || input.size(0) >= 16 ||
          weight.size(-1) != 1 || weight.size(-2) != 1 || get_num_threads() > 1) &&
         (groups > 1
          || (weight.size(-1) > 3 && weight.size(-2) > 3)
          || input.size(0) > 1
          || input.size(0)*input.size(1)*input.size(2)*input.size(3) > 20480) // for some case, native is faster
          );

    #endif
      return false;
        */
    }
    
    pub fn use_nnpack(&mut self, 
        input:  &Tensor,
        weight: &Tensor) -> bool {
        
        todo!();
        /*
            #if AT_NNPACK_ENABLED()
      return _nnpack_available() &&
             input.device().is_cpu() &&
             input.scalar_type() == kFloat && // only on CPU Float Tensors
             !is_dilated() && // or dilation
             !transposed &&   // or transposed tensors
             input.ndimension() == 4 && // must be in NCHW format
             weight.ndimension() == 4 &&
             (weight.size(2) < 17) && (weight.size(3) < 17) // NNPACK only supports kernels up to 16x16
    #if !defined(C10_MOBILE)
             && input.size(0) >= 16 // ensure large enough batch size to ensure perf, tuneable
    #endif
         ;
    #endif
      return false;
        */
    }
    
    pub fn use_xnnpack(&mut self, 
        input:  &Tensor,
        weight: &Tensor,
        bias:   &Tensor) -> bool {
        
        todo!();
        /*
            #if defined(C10_MOBILE)
      if (!transposed) {
        return (input.size(1) == groups) &&
                xnnpack::use_convolution2d(
                    input,
                    weight,
                    bias,
                    padding,
                    stride,
                    dilation,
                    groups,
                    transposed);
      }
    #endif
      return false;
        */
    }

    /**
      | We currently only have depthwise support for
      | the case where groups == nInputPlane and
      | nInputPlane == nOutputPlane (the latter due to
      | the lack of a depthwise multiplier)
      */
    pub fn is_depthwise(&mut self, 
        input:  &Tensor,
        weight: &Tensor) -> bool {
        
        todo!();
        /*
            return input.is_cuda() &&
             !transposed &&
             (input.ndimension() == 4 || input.ndimension() == 5) &&
             input.size(1) == groups &&
             groups > 1 && // no point if there is only a single group
             weight.size(0) % input.size(1) == 0; // output channels must be a multiple of input channels
        */
    }
}

/// Check workload to activate fast depthwise FP16
/// cudnn conv kernels
///
pub fn check_cudnn_depthwise_workload(
        input:  &Tensor,
        stride: i32) -> bool {
    
    todo!();
        /*
            int w = input.size(3);  // same as h
      int ch = input.size(1);
      int bs = input.size(0);
      if (stride==1) {
        if (w >= 7) {
          // All batch sizes and nb_channels
          if (w >= 112) {
            return true;
          }

          // large nb_channels
          if (ch >= 1024) {
            if (w >= 56) {
              return true;
            } else if (bs >= 32) {
              return true;
            }
          }

          // batch_size specific
          if (bs >= 128) {
            if (ch >= 512) {
              return true;
            } else if (ch >= 64) {
              if (w >= 14) {
                return true;
              }
            } else if ((ch >= 32) && (w >=28)) {
              return true;
            }
          } else if (bs >= 64) {
            if ((ch >= 256) && (w >= 14)) {
              return true;
            } else if ((ch >= 32) && (w >= 28)) {
              return true;
            }
          } else if (bs >= 32) {
            if ((ch >= 256) && (w >= 14)) {
              return true;
            } else if ((ch >= 128) && (w >= 28)) {
              return true;
            } else if ((ch >= 32) && (w >= 56)) {
              return true;
            }
          } else if (bs >= 16) {
            if ((ch >= 1024) && (w >= 14)) {
              return true;
            }
            if ((ch >= 256) && (w >= 28)) {
              return true;
            } else if ((ch >= 32) && (w >= 56)) {
              return true;
            }
          } else if (bs >= 8) {
            if ((ch >= 512) && (w >= 28)) {
              return true;
            } else if ((ch >= 64) && (w >= 56)) {
              return true;
            }
          }
        }
      } else if (stride==2) {
        if (ch < 256) {
          return false;
        }

        if (w >= 7) {
          if (bs >= 128) {
            if (ch >= 1024) {
              return true;
            } else if ((ch >= 512) && (w >= 14)) {
              return true;
            } else if (w >= 28) {
              return true;
            }
          } else if (bs >= 64) {
            if ((ch >= 512) && (w >= 14)) {
              return true;
            } else if (w >= 28) {
              return true;
            }
          } else if (bs >= 32) {
            if ((ch >= 1024) && (w >= 14)) {
              return true;
            } else if (w >= 28) {
              return true;
            }
          } else if (bs >= 16) {
            if ((ch >= 512) && (w >= 28)) {
              return true;
            } else if (w >= 56) {
              return true;
            }
          } else if (bs >= 8) {
            if ((ch >= 1024) && (w >= 28)) {
              return true;
            } else if (w >= 56) {
              return true;
            }
          } else if (bs >= 1) {
            if ((ch >= 512) && (w >=112)) {
              return true;
            }
          }
        }
      }
      return false;
        */
}

impl ConvParams {
    
    /// Use cudnn for FP16 depthwise convolutions
    ///
    pub fn use_cudnn_depthwise(&mut self, 
        input:  &Tensor,
        weight: &Tensor) -> bool {
        
        todo!();
        /*
            if (cudnn_conv_use_channels_last(input, weight) && use_cudnn(input, weight)) {
        return true;
      }
      if (getCUDAHooks().supportsDepthwiseConvolutionWithCuDNN()) {
        long cudnn_version = getCUDAHooks().versionCuDNN();
        bool kernel_cond =  (cudnn_version >= 7600 &&
                             use_cudnn(input, weight) &&
                             input.scalar_type() == kHalf && // only for FP16
                             weight.scalar_type() == kHalf &&
                             is_depthwise(input, weight) &&
                             input.ndimension() == 4 &&   // TODO: 5-D contiguous depthwise is not supported yet, need benchmarks
                             weight.size(2) == weight.size(3) && // only square kernels
                             input.size(2) >= 7 && // min width/height 7
                             !is_dilated() && // no dilation supported
                             stride[0] == stride[1] && // equal strides
                             ((weight.size(3) == 3) || (weight.size(3) == 1)) &&
                             input.size(1) >= 32); // min 32 channels supported)
        if (kernel_cond) {
          return check_cudnn_depthwise_workload(input, stride[0]);
        } else {
          return false;
        }
      } else {
        return false;
      }
        */
    }
}

pub fn check_shape_forward(
        input:        &Tensor,
        weight_sizes: &&[i32],
        bias:         &Tensor,
        params:       &ConvParams)  {
    
    todo!();
        /*
            i64 k = input.ndimension();
      i64 weight_dim = weight_sizes.size();
      i64 groups = params.groups;
      auto padding = params.padding;
      auto dilation = params.dilation;
      bool transposed = params.transposed;

      TORCH_CHECK(!params.is_padding_neg(), "negative padding is not supported");
      TORCH_CHECK(!params.is_output_padding_neg(), "negative output_padding is not supported");
      TORCH_CHECK(!params.is_stride_nonpos(), "non-positive stride is not supported");

      TORCH_CHECK(weight_dim == k,
               "Expected ", weight_dim, "-dimensional input for ", weight_dim,
               "-dimensional weight ", weight_sizes, ", but got ", k, "-dimensional input of size ",
               input.sizes(), " instead");
      TORCH_CHECK(weight_sizes[0] >= groups,
               "Given groups=", groups, ", expected weight to be at least ", groups,
               " at dimension 0, but got weight of size ", weight_sizes, " instead");
      TORCH_CHECK(weight_sizes[0] % groups == 0,
               "Given groups=", groups, ", expected weight to be divisible by ",
               groups, " at dimension 0, but got weight of size [", weight_sizes,
               "] instead");

      if (!transposed) {
        vector<i64> input_shape;
        vector<i64> kernel_shape;
        bool kernel_size_correct = true;

        TORCH_CHECK(input.size(1) == (weight_sizes[1] * groups),
                 "Given groups=", groups, ", weight of size ", weight_sizes,
                 ", expected input", input.sizes(), " to have ",
                 (weight_sizes[1] * groups), " channels, but got ", input.size(1),
                 " channels instead");
        TORCH_CHECK(!bias.defined() || (bias.ndimension() == 1 && bias.size(0) == weight_sizes[0]),
                 "Given weight of size ", weight_sizes,
                 ", expected bias to be 1-dimensional with ", weight_sizes[0], " elements",
                 ", but got bias of size ", bias.sizes(), " instead");

        for (const auto i : irange(2, k)) {
          input_shape.push_back(input.size(i) + 2 * padding[i-2]);
          // log new kernel size considering dilation
          kernel_shape.push_back(dilation[i-2] * (weight_sizes[i]-1) + 1);
          if (input_shape.back() < kernel_shape.back()) {
            kernel_size_correct = false;
          }
        }

        TORCH_CHECK(input_shape.size() == kernel_shape.size(), "Inconsistent shape between Input and Kernel");

        if (!kernel_size_correct) {
          // If kernel size is incorrect
          ostringstream input_ss;
          ostringstream kernel_ss;
          string separator = "";

          for (int i = 0, len = input_shape.size(); i < len; ++i) {
            input_ss << separator << input_shape[i];
            kernel_ss << separator << kernel_shape[i];
            separator = " x ";
          }

          AT_ERROR("Calculated padded input size per channel: (", input_ss.str(), "). "
                   "Kernel size: (", kernel_ss.str(), "). Kernel size can't be greater than actual input size");
        }
      } else { // transposed
        TORCH_CHECK(input.size(1) == weight_sizes[0],
                 "Given transposed=", transposed, ", weight of size ", weight_sizes,
                 ", expected input", input.sizes(), " to have ", weight_sizes[0],
                 " channels, but got ", input.size(1), " channels instead");
        TORCH_CHECK(!bias.defined() || (bias.ndimension() == 1 && bias.size(0) == weight_sizes[1] * groups),
                 "Given transposed=", transposed, ", weight of size ", weight_sizes,
                 ", expected bias to be 1-dimensional with ", weight_sizes[1] * groups, " elements",
                 ", but got bias of size ", bias.sizes(), " instead");
      }
        */
}

pub fn view4d(tensor: &Tensor) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(tensor.ndimension() == 3,
               "expected 3D tensor, got tensor with ", tensor.ndimension(),
               " dimensions instead");
      return tensor.unsqueeze(2);
        */
}

pub fn view3d(tensor: &Tensor) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(tensor.ndimension() == 4,
               "expected 4D tensor, got tensor with ", tensor.ndimension(),
               " dimensions instead");
      return tensor.squeeze(2);
        */
}

pub fn subtensor(
        tensor: &mut Tensor,
        dim:    i32,
        groups: i32,
        g:      i32) -> Tensor {
    
    todo!();
        /*
            if (!tensor.defined()) {
        return Tensor();
      }
      i64 n = tensor.sizes()[dim] / groups;
      return tensor.narrow(dim, n * g, n).contiguous();
        */
}

pub fn conv1d_with_bias_opt(
        input:    &Tensor,
        weight:   &Tensor,
        bias_opt: &Option<Tensor>,
        stride:   &[i32],
        padding:  &[i32],
        dilation: &[i32],
        groups:   i64) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> bias_maybe_owned = borrow_from_optional_tensor(bias_opt);
      const Tensor& bias = *bias_maybe_owned;

      return convolution(input, weight, bias, stride, padding, dilation,
                             false, {0}, groups);
        */
}

pub fn conv2d_with_bias_opt(
        input:    &Tensor,
        weight:   &Tensor,
        bias_opt: &Option<Tensor>,
        stride:   &[i32],
        padding:  &[i32],
        dilation: &[i32],
        groups:   i64) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> bias_maybe_owned = borrow_from_optional_tensor(bias_opt);
      const Tensor& bias = *bias_maybe_owned;

      return convolution(input, weight, bias, stride, padding, dilation,
                             false, {{0, 0}}, groups);
        */
}

pub fn conv3d_with_bias_opt(
        input:    &Tensor,
        weight:   &Tensor,
        bias_opt: &Option<Tensor>,
        stride:   &[i32],
        padding:  &[i32],
        dilation: &[i32],
        groups:   i64) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> bias_maybe_owned = borrow_from_optional_tensor(bias_opt);
      const Tensor& bias = *bias_maybe_owned;

      return convolution(input, weight, bias, stride, padding, dilation,
                             false, {{0, 0, 0}}, groups);
        */
}

pub fn convolution_same(
        input:    &Tensor,
        weight:   &Tensor,
        bias:     &Tensor,
        stride:   &[i32],
        dilation: &[i32],
        groups:   i64) -> Tensor {
    
    todo!();
        /*
            auto k = weight.dim();
      auto dim = k - 2;
      TORCH_CHECK(dim > 0, "weight should have at least three dimensions");
      auto weight_sizes = weight.sizes();
      auto input_sizes = input.sizes();
      TORCH_CHECK(k == input.dim(),
                  "Expected ", k, "-dimensional input for ",
                  k, "-dimensional weight", weight_sizes, ", but got ",
                  input.dim(), "-dimensional input of size ",
                  input.sizes(), " instead");
      TORCH_CHECK(stride.size() == dim || stride.size() == 1,
                  "stride cannot broadcast to ", dim, " dimensions");
      TORCH_CHECK(dilation.size() == dim || dilation.size() == 1,
                  "dilation cannot broadcast to ", dim, " dimensions");
      for (i64 i = 0; i < stride.size(); ++i) {
        TORCH_CHECK(stride[i] == 1, "padding='same' is not supported for strided convolutions");
      }

      // Calculate the correct padding
      DimVector padding_l, padding_r;
      bool symmetric_padding = true;
      for (i64 i = 0; i < dim; ++i) {
        auto s = stride.size() == 1 ? stride[0] : stride[i];
        auto d = dilation.size() == 1 ? dilation[0] : dilation[i];
        auto pad = pooling_same_mode_padding_lr(
            input_sizes[i + 2], weight_sizes[i + 2], s, d);
        padding_l.push_back(pad.first);
        padding_r.push_back(pad.second);
        if (pad.first != pad.second) {
          symmetric_padding = false;
        }
      }

      if (symmetric_padding) {
        // All backends handle symmetric padding natively
        DimVector output_padding(static_cast<usize>(dim));
        return native::convolution(input, weight, bias, stride, padding_l, dilation,
                                   false, output_padding, groups);
      }

      TORCH_WARN_ONCE("Using padding='same' with even kernel lengths and odd dilation may"
                      " require a zero-padded copy of the input be created");
      SmallVector<i64, kDimVectorStaticSize * 2> pad_nd(static_cast<usize>(2 * dim));
      for (int i = 0; i < dim; ++i) {
        // Apply padding by the difference, leaving only a symmetric padding
        auto delta_pad = padding_r[i] - padding_l[i];
        auto pad_idx = 2 * (dim - 1 - i);  // F.pad goes from last dim to first
        if (delta_pad > 0) {
          pad_nd[pad_idx + 1] = delta_pad;
        } else {
          pad_nd[pad_idx] = delta_pad;
          padding_l[i] = padding_r[i];
        }
      }
      auto padded_input = constant_pad_nd(input, pad_nd, 0);
      DimVector output_padding(static_cast<usize>(dim));
      return convolution(padded_input, weight, bias, stride, padding_l,
                             dilation, false, output_padding, groups);
        */
}

pub fn convolution_mode(
        input:    &Tensor,
        weight:   &Tensor,
        bias_opt: &Option<Tensor>,
        stride:   &[i32],
        padding:  StringView,
        dilation: &[i32],
        groups:   i64) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> bias_maybe_owned = borrow_from_optional_tensor(bias_opt);
      const Tensor& bias = *bias_maybe_owned;

      if (padding == "same") {
        return native::convolution_same(
            input, weight, bias, stride, dilation, groups);
      } else if (padding == "valid") {
        const i64 padding_[] = {0};
        return native::convolution(
            input, weight, bias, stride, padding_, dilation, false, padding_, groups);
      }
      TORCH_CHECK(false, "Invalid padding string: '", padding, "'");
        */
}

pub fn conv1d(
        input:    &Tensor,
        weight:   &Tensor,
        bias:     &Option<Tensor>,
        stride:   &[i32],
        padding:  StringView,
        dilation: &[i32],
        groups:   i64) -> Tensor {
    
    todo!();
        /*
            return _convolution_mode(
          input, weight, bias, stride, move(padding), dilation, groups);
        */
}

pub fn conv2d(
        input:    &Tensor,
        weight:   &Tensor,
        bias:     &Option<Tensor>,
        stride:   &[i32],
        padding:  StringView,
        dilation: &[i32],
        groups:   i64) -> Tensor {
    
    todo!();
        /*
            return _convolution_mode(
          input, weight, bias, stride, move(padding), dilation, groups);
        */
}

pub fn conv3d(
        input:    &Tensor,
        weight:   &Tensor,
        bias:     &Option<Tensor>,
        stride:   &[i32],
        padding:  StringView,
        dilation: &[i32],
        groups:   i64) -> Tensor {
    
    todo!();
        /*
            return _convolution_mode(
          input, weight, bias, stride, move(padding), dilation, groups);
        */
}

pub fn conv_transpose1d(
        input:          &Tensor,
        weight:         &Tensor,
        bias_opt:       &Option<Tensor>,
        stride:         &[i32],
        padding:        &[i32],
        output_padding: &[i32],
        groups:         i64,
        dilation:       &[i32]) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> bias_maybe_owned = borrow_from_optional_tensor(bias_opt);
      const Tensor& bias = *bias_maybe_owned;

      return convolution(input, weight, bias, stride, padding, dilation,
                             true, output_padding, groups);
        */
}


pub fn conv_transpose2d(
        input:          &Tensor,
        weight:         &Tensor,
        bias_opt:       &Option<Tensor>,
        stride:         &[i32],
        padding:        &[i32],
        output_padding: &[i32],
        groups:         i64,
        dilation:       &[i32]) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> bias_maybe_owned = borrow_from_optional_tensor(bias_opt);
      const Tensor& bias = *bias_maybe_owned;

      return convolution(input, weight, bias, stride, padding, dilation,
                             true, output_padding, groups);
        */
}

pub fn conv_transpose3d(
        input:          &Tensor,
        weight:         &Tensor,
        bias_opt:       &Option<Tensor>,
        stride:         &[i32],
        padding:        &[i32],
        output_padding: &[i32],
        groups:         i64,
        dilation:       &[i32]) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> bias_maybe_owned = borrow_from_optional_tensor(bias_opt);
      const Tensor& bias = *bias_maybe_owned;

      return convolution(input, weight, bias, stride, padding, dilation,
                             true, output_padding, groups);
        */
}

pub fn convolution_a(
        input:          &Tensor,
        weight:         &Tensor,
        bias_opt:       &Option<Tensor>,
        stride:         &[i32],
        padding:        &[i32],
        dilation:       &[i32],
        transposed:     bool,
        output_padding: &[i32],
        groups:         i64) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> bias_maybe_owned = borrow_from_optional_tensor(bias_opt);
      const Tensor& bias = *bias_maybe_owned;

      auto& ctx = globalContext();
      // See Note [Enabling Deterministic Operations]
      bool deterministic = ctx.deterministicCuDNN() || ctx.deterministicAlgorithms();
      return _convolution(input, weight, bias, stride, padding, dilation,
                              transposed, output_padding, groups,
                              ctx.benchmarkCuDNN(), deterministic, ctx.userEnabledCuDNN(), ctx.allowTF32CuDNN());
        */
}

pub fn convolution_overrideable(
        input:          &Tensor,
        weight:         &Tensor,
        bias_opt:       &Option<Tensor>,
        stride:         &[i32],
        padding:        &[i32],
        dilation:       &[i32],
        transposed:     bool,
        output_padding: &[i32],
        groups:         i64) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> bias_maybe_owned = borrow_from_optional_tensor(bias_opt);

      TORCH_CHECK_NOT_IMPLEMENTED(false, "convolution_overrideable not implemented. You are likely triggering this with tensor backend other than CPU/CUDA/MKLDNN, if this is intended, please use TORCH_LIBRARY_IMPL to override this function ");
        */
}

pub fn convolution_b(
        input_r:        &Tensor,
        weight_r:       &Tensor,
        bias_r_opt:     &Option<Tensor>,
        stride:         &[i32],
        padding:        &[i32],
        dilation:       &[i32],
        transposed:     bool,
        output_padding: &[i32],
        groups:         i64,
        benchmark:      bool,
        deterministic:  bool,
        cudnn_enabled:  bool,
        allow_tf32:     bool) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> bias_r_maybe_owned = borrow_from_optional_tensor(bias_r_opt);
      const Tensor& bias_r = *bias_r_maybe_owned;

      const bool input_is_mkldnn = input_r.is_mkldnn();
      auto input = input_r;
      auto weight = weight_r;
      auto bias = bias_r;
      auto k = weight.ndimension();
      IntArrayRef weight_sizes = weight.sizes();
      i64 dim = k - 2;

      TORCH_CHECK(dim > 0, "weight should have at least three dimensions");

      ConvParams params;
      params.stride = expand_param_if_needed(stride_, "stride", dim);
      params.padding = expand_param_if_needed(padding_, "padding", dim);
      params.dilation = expand_param_if_needed(dilation_, "dilation", dim);
      params.transposed = transposed_;
      params.output_padding = expand_param_if_needed(output_padding_, "output_padding", dim);
      params.groups = groups_;
      params.benchmark = benchmark;
      params.deterministic = deterministic;
      params.cudnn_enabled = cudnn_enabled;
      params.allow_tf32 = allow_tf32;

      check_shape_forward(input, weight_sizes, bias, params);

      if (input.size(0) == 0) {
        // don't send empty inputs through backends
        // but need to compute correct output size first and set up history for params
        vector<i64> o;
        if (!params.transposed) {
          o = conv_output_size(input.sizes(), weight_sizes, params.padding,
                               params.stride, params.dilation);
        } else {
          o = conv_input_size(input.sizes(), weight_sizes, params.padding,
                              params.output_padding, params.stride, params.dilation,
                              params.groups);
        }
        if (input_is_mkldnn && weight.is_mkldnn()) {
          // mkldnn will error on the below 0-dim handling code
          return empty_mkldnn(
              o,
              optTypeMetaToScalarType(input.options().dtype_opt()),
              input.options().layout_opt(),
              input.options().device_opt(),
              input.options().pinned_memory_opt());
        }
        auto weight_view = _unsafe_view(weight, -1);
        auto out = input*weight_view[0];
        if (bias.defined())
          out.add_(bias[0]);
        return out.view(o);
      }

      if (k == 3) {
        // avoid accidentally going through NHWC for permuted 3d input.
        if (!input_is_mkldnn) {
          input = input.contiguous();
        }
        params.view1d_as_2d();
        input = view4d(input);
        weight = view4d(weight);
      }

      MemoryFormat cudnn_memory_format = MemoryFormat::Contiguous;
      if (cudnn_conv_use_channels_last(input, weight)) {
        cudnn_memory_format = (k == 5) ? MemoryFormat::ChannelsLast3d : MemoryFormat::ChannelsLast;
      }

      Tensor output;
      if (params.is_depthwise(input, weight)) {
          /* output.resize_(output_size(input, weight)); */

          auto kernel_size = weight.sizes().slice(2);
          auto stride = params.stride;
          auto padding = params.padding;
          auto dilation = params.dilation;
          if (params.use_cudnn_depthwise(input, weight)) {
            output = cudnn_convolution(
                input.contiguous(cudnn_memory_format), weight,
                padding, stride, dilation, params.groups, params.benchmark, params.deterministic, params.allow_tf32);
            if (bias.defined()) {
              output.add_(reshape_bias(input.dim(), bias));
            }

          } else if (params.use_miopen(input, weight, bias.defined())){
            output = miopen_depthwise_convolution(
                input.contiguous(), weight, bias,
                padding, stride, dilation, params.groups, params.benchmark, params.deterministic);
          } else {
              if (input.ndimension() == 4) {
                  output = thnn_conv_depthwise2d(input.contiguous(), weight, kernel_size, bias, stride, padding, dilation);
              }
              else {
                 TORCH_CHECK(input.ndimension() == 5);
                 output = conv_depthwise3d(input.contiguous(), weight, kernel_size, bias, stride, padding, dilation);
              }
          }
      } else if (params.use_cudnn(input, weight)) {
        TORCH_CHECK(input.options().type_equal(weight.options()),
                 "Input type (", input.toString(), ") and weight type (", weight.toString(),
                 ") should be the same");
        TORCH_CHECK(!bias.defined() || (input.options().type_equal(bias.options())),
                 "Input type (", input.toString(), ") and bias type (", bias.toString(),
                 ") should be the same");

        if (params.transposed) {
          output = cudnn_convolution_transpose(
              input.contiguous(cudnn_memory_format), weight,
              params.padding, params.output_padding, params.stride, params.dilation, params.groups, params.benchmark, params.deterministic, params.allow_tf32);
          if (bias.defined()) {
            output.add_(reshape_bias(input.dim(), bias));
          }
        } else {
          output = cudnn_convolution(
              input.contiguous(cudnn_memory_format), weight,
              params.padding, params.stride, params.dilation, params.groups, params.benchmark, params.deterministic, params.allow_tf32);
          if (bias.defined()) {
            output.add_(reshape_bias(input.dim(), bias));
          }
        }
      } else if (params.use_miopen(input, weight, bias.defined())) {
        TORCH_CHECK(input.options().type_equal(weight.options()),
                 "Input type (", input.toString(), ") and weight type (", weight.toString(),
                 ") should be the same");
        TORCH_CHECK(!bias.defined() || (input.options().type_equal(bias.options())),
                 "Input type (", input.toString(), ") and bias type (", bias.toString(),
                 ") should be the same");

        if (params.transposed) {
          output = miopen_convolution_transpose(
              input.contiguous(), weight, bias,
              params.padding, params.output_padding, params.stride, params.dilation, params.groups, params.benchmark, params.deterministic);
        } else {
          output = miopen_convolution(
              input.contiguous(), weight, bias,
              params.padding, params.stride, params.dilation, params.groups, params.benchmark, params.deterministic);
        }
      } else if (params.use_mkldnn(input, weight)) {
    #if AT_MKLDNN_ENABLED()
        TORCH_CHECK(input.options().type_equal(weight.options())
                 || (input.is_mkldnn() && weight.device().is_cpu() && weight.scalar_type() == kFloat),
                 "Input type (", input.toString(), ") and weight type (", weight.toString(),
                 ") should be the same or input should be a MKLDNN tensor and weight is a dense tensor");
        TORCH_CHECK(!bias.defined() || (input.options().type_equal(bias.options()))
                 || (input.is_mkldnn() && bias.device().is_cpu() && bias.scalar_type() == kFloat),
                 "Input type (", input.toString(), ") and bias type (", bias.toString(),
                 ") should be the same or input should be a MKLDNN tensor and bias is a dense tensor");
        if (!input_is_mkldnn) {
          output = mkldnn_convolution(input.contiguous(), weight.contiguous(), bias.defined() ? bias.contiguous() : bias,
                                          params.padding, params.stride, params.dilation, params.groups);
        } else {
          // do not call contiguous on mkldnn tensor
          output = mkldnn_convolution(input, weight, bias,
                                          params.padding, params.stride, params.dilation, params.groups);
        }
    #endif
      } else if (params.use_xnnpack(input, weight, bias)) {
        // Using prepacked conv is preferred, but XNNPACK is still the fastest
        // option for NHWC.
        output = xnnpack::convolution2d(
            input,
            weight,
            bias,
            params.padding,
            params.stride,
            params.dilation,
            params.groups);
      } else if (params.use_cpu_depthwise3x3_winograd(input, weight, bias)) {
        output = convolution_depthwise3x3_winograd_stub(
            input.device().type(),
            input,
            weight,
            bias,
            params.stride,
            params.padding,
            params.groups);
      } else if (
            !params.transposed && (input.ndimension() == 5) &&
            (input.device().is_cpu()) &&
            !params.is_dilated()) {
          // fast path for grouped conv3d
          output = slow_conv3d(
              input,
              weight,
              weight.sizes().slice(2),
              bias,
              params.stride,
              params.padding);
      } else if (input.device().is_cpu() || input.is_cuda()) {
        if (params.groups == 1) {
          output = _convolution_nogroup(
              input.contiguous(), weight, bias, params.stride, params.padding, params.dilation, params.transposed, params.output_padding);
        } else {
          vector<Tensor> outputs(params.groups);
          input = input.contiguous();
          for (int g = 0; g < params.groups; ++g) {
            auto input_g = subtensor(input, 1, params.groups, g);
            auto weight_g = subtensor(weight, 0, params.groups, g);
            auto bias_g = subtensor(bias, 0, params.groups, g);
            outputs[g] = _convolution_nogroup(
                input_g, weight_g, bias_g, params.stride, params.padding, params.dilation, params.transposed, params.output_padding);
          }
          output = cat(outputs, 1);
        }
      } else {
        // Only reach here when input is backend with out-of-source implementation.
        output = convolution_overrideable(input, weight, bias, params.stride, params.padding, params.dilation, params.transposed, params.output_padding, params.groups);
      }

      if (k == 3) {
        output = view3d(output);
      }

      return output;
        */
}

pub fn convolution_c(
        input_r:        &Tensor,
        weight_r:       &Tensor,
        bias_r_opt:     &Option<Tensor>,
        stride:         &[i32],
        padding:        &[i32],
        dilation:       &[i32],
        transposed:     bool,
        output_padding: &[i32],
        groups:         i64,
        benchmark:      bool,
        deterministic:  bool,
        cudnn_enabled:  bool) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> bias_r_maybe_owned = borrow_from_optional_tensor(bias_r_opt);
      const Tensor& bias_r = *bias_r_maybe_owned;

      return _convolution(input_r, weight_r, bias_r, stride_, padding_, dilation_, transposed_, output_padding_, groups_, benchmark, deterministic, cudnn_enabled, globalContext().allowTF32CuDNN());
        */
}

/**
  | A generic function for convolution
  | implementations which don't natively
  | implement groups (e.g., not CuDNN).
  |
  */
pub fn convolution_nogroup(
        input:          &Tensor,
        weight:         &Tensor,
        bias_opt:       &Option<Tensor>,
        stride:         &[i32],
        padding:        &[i32],
        dilation:       &[i32],
        transposed:     bool,
        output_padding: &[i32]) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> bias_maybe_owned = borrow_from_optional_tensor(bias_opt);
      const Tensor& bias = *bias_maybe_owned;

      ConvParams params;
      params.stride = stride.vec();
      params.padding = padding.vec();
      params.dilation = dilation.vec();
      params.transposed = transposed;
      params.output_padding = output_padding.vec();
      params.groups = 1;
      params.benchmark = false;
      params.deterministic = false;
      params.cudnn_enabled = false;

      auto dim = input.ndimension();
      auto dilated = params.is_dilated();
      auto kernel_size = weight.sizes().slice(2);

      if (params.transposed) {
        if (dim == 4) {
          return slow_conv_transpose2d(
              input, weight, kernel_size, bias,
              stride, padding, output_padding, dilation);
        } else if (dim == 5) {
          return slow_conv_transpose3d(
            input, weight, kernel_size, bias,
            stride, padding, output_padding, dilation);
          }
      } else {  /* Not transposed */
        if (dim == 4) {
          if (dilated) {
            return slow_conv_dilated2d(
                input, weight, kernel_size, bias,
                stride, padding, dilation);
          } else {  /* dim == 4, non-dilated */
            if (params.use_nnpack(input, weight)) {
    #if AT_NNPACK_ENABLED()
              return _nnpack_spatial_convolution(
                  input, weight, bias, padding, stride);
    #endif
            } else {
              /* CPU implementation has specialized MM kernels
                 for non-dilated case here */
              return thnn_conv2d(
                  input, weight, kernel_size, bias,
                  stride, padding);
            }
          }
        } else if (dim == 5 && (input.is_cuda() || dilated)) {
          return slow_conv_dilated3d(
              input, weight, kernel_size, bias,
              stride, padding, dilation);
        } else if (dim == 5) { /* dim == 5, CPU, non-dilated */
          /* CPU implementation has specialized MM kernels
             for non-dilated case here */

          // This path is already overwritten with the fast impl in _convolution
          // See: https://github.com/pytorch/pytorch/pull/3635
          return slow_conv3d(
              input, weight, kernel_size, bias,
              stride, padding);
        }
      }

      AT_ERROR("unsupported ConvNd parameters");
        */
}

pub fn convolution_backward_overrideable(
        grad_output:    &Tensor,
        input:          &Tensor,
        weight:         &Tensor,
        stride:         &[i32],
        padding:        &[i32],
        dilation:       &[i32],
        transposed:     bool,
        output_padding: &[i32],
        groups:         i64,
        output_mask:    [bool; 3]) -> (Tensor,Tensor,Tensor) {
    
    todo!();
        /*
            AT_ERROR("You are likely triggering this with tensor backend other than CPU/CUDA/MKLDNN, if this is intended, please use TORCH_LIBRARY_IMPL to override this function ");
      return tuple<Tensor, Tensor, Tensor>(
              empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT),
              empty_like(weight, LEGACY_CONTIGUOUS_MEMORY_FORMAT),
              empty({}));
        */
}

pub fn subvariable(
        var:    &Tensor,
        dim:    i32,
        groups: i32,
        g:      i32) -> Tensor {
    
    todo!();
        /*
            i64 n = var.sizes()[dim] / groups;
      auto result = var.narrow(dim, n * g, n);
      return result;
        */
}

pub fn convolution_double_backward(
        ggi_opt:        &Option<Tensor>,
        ggw_r_opt:      &Option<Tensor>,
        ggb_opt:        &Option<Tensor>,
        go_r:           &Tensor,
        weight_r:       &Tensor,
        input:          &Tensor,
        stride:         &[i32],
        padding:        &[i32],
        dilation:       &[i32],
        transposed:     bool,
        output_padding: &[i32],
        groups:         i64,
        benchmark:      bool,
        deterministic:  bool,
        cudnn_enabled:  bool,
        allow_tf32:     bool,
        output_mask:    [bool; 3]) -> (Tensor,Tensor,Tensor) {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> ggI_maybe_owned = borrow_from_optional_tensor(ggI_opt);
      const Tensor& ggI = *ggI_maybe_owned;
      const Tensor& ggW_r = value_or_else(ggW_r_opt, [] {return Tensor();});
      const Tensor& ggb = value_or_else(ggb_opt, [] {return Tensor();});

      auto ggW = ggW_r;
      auto gO = gO_r;
      auto weight = weight_r;

      ConvParams params;
      params.stride = stride_.vec();
      params.padding = padding_.vec();
      params.dilation = dilation_.vec();
      params.transposed = transposed_;
      params.output_padding = output_padding_.vec();
      // TODO: hacky way of inferring the groups number for grouped Conv3D
      // See: https://github.com/pytorch/pytorch/pull/36355
      if (!params.transposed && input.dim() > 4) {
        params.groups = input.size(1) / weight.size(1);
      } else {
        params.groups = groups_;
      }
      params.benchmark = benchmark;
      params.deterministic = deterministic;
      params.cudnn_enabled = cudnn_enabled;
      params.allow_tf32 = allow_tf32;

      // Compute ggO = conv(ggI, w) + conv(i, ggW) + ggb
      Tensor ggO;
      if (input.numel() != 0) {
        if (ggI.defined()) {
          if (weight.is_cuda()) {
            weight = weight.contiguous();
          }
          ggO = _convolution(ggI, weight, Tensor(), params.stride, params.padding, params.dilation, params.transposed, params.output_padding, params.groups, params.benchmark, params.deterministic, params.cudnn_enabled, params.allow_tf32);
        }

        if (ggW.defined()) {
          if (ggW.is_cuda()) {
            ggW = ggW.contiguous();
          }
          auto ggW_term = _convolution(input, ggW, Tensor(), params.stride, params.padding, params.dilation, params.transposed, params.output_padding, params.groups, params.benchmark, params.deterministic, params.cudnn_enabled, params.allow_tf32);
          if (ggO.defined()) {
            ggO = ggO + ggW_term;
          } else {
            ggO = ggW_term;
          }
        }
      }

      if (ggb.defined()) {
        // View as (1, ggb.size(0), 1, 1...)

        // Expand
        vector<i64> new_size(gO.ndimension(), 1);
        new_size[1] = ggb.sizes()[0];
        auto ggb_contiguous = ggb.contiguous();
        auto ggb_view = ggb_contiguous.view(new_size);

        // Expand
        auto ggb_expanded = ggb_view.expand(gO.sizes());

        if (ggO.defined()) {
          ggO = ggO + ggb_expanded;
        } else {
          ggO = ggb_expanded;
        }
      }

      // Compute gW = conv(ggI, gO)
      Tensor gW;
      if (ggI.defined()) {

        // Modified params with correct padding
        ConvParams gw_conv_params(params);

        // Disable groups as they are handled separately
        auto groups = gw_conv_params.groups;
        gw_conv_params.groups = 1;
        swap(gw_conv_params.dilation, gw_conv_params.stride);

        // Transpose gO and ggI to accumulate over batch
        auto gOt = gO.transpose(0, 1);
        auto ggIt = ggI.transpose(0, 1);

        Tensor gWt;
        // Compute conv
        if (input.numel() != 0) {
          if (groups == 1) {

            if (gOt.is_cuda()) {
              gOt = gOt.contiguous();
            }
            // Compute conv
            if (params.transposed) {
              gw_conv_params.transposed = false;
              gWt = _convolution(gOt, ggIt, Tensor(), gw_conv_params.stride, gw_conv_params.padding, gw_conv_params.dilation, gw_conv_params.transposed, gw_conv_params.output_padding, gw_conv_params.groups, gw_conv_params.benchmark, gw_conv_params.deterministic, gw_conv_params.cudnn_enabled, params.allow_tf32);
            } else {
              gWt = _convolution(ggIt, gOt, Tensor(), gw_conv_params.stride, gw_conv_params.padding, gw_conv_params.dilation, gw_conv_params.transposed, gw_conv_params.output_padding, gw_conv_params.groups, gw_conv_params.benchmark, gw_conv_params.deterministic, gw_conv_params.cudnn_enabled, params.allow_tf32);
            }
          } else {
            vector<Tensor> gWt_list(groups);
            for (int g = 0; g < groups; ++g) {
              auto ggIt_g = subvariable(ggIt, 0, groups, g);
              auto gOt_g = subvariable(gOt, 0, groups, g);
              if (gOt_g.is_cuda()) {
                gOt_g = gOt_g.contiguous();
              }

              // Compute conv
              if (params.transposed) {
                gw_conv_params.transposed = false;
                gWt_list[g] = _convolution(gOt_g, ggIt_g, Tensor(), gw_conv_params.stride, gw_conv_params.padding, gw_conv_params.dilation, gw_conv_params.transposed, gw_conv_params.output_padding, gw_conv_params.groups, gw_conv_params.benchmark, gw_conv_params.deterministic, gw_conv_params.cudnn_enabled, params.allow_tf32);
              } else {
                gWt_list[g] = _convolution(ggIt_g, gOt_g, Tensor(), gw_conv_params.stride, gw_conv_params.padding, gw_conv_params.dilation, gw_conv_params.transposed, gw_conv_params.output_padding, gw_conv_params.groups, gw_conv_params.benchmark, gw_conv_params.deterministic, gw_conv_params.cudnn_enabled, params.allow_tf32);
              }
            }

            gWt = cat(gWt_list, 1);
          }

          // Transpose gW to match chan_in and chan_out
          gW = gWt.transpose(0, 1);

          // narrow gW to only relevant portion
          // we do it this way instead of narrowing the input itself because
          // the ConvForward kernels don't support asymmetric padding.
          auto gW_size = gW.sizes();
          auto w_size = weight.sizes();
          for (usize i = 2; i < gW_size.size(); ++i) {
            if (gW_size[i] > w_size[i]) {
                gW = gW.narrow(i, 0, w_size[i]);
                gW_size = gW.sizes();
            }
          }
        }
      }

      // Compute gI = convT(gO, ggW) if !transposed
      //         gI = conv(gO, ggw)  if transposed
      Tensor gI;
      if (input.numel() != 0) {
        if (ggW.defined()) {
          ConvParams gi_conv_params(params);
          gi_conv_params.transposed = !params.transposed;

          if (params.transposed) {
            if (gO.is_cuda()) {
              gO = gO.contiguous();
            }
            gI = _convolution(gO, ggW, Tensor(), gi_conv_params.stride, gi_conv_params.padding, gi_conv_params.dilation, gi_conv_params.transposed, gi_conv_params.output_padding, gi_conv_params.groups, gi_conv_params.benchmark, gi_conv_params.deterministic, gi_conv_params.cudnn_enabled, params.allow_tf32);

            // narrow gI to only relevant portion
            // we do it this way because negative output_padding is not supported
            // TODO: figure out if we can narrow gO and save some compute,
            // rather than narrowing the computed gI
            auto gI_size = gI.sizes();
            auto i_size = input.sizes();
            for (usize i = 2; i < gI_size.size(); ++i) {
              if (gI_size[i] > i_size[i]) {
                gI = gI.narrow(i, 0, i_size[i]);
                gI_size = gI.sizes();
              }
            }
          } else {
            // calculate output_padding
            // TODO: figure out why this needs to be computed...
            auto kernel_size = weight.sizes().slice(2);
            auto input_shape = input.sizes().slice(2);
            auto grad_output_shape = gO.sizes().slice(2);

            if (kernel_size.size() == 1) {
              auto expected_input_shape = (kernel_size[0] - 1) * gi_conv_params.dilation[1]
                - 2 * gi_conv_params.padding[1]
                + (gi_conv_params.stride[1] * (grad_output_shape[0] - 1) + 1);
              if (expected_input_shape != input_shape[0]) {
                gi_conv_params.output_padding[1] = input_shape[0] - expected_input_shape;
              }
            } else {
              for(usize i = 0; i < kernel_size.size(); ++i) {
                // Check if whole input has been used or not
                auto expected_input_shape = (kernel_size[i] - 1) * gi_conv_params.dilation[i]
                  - 2 * gi_conv_params.padding[i]
                  + (gi_conv_params.stride[i] * (grad_output_shape[i] - 1) + 1);
                if (expected_input_shape != input_shape[i]) {
                  gi_conv_params.output_padding[i] = input_shape[i] - expected_input_shape;
                }
              }
            }

            if (gO.is_cuda()) {
              gO = gO.contiguous();
            }
            gI = _convolution(gO, ggW, Tensor(), gi_conv_params.stride, gi_conv_params.padding, gi_conv_params.dilation, gi_conv_params.transposed, gi_conv_params.output_padding, gi_conv_params.groups, gi_conv_params.benchmark, gi_conv_params.deterministic, gi_conv_params.cudnn_enabled, params.allow_tf32);
          }
        }
      }

      return tuple<Tensor,Tensor,Tensor>{ggO, gI, gW};
        */
}
