crate::ix!();

/**
  | PackedWeight struct for QNNPACK stores the
  | original Weight and Bias as QNNPACK currently
  | does not support an unpack function.
  |
  | For PyTorch Mobile, once the model is scripted
  | and serialized we don't need to call unpack, so
  | we can save some memory by checking for this
  | case and free the original weights after
  | packing.
  |
  | Input scale is set to null in pre-pack
  | step. QNNPACK needs bias quantized with input
  | scale which is available at runtime in
  | pytorch. During runtime if input scale value
  | changes then we requantize bias with the
  | updated scale. For inference we expect the
  | graph to be static so the input scale should
  | not change across consecutive inference calls.
  |
  */
#[cfg(USE_PYTORCH_QNNPACK)]
pub struct PackedLinearWeightsQnnp {
    base:                  LinearPackedParamsBase,
    w:                     Box<QnnPackPackBMatrix>,
    orig_weight:           Tensor,
    bias:                  Tensor,
    input_scale:           Option<f64>,
    w_scales:              Tensor,
    w_zero_points:         Vec<u8>,
    requantization_scales: Vec<f32>,
}

impl PackedLinearWeightsQnnp {
    
    #[cfg(USE_PYTORCH_QNNPACK)]
    pub fn unpack(&mut self) -> (Tensor,Option<Tensor>) {
        
        todo!();
        /*
            TORCH_CHECK(
          orig_weight.defined(),
          "Cannot unpack weights. "
          "Call globalContext()::setReleaseOriginalWeights(false) before packing or loading to enable unpacking.");
      return tuple<Tensor, optional<Tensor>>(orig_weight, bias_);
        */
    }
    
    #[cfg(USE_PYTORCH_QNNPACK)]
    pub fn prepack(&mut self, 
        weight:  Tensor,
        bias_in: Option<Tensor>) -> IntrusivePtr<LinearPackedParamsBase> {
        
        todo!();
        /*
            TORCH_CHECK(
          weight.dim() == 2,
          "quantized::linear_prepack (qnnpack): Weight tensor rank should be == 2");

      i64 rows_w = weight.size(0);
      Tensor bias_fp32;
      if (bias_in.has_value()) {
        bias_fp32 = bias_in.value();
      } else {
        bias_fp32 = zeros(rows_w, weight.options().dtype(kFloat));
      }
      TORCH_CHECK(
          !bias_fp32.defined() ||
              (bias_fp32.ndimension() == 1 && bias_fp32.size(0) == rows_w),
          "quantized::linear_prepack (qnnpack): Given weight of size ",
          weight.sizes(),
          ", expected bias to be 1-dimensional with ",
          rows_w,
          " elements",
          ", but got bias of size ",
          bias_fp32.sizes(),
          " instead");

      Tensor weight_contig = weight.contiguous();
      vector<u8> w_zero_points;
      Tensor  w_scales;
      tie(w_zero_points, w_scales) =
          make_zero_points_and_scales_tensor(weight_contig);

      native::initQNNPACK();

      // We set the pre-packed linear weights to nullptr below as we call pre-pack
      // during the first invocation of operator run. Refer to qlinear.cpp for more
      // details. TODO Update to actually call pre-pack here once bias is removed
      // from pre-packing step.
      auto wt_ptr = make_intrusive<PackedLinearWeightsQnnp>(
          nullptr,
          weight_contig, /* i8 weight */
          bias_fp32.contiguous(), /* fp32 bias */
          nullopt, /* input_scale */
          w_scales,
          move(w_zero_points));
      return wt_ptr;
        */
    }
    
    #[cfg(USE_PYTORCH_QNNPACK)]
    pub fn apply_impl<const ReluFused: bool>(&mut self, 
        input:             Tensor,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor {
    
        todo!();
        /*
            TORCH_CHECK(
          input.dim() >= 2,
          "quantized::linear(): Input tensor rank should be >= 2");
      auto input_contig = input.contiguous();

      auto packB = w.get();
      usize rows_w = bias_.size(0);
      usize cols_w = input_contig.size(input_contig.dim() - 1);
      auto input_scale = input_contig.q_scale();

      if (!this->input_scale.has_value() ||
          this->input_scale.value() != input_scale) {
        // Get the original weight and adjust it to uint8 from int8
        auto weight_contig = orig_weight;
        auto bias_fp32 = bias_;
        i8* w_data = (i8*)weight_contig.data_ptr<qint8>();

        float* weight_scales_data = w_scales.data_ptr<float>();
        // We calculate requant scale here as the vector holding the requant scale
        // is owned by this module. The pointer is then passed to qnnpack backend.
        generate_requantization_scales(
            w_scales, input_scale, output_scale, requantization_scales);

        Tensor qnnp_weight = _empty_affine_quantized(
            weight_contig.sizes(),
            device(kCPU).dtype(kQUInt8),
            weight_scales_data[0],
            w_zero_points[0]);
        auto* qnnp_w_data = qnnp_weight.data_ptr<quint8>();
        auto wt_numel = weight_contig.numel();
        for (int i = 0; i < wt_numel; ++i) {
          qnnp_w_data[i] = static_cast<quint8>(w_data[i] + 128);
        }
        // Original bias was float, so we requantize it here.
        const bool is_per_channel = orig_weight.qscheme() == kPerChannelAffine;
        Tensor qbias;
        // Original bias was float, so we requantize it here.
        if (is_per_channel) {
          Tensor bias_quant_scales =
              weight_contig.q_per_channel_scales() * input_scale;
          Tensor bias_zp = zeros(bias_quant_scales.sizes(), kInt);
          qbias = native::quantize_per_channel_cpu(
              bias_fp32, bias_quant_scales, bias_zp, 0, kQInt32);
        } else {
          qbias = native::quantize_per_tensor(
              bias_fp32, weight_contig.q_scale() * input_scale, 0, kQInt32);
        }

        // Update the input scale to not pack again.
        this->input_scale = input_scale;
        w.reset();
        w = make_unique<qnnpack::PackBMatrix>(
            cols_w /* input_channels */,
            rows_w /* output_channels */,
            w_zero_points.data(),
            requantization_scales.data(),
            reinterpret_cast<u8*>(qnnp_w_data),
            reinterpret_cast<i32*>(qbias.data_ptr<qint32>()));
        packB = w.get();
        if (globalContext().releaseWeightsWhenPrepacking()) {
          // On mobile, we release the original weight by resetting the intrusive_ptr.
          // Calling unpack after this will throw an assertion.
          orig_weight.reset();
        }
      }

      usize rows_input = 1;
      usize cols_input = input_contig.size(input_contig.dim() - 1);
      for (usize i = 0; i < input_contig.dim() - 1; ++i) {
        rows_input *= input_contig.size(i);
      }

      TORCH_CHECK(
          cols_input == cols_w,
          "quantized::linear(): input size does not match weight dimension 1 size: \
             got ",
          cols_input,
          " but expected ",
          cols_w);

      // Allocate output Tensor and a buffer for QNNPACK to use
      // The resulting matrix here is 2-D, let's view it with the original
      // left hand dimensions of the input. Here are two examples:
      // 1. If the input tensor is {M, K}, the output tensor is {M, N}.
      // 2. If the input tensor is {b, M, K}, the output tensor is {b, M, N}.
      vector<i64> out_sizes = input.sizes().vec();
      out_sizes.back() = static_cast<long>(rows_w);
      Tensor output = _empty_affine_quantized(
          out_sizes,
          input.options(),
          output_scale,
          output_zero_point);

      auto output_min = ReluFused
          ? activationLimits(output_scale, output_zero_point, Activation::RELU)
                .first
          : u8::min;
      auto output_max = ReluFused
          ? activationLimits(output_scale, output_zero_point, Activation::RELU)
                .second
          : u8::max;
      TORCH_INTERNAL_ASSERT(packB != nullptr, "Packed Weights are NULL");
      const pytorch_qnnp_status runStatus = qnnpack::qnnpackLinear(
          rows_input /* batch_size */,
          cols_input /* input_channels */,
          rows_w /* output_channels */,
          input_contig.q_zero_point(),
          w_zero_points.data(),
          requantization_scales.data(),
          output_zero_point,
          output_min,
          output_max,
          (u8*)input_contig.data_ptr<quint8>(),
          cols_input /* input_stride */,
          packB->getPackedWeights(),
          (u8*)output.data_ptr<quint8>(),
          rows_w /* output_stride */,
          // TODO (Ashkan): Disabling temporarily.
          // Throws a floating point exception with OSS pthreadpool.
          pthreadpool_() /* threadpool */);

      TORCH_INTERNAL_ASSERT(
          runStatus == pytorch_qnnp_status_success,
          "failed to run QNNPACK Linear operator");

      return output;
        */
    }
    
    #[cfg(USE_PYTORCH_QNNPACK)]
    pub fn apply(&mut self, 
        input:             Tensor,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor {
        
        todo!();
        /*
            return apply_impl<false>(move(input), output_scale, output_zero_point);
        */
    }
    
    #[cfg(USE_PYTORCH_QNNPACK)]
    pub fn apply_relu(&mut self, 
        input:             Tensor,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor {
        
        todo!();
        /*
            return apply_impl<true>(move(input), output_scale, output_zero_point);
        */
    }
    
    #[cfg(USE_PYTORCH_QNNPACK)]
    pub fn new(
        w:           Box<QnnPackPackBMatrix>,
        orig_weight: Tensor,
        bias:        Tensor,
        input_scale: Option<f64>,
        w_scales:    Tensor,
        w_zps:       Vec<u8>) -> Self {
    
        todo!();
        /*


            : w(move(w)),
            orig_weight(move(orig_weight)),
            bias_(native::mobile::allocate_padded_contiguous_if_needed(
                bias, bias.suggest_memory_format())),
            input_scale(move(input_scale)),
            w_scales(w_scales),
            w_zero_points(move(w_zps))
        */
    }
    
    #[cfg(USE_PYTORCH_QNNPACK)]
    pub fn apply(&mut self, 
        input:             Tensor,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor {
        
        todo!();
        /*
        
        */
    }
    
    #[cfg(USE_PYTORCH_QNNPACK)]
    pub fn apply_relu(&mut self, 
        input:             Tensor,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor {
        
        todo!();
        /*
        
        */
    }
    
    #[cfg(USE_PYTORCH_QNNPACK)]
    pub fn apply_dynamic(&mut self, 
        input:        Tensor,
        reduce_range: bool) -> Tensor {
        let reduce_range: bool = reduce_range.unwrap_or(false);

        todo!();
        /*
        
        */
    }
    
    #[cfg(USE_PYTORCH_QNNPACK)]
    pub fn apply_dynamic_relu(&mut self, 
        input:        Tensor,
        reduce_range: bool) -> Tensor {
        let reduce_range: bool = reduce_range.unwrap_or(false);

        todo!();
        /*
        
        */
    }
    
    #[cfg(USE_PYTORCH_QNNPACK)]
    pub fn unpack(&mut self) -> (Tensor,Option<Tensor>) {
        
        todo!();
        /*
        
        */
    }
    
    #[cfg(USE_PYTORCH_QNNPACK)]
    pub fn bias(&mut self) -> Option<Tensor> {
        
        todo!();
        /*
            return bias_;
        */
    }
    
    #[cfg(USE_PYTORCH_QNNPACK)]
    pub fn prepack(
        weight: Tensor,
        bias:   Option<Tensor>) -> IntrusivePtr<LinearPackedParamsBase> {
        
        todo!();
        /*
        
        */
    }
    
    #[cfg(USE_PYTORCH_QNNPACK)]
    pub fn apply_impl<const ReluFused: bool>(&mut self, 
        input:             Tensor,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor {
    
        todo!();
        /*
        
        */
    }
    
    #[cfg(USE_PYTORCH_QNNPACK)]
    pub fn apply_dynamic_impl<const ReluFused: bool>(&mut self, input: Tensor) -> Tensor {
    
        todo!();
        /*
        
        */
    }
    
    #[cfg(USE_PYTORCH_QNNPACK)]
    pub fn apply_dynamic_impl<const ReluFused: bool>(&mut self, input: Tensor) -> Tensor {
    
        todo!();
        /*
            using Tensor;
      TORCH_CHECK(
          input.dim() >= 2,
          "The dimension of input tensor should be larger than or equal to 2");
      auto input_contig = input.contiguous();
      // C(output) = A(input) x B(weight), where C, A, B are M x N, M x K, K x N
      // matrices, respectively.

      auto packB = w.get();
      usize rows_w = bias_.size(0);
      usize cols_w = input_contig.size(input_contig.dim() - 1);

      Tensor bias_vec = bias_;

      TORCH_CHECK(bias_vec.dim() == 1, "bias should be a vector (1D Tensor)");

      auto bias_contig = bias_vec.contiguous();
      const float* bias_ptr = bias_contig.data_ptr<float>();

      // Calculate statistics for quantization of input Tensor
      // TODO: optimized kernel
      float x_min;
      float x_max;
      if (input.numel() > 0) {
        x_min = input_contig.min().item<float>();
        x_max = input_contig.max().item<float>();
      } else {
        // On empty input, no output data will be generated,
        // so use arbitrary qparams.
        x_min = 0;
        x_max = 0;
      }

      auto q_params = quant_utils::ChooseQuantizationParams(
          /*min=*/x_min,
          /*max=*/x_max,
          /*qmin=*/0,
          /*qmax=*/255);
      float* weight_scales_data = w_scales.data_ptr<float>();
      if (!input_scale.has_value() || input_scale.value() != q_params.scale) {
        generate_requantization_scales(
            w_scales, q_params.scale, 1.f, requantization_scales);
      }

      if (!input_scale.has_value()) {
        // Get the original weight and adjust it to uint8 from int8
        auto weight_contig = orig_weight;

        // TODO(kimishpatel), we are allocating affine_quantized regardless of per channel or not.
        // This allocation is actually used only for packing weight and thus will be freed.
        // Still we should be consistent. Fix this.
        Tensor qnnp_weight = _empty_affine_quantized(
            weight_contig.sizes(),
            device(kCPU).dtype(kQUInt8),
            weight_scales_data[0],
            w_zero_points[0]);
        auto* qnnp_w_data = qnnp_weight.data_ptr<quint8>();
        i8* w_data = (i8*)weight_contig.data_ptr<qint8>();
        auto wt_numel = weight_contig.numel();
        for (int i = 0; i < wt_numel; ++i) {
          qnnp_w_data[i] = static_cast<quint8>(w_data[i] + 128);
        }

        // Pass in nullptr for bias, as we pass FP32 bias to run function.
        w.reset();
        w = make_unique<qnnpack::PackBMatrix>(
            cols_w /* input_channels */,
            rows_w /* output_channels */,
            w_zero_points.data(),
            requantization_scales.data(),
            (u8*)qnnp_w_data,
            nullptr);
        packB = w.get();
        if (globalContext().releaseWeightsWhenPrepacking()) {
          // On mobile, we release the original weight by resetting the intrusive_ptr.
          // Calling unpack after this will throw an assertion.
          orig_weight.reset();
        }
      }

      // Update the input scale to not pack weights again.
      // as well as to avoid repopulating requant scale if scale has not changed.
      input_scale = q_params.scale;

      // Quantize input
      Tensor q_input = quantize_per_tensor(
          input_contig, q_params.scale, q_params.zero_point, kQUInt8);

      // The resulting matrix here is 2-D, let's view it with the original
      // left hand dimensions of the input. Here are two examples:
      // 1. If the input tensor is {M, K}, the output tensor is {M, N}.
      // 2. If the input tensor is {b, M, K}, the output tensor is {b, M, N}.
      vector<i64> out_sizes = input.sizes().vec();
      out_sizes.back() = rows_w;

      auto output = empty(out_sizes, input.options().dtype(kFloat));

      usize rows_input = 1;
      usize cols_input = input_contig.size(input_contig.dim() - 1);
      for (usize i = 0; i < input_contig.dim() - 1; ++i) {
        rows_input *= input_contig.size(i);
      }
      pytorch_qnnp_status runStatus = qnnpack::qnnpackLinearDynamic(
          rows_input /* batch_size */,
          cols_input /* input_channels */,
          rows_w /* output_channels */,
          q_input.q_zero_point(),
          w_zero_points.data(),
          /* for dynamic should really be called dequant scale */
          requantization_scales.data(),
          (u8*)q_input.data_ptr<quint8>(),
          cols_input /* input_stride */,
          packB->getPackedWeights(),
          bias_ptr,
          output.data_ptr<float>(),
          rows_w /* output_stride */,
          pthreadpool_() /* threadpool */);

      TORCH_INTERNAL_ASSERT(
          runStatus == pytorch_qnnp_status_success,
          "failed to run QNNPACK Linear operator");
      return output;
        */
    }
    
    #[cfg(USE_PYTORCH_QNNPACK)]
    pub fn apply_dynamic(&mut self, 
        input:        Tensor,
        reduce_range: bool) -> Tensor {
        
        todo!();
        /*
            return apply_dynamic_impl</*ReluFused=*/false>(move(input));
        */
    }
    
    #[cfg(USE_PYTORCH_QNNPACK)]
    pub fn apply_dynamic_relu(&mut self, 
        input:        Tensor,
        reduce_range: bool) -> Tensor {
        
        todo!();
        /*
            return apply_dynamic_impl</*ReluFused=*/true>(move(input));
        */
    }
}
