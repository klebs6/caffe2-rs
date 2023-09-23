crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/QTensor.cpp]

pub fn quantize_per_tensor(
        self_:      &Tensor,
        scale:      f64,
        zero_point: i64,
        dtype:      ScalarType) -> Tensor {
    
    todo!();
        /*
            auto quantizer = make_per_tensor_affine_quantizer(scale, zero_point, dtype);
      return quantizer->quantize(self);
        */
}

pub fn quantize_per_tensor_tensor_qparams(
        self_:      &Tensor,
        scale:      &Tensor,
        zero_point: &Tensor,
        dtype:      ScalarType) -> Tensor {
    
    todo!();
        /*
            auto quantizer = make_per_tensor_affine_quantizer(scale.item().toDouble(), zero_point.item().toLong(), dtype);
      return quantizer->quantize(self);
        */
}

pub fn quantize_per_tensor_list_cpu(
        tensors:     &[Tensor],
        scales:      &Tensor,
        zero_points: &Tensor,
        dtype:       ScalarType) -> Vec<Tensor> {
    
    todo!();
        /*
            vector<Tensor> quantized_tensors;
      for (const auto i : irange(tensors.size())) {
        quantized_tensors.push_back(quantize_per_tensor(
            tensors[i],
            scales[i].item<double>(),
            zero_points[i].item<i64>(),
            dtype));
      }
      return quantized_tensors;
        */
}

pub fn quantize_per_channel_cpu(
        self_:       &Tensor,
        scales:      &Tensor,
        zero_points: &Tensor,
        axis:        i64,
        dtype:       ScalarType) -> Tensor {
    
    todo!();
        /*
            auto quantizer = make_per_channel_affine_quantizer(scales, zero_points, axis, dtype);
      return quantizer->quantize(self);
        */
}

pub fn dequantize_cpu(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(!self.is_quantized());
      return self.to(kFloat);
        */
}


pub fn dequantize_quantized_cpu(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            return get_qtensorimpl(self)->quantizer()->dequantize(self);
        */
}


pub fn dequantize_tensors_quantized_cpu(tensors: &[Tensor]) -> Vec<Tensor> {
    
    todo!();
        /*
            vector<Tensor> dequantized_tensors;
      for (const auto i : irange(tensors.size())) {
        dequantized_tensors.push_back(tensors[i].dequantize());
      }
      return dequantized_tensors;
        */
}


pub fn q_scale_quant(self_: &Tensor) -> f64 {
    
    todo!();
        /*
            auto quantizer = get_qtensorimpl(self)->quantizer();
      TORCH_CHECK(quantizer->qscheme() == kPerTensorAffine);
      return static_cast<PerTensorAffineQuantizer*>(quantizer.get())->scale();
        */
}


pub fn q_zero_point_quant(self_: &Tensor) -> i64 {
    
    todo!();
        /*
            auto quantizer = get_qtensorimpl(self)->quantizer();
      TORCH_CHECK(quantizer->qscheme() == kPerTensorAffine);
      return static_cast<PerTensorAffineQuantizer*>(quantizer.get())->zero_point();
        */
}


pub fn q_per_channel_scales(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            auto quantizer = get_qtensorimpl(self)->quantizer();
      TORCH_CHECK(quantizer->qscheme() == kPerChannelAffine || quantizer->qscheme() == kPerChannelAffineFloatQParams);
      return static_cast<PerChannelAffineQuantizer*>(quantizer.get())->scales();
        */
}


pub fn q_per_channel_zero_points(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            auto quantizer = get_qtensorimpl(self)->quantizer();
      TORCH_CHECK(quantizer->qscheme() == kPerChannelAffine || quantizer->qscheme() == kPerChannelAffineFloatQParams);
      return static_cast<PerChannelAffineQuantizer*>(quantizer.get())->zero_points();
        */
}


pub fn q_per_channel_axis(self_: &Tensor) -> i64 {
    
    todo!();
        /*
            auto quantizer = get_qtensorimpl(self)->quantizer();
      TORCH_CHECK(quantizer->qscheme() == kPerChannelAffine || quantizer->qscheme() == kPerChannelAffineFloatQParams);
      return static_cast<PerChannelAffineQuantizer*>(quantizer.get())->axis();
        */
}


pub fn make_per_channel_quantized_tensor_cpu(
        self_:       &Tensor,
        scales:      &Tensor,
        zero_points: &Tensor,
        axis:        i64) -> Tensor {
    
    todo!();
        /*
            Tensor dst = _empty_per_channel_affine_quantized(
          self.sizes(),
          scales,
          zero_points,
          axis,
          self.options().dtype(toQIntType(self.scalar_type())));
      Tensor self_contig = self.contiguous();
      AT_DISPATCH_QINT_TYPES(
          dst.scalar_type(), "per_channel_affine_qtensor", [&]() {
            underlying_t* self_data = self_contig.data_ptr<underlying_t>();
            underlying_t* dst_data =
                reinterpret_cast<underlying_t*>(dst.data_ptr<Scalar>());
            if (self.numel() > 0) {
              memcpy(dst_data, self_data, self.nbytes());
            }
          });
      return dst;
        */
}


pub fn set_storage_quantized<'a>(
        self_:          &mut Tensor,
        storage:        Storage,
        storage_offset: i64,
        sizes:          &[i32],
        strides:        &[i32]) -> &'a mut Tensor {
    
    todo!();
        /*
            auto* self_ = self.unsafeGetTensorImpl();
      self_->set_storage_keep_dtype(storage);
      self_->set_storage_offset(storage_offset);
      self_->set_sizes_and_strides(sizes, strides);
      return self;
        */
}


pub fn qscheme_quant(self_: &Tensor) -> QScheme {
    
    todo!();
        /*
            auto quantizer = get_qtensorimpl(self)->quantizer();
      return quantizer->qscheme();
        */
}


pub fn quantized_clone(
        self_:                  &Tensor,
        optional_memory_format: Option<MemoryFormat>) -> Tensor {
    
    todo!();
        /*
            auto memory_format =
          optional_memory_format.value_or(MemoryFormat::Contiguous);

      // TODO: To support all features of MemoryFormat::Preserve we need to add
      // _empty_affine_quantized_strided function and use it similarly to
      // Tensor clone(const Tensor& src, optional<MemoryFormat>
      // optional_memory_format) if (self.is_non_overlapping_and_dense()) ->
      // _empty_affine_quantized_strided
      if (memory_format == MemoryFormat::Preserve) {
        memory_format = self.suggest_memory_format();
      }

      Tensor dst;
      if (self.qscheme() == kPerTensorAffine) {
        dst = _empty_affine_quantized(
            self.sizes(),
            self.options().memory_format(memory_format),
            self.q_scale(),
            self.q_zero_point(),
            nullopt);
      } else if (self.qscheme() == kPerChannelAffine) {
        dst = _empty_per_channel_affine_quantized(
            self.sizes(),
            self.q_per_channel_scales(),
            self.q_per_channel_zero_points(),
            self.q_per_channel_axis(),
            self.options().memory_format(memory_format),
            nullopt);
      } else {
        TORCH_CHECK(false, "clone for quantized Tensor only works for \
          PerTensorAffine and PerChannelAffine qscheme right now");
      }

      native::copy_(dst, self, false);

      return dst;
        */
}


pub fn equal_quantized_cpu(
        self_: &Tensor,
        other: &Tensor) -> bool {
    
    todo!();
        /*
            TORCH_CHECK(
          self.device().type() == kCPU && other.device().type() == kCPU,
          "quantized_equal is implemented only for the QuantizedCPU backend");
      if (!other.is_quantized()) {
        return false;
      }

      // Delegate to virtual equalTo method. This will ensure different concrete
      // Quantizers can have specific logic for comparison
      auto self_quantizer = get_qtensorimpl(self)->quantizer();
      auto other_quantizer = get_qtensorimpl(other)->quantizer();
      if (!self_quantizer->equalTo(other_quantizer)) {
        return false;
      }

      // Sizes and element types must be the same
      if (self.sizes() != other.sizes()) {
        return false;
      }
      if (self.element_size() != other.element_size()) {
        return false;
      }

      // Data must be the same
      auto self_contig = self.contiguous();
      auto other_contig = other.contiguous();

      void* self_data = self_contig.data_ptr();
      void* other_data = other_contig.data_ptr();
      return 0 == memcmp(self_data, other_data, self.numel() * self.element_size());
        */
}

/**
  | Calculate the quantization params
  | for the activation tensor
  |
  */
pub fn choose_qparams_per_tensor(
        self_:        &Tensor,
        reduce_range: bool) -> (f64,i64) {
    
    todo!();
        /*
            Tensor a;
      auto input_contig = self.contiguous();
      float x_min = input_contig.min().item<float>();
      float x_max = input_contig.max().item<float>();

      if (reduce_range && globalContext().qEngine() == QEngine::QNNPACK) {
        reduce_range = false;
      }

      auto q_params = quant_utils::ChooseQuantizationParams(
          /*min=*/x_min,
          /*max=*/x_max,
          /*qmin=*/0,
          /*qmax=*/255,
          /*preserve_sparsity=*/false,
          /*force_scale_power_of_two=*/false,
          /*reduce_range=*/reduce_range);

      return make_tuple(q_params.scale, q_params.zero_point);
        */
}


pub fn calculate_quant_loss(
        input:     *const f32,
        numel:     i32,
        xmin:      f32,
        xmax:      f32,
        q_input:   *mut f32,
        bit_width: i32) -> f32 {
    
    todo!();
        /*
            xmin = static_cast<Half>(xmin);
      float data_range = xmax - xmin;
      float qmax = (1 << bit_width) - 1;
      float scale = data_range == 0
          ? 1.0
          : static_cast<float>(static_cast<Half>(data_range / qmax));
      float inverse_scale = scale == 0 ? 1.0f : 1.0f / scale;

      float norm = 0.0f;
      int i = 0;

      // TODO add FBGEMM kernel
      // #ifdef USE_FBGEMM
      // #endif

      // remainder loop
      for (; i < numel; i++) {
        q_input[i] = max(
            0.0f, min<float>(nearbyint((input[i] - xmin) * inverse_scale), qmax));
        q_input[i] = q_input[i] * scale + xmin;
        norm += (input[i] - q_input[i]) * (input[i] - q_input[i]);
      }
      return sqrt(norm);
        */
}

/**
  | Helper function to find the best min/max
  | for a tensor to calculate qparams.
  | 
  | It uses a greedy approach to nudge the
  | min and max and calculate the l2 norm
  | and tries to minimize the quant error
  | by doing `torch.norm(x-fake_quant(x,s,z))`
  | 
  | Returns the optimized xmax and xmin
  | value of the tensor.
  |
  */

pub fn choose_qparams_optimized(
        input_tensor: &Tensor,
        numel:        i64,
        n_bins:       i64,
        ratio:        f64,
        bit_width:    i64) -> (Tensor,Tensor) {
    
    todo!();
        /*
            const float* input_row = input_tensor.data_ptr<float>();
      float xmin = *min_element(input_row, input_row + numel);
      float xmax = *max_element(input_row, input_row + numel);

      float stepsize = (xmax - xmin) / n_bins;
      int min_bins = n_bins * (1.0 - (float) ratio);
      Tensor input_tensor_contig = input_tensor.contiguous();
      const float* input = input_tensor_contig.data_ptr<float>();
      vector<float> q_input(numel);

      float loss =
          calculate_quant_loss(input, numel, xmin, xmax, q_input.data(), bit_width);
      float best_loss = loss;

      float cur_min = xmin;
      float cur_max = xmax;
      float cur_loss = loss;

      float thr = min_bins * stepsize;
      while (cur_min + thr < cur_max) {
        // move left
        float loss1 = calculate_quant_loss(
            input, numel, cur_min + stepsize, cur_max, q_input.data(), bit_width);
        // move right
        float loss2 = calculate_quant_loss(
            input, numel, cur_min, cur_max - stepsize, q_input.data(), bit_width);
        if (cur_loss < loss1 && cur_loss < loss2 && cur_loss < best_loss) {
          // found a local optima
          best_loss = cur_loss;
          xmin = cur_min;
          xmax = cur_max;
        }
        if (loss1 < loss2) {
          cur_min = cur_min + stepsize;
          cur_loss = loss1;
        } else {
          cur_max = cur_max - stepsize;
          cur_loss = loss2;
        }
      }

      Tensor xmax_tensor = empty({1});
      Tensor xmin_tensor = empty({1});
      xmax_tensor[0] = xmax;
      xmin_tensor[0] = xmin;
      return make_tuple(xmax_tensor, xmin_tensor);
        */
}
