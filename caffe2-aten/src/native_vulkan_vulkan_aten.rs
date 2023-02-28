crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/VulkanAten.h]

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/VulkanAten.cpp]

pub type VulkanTensorImpl = VulkanOpaqueTensorImpl<VulkanTensor>;

pub fn normalize_dim(d: i64, n: i64) -> i64 {
    
    todo!();
        /*
            return (d % n + n) % n;
        */
}

pub fn new_with_vtensor_vulkan(
        vt:      VulkanTensor,
        options: &TensorOptions) -> Tensor {
    
    todo!();
        /*
            auto sizes = vt.sizes();
      auto strides = vt.strides();
      return make_tensor<VulkanTensorImpl>(
          DispatchKeySet(DispatchKey::Vulkan),
          options.dtype(),
          Device(kVulkan),
          move(vt),
          vector<i64>(sizes.begin(), sizes.end()),
          vector<i64>(strides.begin(), strides.end()));
        */
}

pub fn vtensor_from_vulkan(tensor: &Tensor) -> &VulkanTensor {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT(
          tensor.is_vulkan(), "vtensor_from_vulkan expects Vulkan tensor input");
      VulkanTensorImpl* const impl =
          static_cast<VulkanTensorImpl*>(tensor.unsafeGetTensorImpl());
      return impl->unsafe_opaque_handle();
        */
}

pub fn vtensor_from_vulkan_mut(tensor: &mut Tensor) -> &mut VulkanTensor {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT(
          tensor.is_vulkan(), "vtensor_from_vulkan expects Vulkan tensor input");
      VulkanTensorImpl* const impl =
          static_cast<VulkanTensorImpl*>(tensor.unsafeGetTensorImpl());
      return impl->unsafe_opaque_handle();
        */
}

pub fn empty(
    size:          &[i32],
    dtype:         Option<ScalarType>,
    layout:        Option<Layout>,
    device:        Option<Device>,
    pin_memory:    Option<bool>,
    memory_format: Option<MemoryFormat>) -> Tensor {

    todo!();
        /*
            TORCH_CHECK(
          !pin_memory.has_value(),
          "'pin_memory' argument is incompatible with Vulkan tensor");
      TORCH_CHECK(
          !memory_format.has_value(),
          "'memory_format' argument is incompatible with Vulkan tensor");
      VulkanTensor vt{size.vec()};
      return new_with_vtensor_vulkan(
          move(vt), device(kVulkan).dtype(dtype));
        */
}

pub fn empty_strided(
    size:       &[i32],
    stride:     &[i32],
    dtype:      Option<ScalarType>,
    layout:     Option<Layout>,
    device:     Option<Device>,
    pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            return vulkan::empty(
          size, dtype, layout, device, pin_memory, nullopt);
        */
}

pub fn upsample_nearest2d(
    input:        &Tensor,
    output_sizes: &[i32],
    scales_h:     Option<f64>,
    scales_w:     Option<f64>) -> Tensor {

    todo!();
        /*
            const auto& x = vtensor_from_vulkan(input);
      const auto inputSizes = input.sizes();
      const auto in = inputSizes[0];
      const auto ic = inputSizes[1];
      const auto ih = inputSizes[2];
      const auto iw = inputSizes[3];

      const auto oh = outputSizes[0];
      const auto ow = outputSizes[1];
      const float height_scale = compute_scales_value<float>(scales_h, ih, oh);
      const float width_scale = compute_scales_value<float>(scales_w, iw, ow);
      VulkanTensor output{{in, ic, oh, ow}};
      vulkan::upsample_nearest2d(
          output, x, ih, iw, oh, ow, in, ic, height_scale, width_scale);
      return new_with_vtensor_vulkan(move(output), input.options());
        */
}

pub fn adaptive_avg_pool2d(
        input:       &Tensor,
        output_size: &[i32]) -> Tensor {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT(
          input.dim() == 4,
          "vulkan_adaptive_avg_pool2d expects 4-dimensional input");
      const auto& x = vtensor_from_vulkan(input);
      const auto inputSize = input.sizes();
      const auto in = inputSize[0];
      const auto ic = inputSize[1];
      const auto ih = inputSize[2];
      const auto iw = inputSize[3];

      const auto oh = outputSize[0];
      const auto ow = outputSize[1];
      VulkanTensor output{{in, ic, oh, ow}};
      vulkan::adaptive_avg_pool2d(output, x, ih, iw, oh, ow, in, ic);
      return new_with_vtensor_vulkan(move(output), input.options());
        */
}

pub fn avg_pool2d(
    self_:             &Tensor,
    kernel_size:       &[i32],
    stride:            &[i32],
    padding:           &[i32],
    ceil_mode:         bool,
    count_include_pad: bool,
    divisor_override:  Option<i64>) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(
          kernel_size.size() == 1 || kernel_size.size() == 2,
          "avg_pool2d: kernel_size must either be a single int, or a tuple of two ints");
      const int kH = safe_downcast<int>(kernel_size[0]);
      const int kW =
          kernel_size.size() == 1 ? kH : safe_downcast<int>(kernel_size[1]);

      TORCH_CHECK(
          stride.empty() || stride.size() == 1 || stride.size() == 2,
          "avg_pool2d: stride must either be omitted, a single int, or a tuple of two ints");
      const int dH = stride.empty() ? kH : safe_downcast<int>(stride[0]);
      const int dW = stride.empty()
          ? kW
          : stride.size() == 1 ? dH : safe_downcast<int>(stride[1]);

      TORCH_CHECK(
          padding.size() == 1 || padding.size() == 2,
          "avg_pool2d: padding must either be a single int, or a tuple of two ints");
      const int padH = safe_downcast<int>(padding[0]);
      const int padW = padding.size() == 1 ? padH : safe_downcast<int>(padding[1]);

      const auto& x = vtensor_from_vulkan(self);
      auto inputSize = self.sizes();
      const i64 iN = inputSize[0];
      const i64 iC = inputSize[1];
      const i64 iH = inputSize[2];
      const i64 iW = inputSize[3];

      const i64 oH =
          pooling_output_shape<i64>(iH, kH, padH, dH, 1, ceil_mode);
      const i64 oW =
          pooling_output_shape<i64>(iW, kW, padW, dW, 1, ceil_mode);

      pool2d_shape_check(
          self, kH, kW, dH, dW, padH, padW, 1, 1, iC, iH, iW, oH, oW, self.suggest_memory_format());

      VulkanTensor y{{iN, iC, oH, oW}};
      vulkan::avg_pool2d(
          y, x, iH, iW, oH, oW, iN, iC, kH, kW, dH, dW, padH, padW);
      return new_with_vtensor_vulkan(move(y), self.options());
        */
}

pub fn max_pool2d(
    self_:       &Tensor,
    kernel_size: &[i32],
    stride:      &[i32],
    padding:     &[i32],
    dilation:    &[i32],
    ceil_mode:   bool) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(
          kernel_size.size() == 1 || kernel_size.size() == 2,
          "Vulkan max_pool2d: kernel_size must either be a single int, or a tuple of two ints")
      const int kH = safe_downcast<int>(kernel_size[0]);
      const int kW =
          kernel_size.size() == 1 ? kH : safe_downcast<int>(kernel_size[1]);
      TORCH_CHECK(
          stride.size() == 0 || stride.size() == 1 || stride.size() == 2,
          "Vulkan max_pool2d: stride must either be omitted, a single int, or a tuple of two ints")
      const int dH = stride.empty() ? kH : safe_downcast<int>(stride[0]);
      const int dW = stride.empty()
          ? kW
          : stride.size() == 1 ? dH : safe_downcast<int>(stride[1]);

      TORCH_CHECK(
          padding.size() == 1 || padding.size() == 2,
          "Vulkan max_pool2d: padding must be either be a single int, or a tuple of two ints");
      const int padH = safe_downcast<int>(padding[0]);
      const int padW = padding.size() == 1 ? padH : safe_downcast<int>(padding[1]);

      TORCH_CHECK(
          dilation.size() == 1 || dilation.size() == 2,
          "Vulkan max_pool2d: dilation must be either a single int, or a tuple of two ints");
      const int dilationH = safe_downcast<int>(dilation[0]);
      const int dilationW =
          dilation.size() == 1 ? dilationH : safe_downcast<int>(dilation[1]);
      TORCH_CHECK(
          self.dim() == 4, "Vulkan max_pool2d is implemented for 4-dim input");

      const auto& x = vtensor_from_vulkan(self);
      const auto inputSize = self.sizes();
      const i64 iN = inputSize[0];
      const i64 iC = inputSize[1];
      const i64 iH = inputSize[2];
      const i64 iW = inputSize[3];

      const i64 oH =
          pooling_output_shape<i64>(iH, kH, padH, dH, dilationH, ceil_mode);
      const i64 oW =
          pooling_output_shape<i64>(iW, kW, padW, dW, dilationW, ceil_mode);

      pool2d_shape_check(
          self,
          kH,
          kW,
          dH,
          dW,
          padH,
          padW,
          dilationH,
          dilationW,
          iC,
          iH,
          iW,
          oH,
          oW,
          self.suggest_memory_format());

      VulkanTensor y{{iN, iC, oH, oW}};
      vulkan::max_pool2d(
          y,
          x,
          iH,
          iW,
          oH,
          oW,
          iN,
          iC,
          kH,
          kW,
          dH,
          dW,
          padH,
          padW,
          dilationH,
          dilationW);
      return new_with_vtensor_vulkan(move(y), self.options());
        */
}


pub fn reshape(
        input: &Tensor,
        shape: &[i32]) -> Tensor {
    
    todo!();
        /*
            return new_with_vtensor_vulkan(
          vulkan::reshape_copy(vtensor_from_vulkan(input), shape.vec()),
          input.options());
        */
}


pub fn cat(
        tensors: TensorList,
        dim:     i64) -> Tensor {
    
    todo!();
        /*
            const auto norm_dim = normalize_dim(dim, 4);
      TORCH_INTERNAL_ASSERT(
          norm_dim == 0 || norm_dim == 1,
          "Vulkan cat is implemented only for batch and channels dimensions");
      Tensor tensor = tensors[0];
      i64 cat_dim_size = 0;

      vector<VulkanTensor> vTensors{};
      for (int i = 0; i < tensors.size(); ++i) {
        const auto& t = tensors[i];
        TORCH_INTERNAL_ASSERT(
            t.dim() == 4, "Vulkan cat expects 4 dimensional inputs");
        TORCH_INTERNAL_ASSERT(t.is_vulkan(), "Vulkan cat expects Vulkan inputs");

        for (int d = 0; d < 4; ++d) {
          if (d == dim) {
            continue;
          }
          TORCH_INTERNAL_ASSERT(
              t.size(d) == tensor.size(d),
              "Vulkan cat inputs must have matching sizes except concatenated dimension");
        }
        vTensors.push_back(vtensor_from_vulkan(t));
        cat_dim_size += t.size(dim);
      }

      auto result_size = tensor.sizes().vec();
      result_size[dim] = cat_dim_size;

      VulkanTensor output{result_size};

      vulkan::cat(output, vTensors, dim);
      return new_with_vtensor_vulkan(move(output), tensor.options());
        */
}

pub fn transpose(
    self_: &Tensor,
    dim0:  i64,
    dim1:  i64) -> Tensor {
    
    todo!();
        /*
            return new_with_vtensor_vulkan(
          vulkan::transpose(vtensor_from_vulkan(self), dim0, dim1),
          self.options());
        */
}

pub fn transpose_mut(
    self_: &mut Tensor,
    dim0:  i64,
    dim1:  i64) -> &mut Tensor {
    
    todo!();
        /*
            auto& x = vtensor_from_vulkan(self);
      x = vulkan::transpose(x, dim0, dim1);
      return self;
        */
}

pub fn view(
    self_: &Tensor,
    size:  &[i32]) -> Tensor {
    
    todo!();
        /*
            return new_with_vtensor_vulkan(
          vulkan::reshape_copy(
              vtensor_from_vulkan(self), infer_size(size, self.numel())),
          self.options());
        */
}

pub fn contiguous(
    self_:         &Tensor,
    memory_format: MemoryFormat) -> Tensor {
    
    todo!();
        /*
            return self;
        */
}

pub fn slice(
    self_: &Tensor,
    dim:   i64,
    start: i64,
    end:   i64,
    step:  i64) -> Tensor {

    todo!();
        /*
            return new_with_vtensor_vulkan(
          vulkan::slice(vtensor_from_vulkan(self), dim, start, end, step),
          self.options());
        */
}

pub fn add(
    self_: &Tensor,
    other: &Tensor,
    alpha: &Scalar) -> Tensor {
    
    todo!();
        /*
            auto xt = self.is_vulkan() ? self : self.vulkan();
      const auto& x = vtensor_from_vulkan(xt);
      auto yt = other.is_vulkan() ? other : other.vulkan();
      const auto& y = vtensor_from_vulkan(yt);
      const float a = alpha.to<float>();

      VulkanTensor output{self.sizes().vec()};
      vulkan::add(output, x, y, a);
      return new_with_vtensor_vulkan(move(output), self.options());
        */
}

pub fn vtensor_mut(t: &mut Tensor) -> &mut VulkanTensor {
    
    todo!();
        /*
            if (t.is_vulkan()) {
        return vtensor_from_vulkan(t);
      }
      auto tv = t.vulkan();
      return vtensor_from_vulkan(tv);
        */
}

pub fn vtensor(t: &Tensor) -> &VulkanTensor {
    
    todo!();
        /*
            if (t.is_vulkan()) {
        return vtensor_from_vulkan(t);
      }
      const auto tv = t.vulkan();
      return vtensor_from_vulkan(tv);
        */
}

pub fn add_mut(
    self_: &mut Tensor,
    other: &Tensor,
    alpha: &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            auto& x = vtensor(self);
      const auto& y = vtensor(other);
      float a = alpha.to<float>();

      VulkanTensor output{self.sizes().vec()};
      vulkan::add(output, x, y, a);
      x = move(output);
      return self;
        */
}

pub fn add_scalar(
    self_: &Tensor,
    other: &Scalar,
    alpha: &Scalar) -> Tensor {
    
    todo!();
        /*
            const auto& x = vtensor_from_vulkan(self);
      const float s = other.to<float>();
      const float a = alpha.to<float>();
      VulkanTensor output{self.sizes().vec()};
      vulkan::add(output, x, s * a);
      return new_with_vtensor_vulkan(move(output), self.options());
        */
}


pub fn mul_scalar(
        self_: &Tensor,
        other: &Scalar) -> Tensor {
    
    todo!();
        /*
            const auto& x = vtensor_from_vulkan(self);
      const float s = other.to<float>();
      VulkanTensor output{self.sizes().vec()};
      vulkan::mul(output, x, s);
      return new_with_vtensor_vulkan(move(output), self.options());
        */
}


pub fn select(
        self_: &Tensor,
        dim:   i64,
        index: i64) -> Tensor {
    
    todo!();
        /*
            auto sliced = vulkan::slice(self, dim, index, index + 1, 1);
      auto sizes = self.sizes().vec();
      sizes.erase(sizes.begin() + dim);
      return vulkan::reshape(sliced, sizes);
        */
}


pub fn unsqueeze(
        self_: &Tensor,
        dim:   i64) -> Tensor {
    
    todo!();
        /*
            auto sizes = self.sizes().vec();
      sizes.insert(sizes.begin() + dim, 1);
      return vulkan::reshape(self, sizes);
        */
}



pub fn convolution(
        input:          &Tensor,
        weight:         &Tensor,
        bias:           &Option<Tensor>,
        stride:         &[i32],
        padding:        &[i32],
        dilation:       &[i32],
        transposed:     bool,
        output_padding: &[i32],
        groups:         i64) -> Tensor {
    
    todo!();
        /*
            const vulkan::Conv2DParams params{
          input.sizes(), weight.sizes(), padding, stride, dilation, groups};
      TORCH_INTERNAL_ASSERT(
          input.dim() == 4, "convolution: Expected 4-dimensional input");
      TORCH_INTERNAL_ASSERT(
          weight.dim() == 4, "convolution: Expected 4-dimensional weight");
      TORCH_INTERNAL_ASSERT(
          groups == 1 || groups == params.C,
          "convolution: only nogroup or depthwise convolutions supported");
      TORCH_INTERNAL_ASSERT(!transposed, "convolution: transposed not supported");

      const VulkanTensor& vinput = vtensor_from_vulkan(input);
      VulkanTensor voutput = VulkanTensor{params.output_sizes()};

      vulkan::conv2d(
          voutput,
          vinput,
          weight.data_ptr<float>(),
          (bias.has_value() && bias->defined())
              ? make_optional<const float*>(bias->data_ptr<float>())
              : nullopt,
          params);
      return new_with_vtensor_vulkan(move(voutput), input.options());
        */
}


pub fn addmm(
        self_: &Tensor,
        mat1:  &Tensor,
        mat2:  &Tensor,
        beta:  &Scalar,
        alpha: &Scalar) -> Tensor {
    
    todo!();
        /*
            const VulkanTensor t =
          vtensor_from_vulkan(self.is_vulkan() ? self : self.vulkan());
      const VulkanTensor m1 =
          vtensor_from_vulkan(mat1.is_vulkan() ? mat1 : mat1.vulkan());
      const VulkanTensor m2 =
          vtensor_from_vulkan(mat2.is_vulkan() ? mat2 : mat2.vulkan());
      const float b = beta.to<float>();
      const float a = alpha.to<float>();

      VulkanTensor output = VulkanTensor{self.sizes().vec()};
      vulkan::addmm(output, t, m1, m2, b, a);
      return new_with_vtensor_vulkan(move(output), self.options());
        */
}


pub fn mm(
        self_: &Tensor,
        mat2:  &Tensor) -> Tensor {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT(
          self.dim() == 2 && mat2.dim() == 2,
          "vulkan_mm expects 2-dimensional tensors");
      const auto m1Sizes = self.sizes();
      const auto m2Sizes = mat2.sizes();
      TORCH_INTERNAL_ASSERT(
          m1Sizes[1] == m2Sizes[0],
          "vulkan_mm expects self.sizes[1] equal mat2.sizes[0]");

      const auto& m1 = vtensor_from_vulkan(self.is_vulkan() ? self : self.vulkan());
      const auto& m2 = vtensor_from_vulkan(mat2.is_vulkan() ? mat2 : mat2.vulkan());

      VulkanTensor output{{m1Sizes[0], m2Sizes[1]}};
      vulkan::addmm(output, nullopt, m1, m2, 0.f, 1.f);
      return new_with_vtensor_vulkan(move(output), self.options());
        */
}


pub fn clamp(
        self_: &Tensor,
        min:   &Option<Scalar>,
        max:   &Option<Scalar>) -> Tensor {
    
    todo!();
        /*
            const auto& x = vtensor_from_vulkan(self);
      VulkanTensor output{self.sizes().vec()};
      vulkan::clamp(
          output,
          x,
          min ? min.value().to<float>() : -numeric_limits<float>::infinity(),
          max ? max.value().to<float>() : numeric_limits<float>::infinity());
      return vulkan::new_with_vtensor_vulkan(
          move(output), self.options());
        */
}

pub fn clamp_mut(
        self_: &mut Tensor,
        min:   &Option<Scalar>,
        max:   &Option<Scalar>) -> &mut Tensor {
    
    todo!();
        /*
            auto& x = vtensor_from_vulkan(self);
      VulkanTensor output{self.sizes().vec()};
      vulkan::clamp(
          output,
          x,
          min ? min.value().to<float>() : -numeric_limits<float>::infinity(),
          max ? max.value().to<float>() : numeric_limits<float>::infinity());
      x = move(output);
      return self;
        */
}

pub fn hardtanh(
        self_: &Tensor,
        min:   &Scalar,
        max:   &Scalar) -> Tensor {
    
    todo!();
        /*
            return vulkan::clamp(self, min, max);
        */
}

pub fn hardtanh_mut(
        self_: &mut Tensor,
        min:   &Scalar,
        max:   &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            return vulkan::clamp_(self, min, max);
        */
}

pub fn relu(self_: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return vulkan::clamp_(self, 0, nullopt);
        */
}


pub fn mean(
        self_:   &Tensor,
        dim:     &[i32],
        keepdim: bool,
        dtype:   Option<ScalarType>) -> Tensor {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT(!keepdim, "keepdim not implemented for Vulkan mean");
      TORCH_INTERNAL_ASSERT(self.is_vulkan(), "mean expects Vulkan tensor input");

      // Mean is implemented only for HW dimensions of 4-d tensor
      TORCH_INTERNAL_ASSERT(self.dim() == 4);
      static const unordered_set<i64> expected_dims_set({2, 3});
      unordered_set<i64> dims_set;
      for (const auto& d : dim) {
        dims_set.insert(normalize_dim(d, 4));
      }
      TORCH_INTERNAL_ASSERT(expected_dims_set == dims_set);

      const auto& x = vtensor_from_vulkan(self);
      const auto sizes = self.sizes();
      VulkanTensor output{vector<i64>{sizes[0], sizes[1]}};
      vulkan::mean(output, x);
      return new_with_vtensor_vulkan(move(output), self.options());
        */
}

#[cfg(not(USE_VULKAN_API))]
lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
      m.impl("slice.Tensor", TORCH_FN(native::vulkan::slice));
      m.impl("view", TORCH_FN(native::vulkan::reshape));
      m.impl("select.int", TORCH_FN(native::vulkan::select));
      m.impl("transpose.int", TORCH_FN(native::vulkan::transpose));
      m.impl("transpose_", native::vulkan::transpose_);
      m.impl("view", TORCH_FN(native::vulkan::view));
      m.impl("unsqueeze", TORCH_FN(native::vulkan::unsqueeze));
      m.impl("empty.memory_format", native::vulkan::empty);
      m.impl("empty_strided", TORCH_FN(native::vulkan::empty_strided));
      m.impl("add.Tensor", TORCH_FN(native::vulkan::add));
      m.impl("clamp", TORCH_FN(native::vulkan::clamp));
      m.impl("mean.dim", TORCH_FN(native::vulkan::mean));
      m.impl("mm", TORCH_FN(native::vulkan::mm));
      m.impl("addmm", TORCH_FN(native::vulkan::addmm));
      m.impl(
          "upsample_nearest2d",
          TORCH_FN(native::vulkan::upsample_nearest2d));
      m.impl(
          "_adaptive_avg_pool2d",
          TORCH_FN(native::vulkan::adaptive_avg_pool2d));
      m.impl("avg_pool2d", TORCH_FN(native::vulkan::avg_pool2d));
      m.impl("max_pool2d", TORCH_FN(native::vulkan::max_pool2d));
      m.impl("_cat", TORCH_FN(native::vulkan::cat));
      m.impl("mul.Scalar", TORCH_FN(native::vulkan::mul_scalar));
      m.impl("add.Scalar", TORCH_FN(native::vulkan::add_scalar));
      m.impl(
          "convolution_overrideable", native::vulkan::convolution);
      m.impl("hardtanh_", native::vulkan::hardtanh_);
      m.impl("relu_", native::vulkan::relu_);
      m.impl("add_.Tensor", native::vulkan::add_);
    }
    */
}


pub fn copy_from_vulkan(
        self_: &mut Tensor,
        src:   &Tensor) -> &mut Tensor {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT(
          src.device().type() == DeviceType_Vulkan,
          "copy_from_vulkan input tensor's device is not Vulkan");
      TORCH_INTERNAL_ASSERT(
          self.device().is_cpu(),
          "copy_from_vulkan is implemented only for CPU device output");
      TORCH_INTERNAL_ASSERT(
          self.layout() == Layout::Strided,
          "copy_from_vulkan is implemented only for Strided layout output");
      TORCH_INTERNAL_ASSERT(
          self.scalar_type() == ScalarType::Float,
          "copy_from_vulkan is implemented only for float dtype output, got:",
          self.scalar_type());
      TORCH_INTERNAL_ASSERT(
          self.is_contiguous(),
          "copy_from_vulkan is implemented only for contiguous output tensor");

      const auto& vtensor = vtensor_from_vulkan(src);
      vtensor.copy_data_to_host(self.data_ptr<float>());
      return self;
        */
}


pub fn copy_to_vulkan(
        self_: &mut Tensor,
        src:   &Tensor) -> &mut Tensor {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT(
          self.device().type() == DeviceType_Vulkan,
          "copy_to_vulkan output tensor's device is not Vulkan");
      TORCH_INTERNAL_ASSERT(
          src.device().is_cpu(),
          "copy_to_vulkan is implemented only for CPU device input");
      TORCH_INTERNAL_ASSERT(
          src.layout() == Layout::Strided,
          "copy_to_vulkan is implemented only for Strided layout input");
      TORCH_INTERNAL_ASSERT(
          src.scalar_type() == ScalarType::Float,
          "copy_to_vulkan is implemented only for float dtype");

      auto cpu_tensor_contiguous = src.contiguous();
      VulkanTensor& vtensor = vtensor_from_vulkan(self);
      vtensor.set_data_from_host(cpu_tensor_contiguous.data_ptr<float>());
      return self;
        */
}

pub fn vulkan_copy_impl(
        self_: &mut Tensor,
        src:   &Tensor) -> &mut Tensor {
    
    todo!();
        /*
            if (src.device().type() == kVulkan && self.device().type() == kCPU) {
        return copy_from_vulkan_(self, src);
      }
      if (src.device().type() == kCPU && self.device().type() == kVulkan) {
        return copy_to_vulkan_(self, src);
      }
      TORCH_INTERNAL_ASSERT(
          src.device().type() == DeviceType_Vulkan,
          "vulkan_copy_ is implemented only for CPU,Strided,float->Vulkan; Vulkan->CPU,Strided,float");
      return self;
        */
}

pub struct VulkanImpl {
    base: VulkanImplInterface,
}

impl VulkanImpl {
    
    pub fn is_vulkan_available(&self) -> bool {
        
        todo!();
        /*
            return native::vulkan::is_available();
        */
    }
    
    pub fn vulkan_copy(&self, 
        self_: &mut Tensor,
        src:   &Tensor) -> &mut Tensor {
        
        todo!();
        /*
            return vulkan_copy_impl_(self, src);
        */
    }
}

lazy_static!{
    /*
    static VulkanImplRegistrar g_vulkan_impl(new VulkanImpl());
    */
}

pub fn convolution_prepack_weights(weight: &Tensor) -> Tensor {
    
    todo!();
        /*
            const auto wsizes = weight.sizes();
      TORCH_INTERNAL_ASSERT(
          wsizes.size() == 4,
          "convolution_prepack_weights: Expected 4-dimensional weight");

      const i64 OC = wsizes[0];
      const i64 C = wsizes[1];
      const i64 KH = wsizes[2];
      const i64 KW = wsizes[3];
      VulkanTensor voutput =
          VulkanTensor{{UP_DIV(OC, 4), UP_DIV(C, 4), KH * KW, 16}};

      vulkan::conv2d_prepack_weights(
          voutput, weight.data_ptr<float>(), OC, C, KH, KW);
      return new_with_vtensor_vulkan(
          move(voutput), device(kVulkan).dtype(kFloat));
        */
}

pub fn convolution_prepacked(
    input:                   &Tensor,
    weight_sizes:            &[i32],
    weight_prepacked_vulkan: &Tensor,
    bias:                    &Option<Tensor>,
    padding:                 &[i32],
    stride:                  &[i32],
    dilation:                &[i32],
    groups:                  i64,
    output_min:              f32,
    output_max:              f32) -> Tensor {

    todo!();
        /*
            TORCH_INTERNAL_ASSERT(
          input.dim() == 4, "Vulkan convolution: Expected 4-dimensional input");
      TORCH_INTERNAL_ASSERT(
          weight_prepacked_vulkan.dim() == 4,
          "Vulkan convolution: Expected 4-dimensional weight");
      vulkan::Conv2DParams params{
          input.sizes(), weightSizes, padding, stride, dilation, groups};
      TORCH_INTERNAL_ASSERT(
          groups == 1 || groups == params.C,
          "Vulkan convolution: only nogroup or depthwise convolutions supported");
      const VulkanTensor& vinput = vtensor_from_vulkan(input);
      const VulkanTensor& vweight =
          vtensor_from_vulkan(weight_prepacked_vulkan);
      VulkanTensor voutput =
          VulkanTensor{{params.N, params.OC, params.OH, params.OW}};
      const bool hasBias = bias.has_value() && bias->defined();
      if (hasBias && bias->is_vulkan()) {
        const VulkanTensor& vbias = vtensor_from_vulkan(*bias);
        vulkan::conv2d(
            voutput, vinput, vweight, vbias, params, output_min, output_max);
      } else {
        vulkan::conv2d(
            voutput,
            vinput,
            vweight,
            hasBias ? make_optional<const float*>((*bias).data_ptr<float>())
                    : nullopt,
            params,
            output_min,
            output_max);
      }
      return new_with_vtensor_vulkan(move(voutput), input.options());
        */
}
