
#[cfg(feature = "fbgemm")]
pub struct PackedConvWeight<const SPATIAL_DIM: usize = 2> {
    base:           ConvPackedParamsBase<SpatialDim>,
    w:              Box<FbgemmPackWeightsForConv<SpatialDim>>,
    bias:           Option<Tensor>,
    stride:         TorchList<i64>,
    padding:        TorchList<i64>,
    output_padding: TorchList<i64>,
    dilation:       TorchList<i64>,
    groups:         i64,
    transpose:      u8,
    col_offsets:    Vec<i32>,
    kernel:         Vec<i64>,
    w_scale:        Vec<f32>,
    w_zp:           Vec<i32>,
    q_scheme:       QScheme,
}

#[cfg(feature = "fbgemm")]
impl PackedConvWeight {
    
    pub fn new(
        w:              Box<FbgemmPackWeightsForConv<SpatialDim>>,
        bias:           Option<Tensor>,
        stride:         TorchList<i64>,
        padding:        TorchList<i64>,
        output_padding: TorchList<i64>,
        dilation:       TorchList<i64>,
        groups:         i64,
        transpose:      u8,
        col_offsets:    Vec<i32>,
        kernel:         Vec<i64>,
        w_scale:        Vec<f32>,
        w_zp:           Vec<i32>,
        q_scheme:       QScheme) -> Self {
    
        todo!();
        /*


            : w(move(w)),
        bias(move(bias)),
        stride_(move(stride)),
        padding_(move(padding)),
        output_padding_(move(output_padding)),
        dilation_(move(dilation)),
        groups_(groups),
        transpose_(transpose),
        col_offsets(move(col_offsets)),
        kernel(move(kernel)),
        w_scale(move(w_scale)),
        w_zp(move(w_zp)),
        q_scheme(q_scheme)
        */
    }
    
    pub fn apply(&mut self, 
        input:             &Tensor,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor {
        
        todo!();
        /*
        
        */
    }
    
    pub fn apply_relu(&mut self, 
        input:             &Tensor,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor {
        
        todo!();
        /*
        
        */
    }
    
    pub fn unpack(&mut self) -> (Tensor,Option<Tensor>) {
        
        todo!();
        /*
        
        */
    }
    
    pub fn prepack(
        weight:         Tensor,
        bias:           Option<Tensor>,
        stride:         TorchList<i64>,
        padding:        TorchList<i64>,
        output_padding: TorchList<i64>,
        dilation:       TorchList<i64>,
        groups:         i64,
        transpose:      bool) -> IntrusivePtr<ConvPackedParamsBase<SpatialDim>> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn get_bias_data(&mut self, bias: *mut Tensor) -> *const f32 {
        
        todo!();
        /*
        
        */
    }
    
    pub fn get_quantization_params(&mut self, 
        act_scale:               f32,
        out_scale:               f32,
        output_multiplier_float: *mut Vec<f32>,
        act_times_w_scale:       *mut Vec<f32>)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn stride(&self) -> TorchList<i64> {
        
        todo!();
        /*
            return stride_;
        */
    }
    
    pub fn padding(&self) -> TorchList<i64> {
        
        todo!();
        /*
            return padding_;
        */
    }
    
    pub fn output_padding(&self) -> TorchList<i64> {
        
        todo!();
        /*
            return output_padding_;
        */
    }
    
    pub fn dilation(&self) -> TorchList<i64> {
        
        todo!();
        /*
            return dilation_;
        */
    }
    
    pub fn groups(&self) -> i64 {
        
        todo!();
        /*
            return groups_;
        */
    }
    
    pub fn transpose(&self) -> bool {
        
        todo!();
        /*
            return (bool)transpose_;
        */
    }
    
    pub fn apply_impl<const ReluFused: bool>(&mut self, 
        input:             &Tensor,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor {
    
        todo!();
        /*
        
        */
    }
}

/// PackWeight: Convert the weight from uint8 to int8.
#[cfg(feature = "fbgemm")]
#[inline] pub fn convert_uint8_int8(
        len:       i32,
        src_uint8: *const u8,
        dst_int8:  *mut i8)  {
    
    todo!();
        /*
            for (int i = 0; i < len; ++i) {
        dst_int8[i] = static_cast<i8>(static_cast<i32>(src_uint8[i]) - 128);
      }
        */
}

/// UnpackWeight: Convert the weight from int8 to
/// uint8.
///
#[cfg(feature = "fbgemm")]
#[inline] pub fn convert_int8_uint8(
        len:       i32,
        src_int8:  *const i8,
        dst_uint8: *mut u8)  {
    
    todo!();
        /*
            for (int i = 0; i < len; ++i) {
        dst_uint8[i] =
            static_cast<u8>(static_cast<i32>(src_int8[i]) + 128);
      }
        */
}

#[cfg(feature = "fbgemm")]
pub fn make_fbgemm_conv_param<const SPATIAL_DIM: usize = 2>(
        N:              i32,
        C:              i32,
        M:              i32,
        image_shape:    &Vec<i32>,
        groups:         i32,
        kernels:        &Vec<i32>,
        strides:        &Vec<i32>,
        pads:           &Vec<i32>,
        dilations:      &Vec<i32>,
        output_padding: &Vec<i32>,
        transposed:     bool) -> FbgemmConvParam<SpatialDim> {
    let transposed: bool = transposed.unwrap_or(false);
    let output_padding = Vec::with_capacity(SPATIAL_DIM);

    todo!();
        /*
        
        */
}

/**
  | TODO: Remove functions below when
  | ChannelsLast3d is ready.
  |
  */
#[cfg(feature = "fbgemm")]
pub fn make_strided_qtensor_cpu(
        sizes:     &&[i32],
        strides:   &&[i32],
        options:   &TensorOptions,
        quantizer: QuantizerPtr) -> Tensor {
    
    todo!();
        /*
        
        */
}

#[cfg(feature = "fbgemm")]
pub fn make_empty_affine_quantized_channels_last3d_tensor(
        N:          i64,
        C:          i64,
        D:          i64,
        H:          i64,
        W:          i64,
        options:    &TensorOptions,
        scale:      f64,
        zero_point: i64) -> Tensor {
    
    todo!();
        /*
        
        */
}

#[cfg(feature = "fbgemm")]
pub fn make_empty_per_channel_affine_quantized_channels_last3d_tensor(
        N:           i64,
        C:           i64,
        D:           i64,
        H:           i64,
        W:           i64,
        options:     &TensorOptions,
        scales:      &Tensor,
        zero_points: &Tensor) -> Tensor {
    
    todo!();
        /*
        
        */
}

#[cfg(feature = "fbgemm")]
pub fn convert_to_channels_last3d_tensor(src: &Tensor) -> Tensor {
    
    todo!();
        /*
        
        */
}

#[cfg(feature = "fbgemm")]
pub fn transpose_conv_tensor_unpack_conversion<const SPATIAL_DIM: usize = 2>(
        src:    &Tensor,
        groups: i32) -> Tensor {
    
    todo!();
        /*
        
        */
}

#[cfg(feature = "fbgemm")]
pub fn convert_conv_weights_to_channel_last_tensor<const SPATIAL_DIM: usize>(
        src:       &Tensor,
        groups:    i32,
        transpose: bool) -> Tensor {
    
    todo!();
        /*
        
        */
}

pub struct PackedEmbeddingBagWeight {
    base:     EmbeddingPackedParamsBase,
    packed_w: Tensor,
    w_scale:  Vec<f32>,
    w_zp:     Vec<f32>,
    bit_rate: i64,
    q_scheme: QScheme,
    version:  i64,
}

impl PackedEmbeddingBagWeight {
    
    pub fn new(
        packed_w: Tensor,
        w_scale:  Vec<f32>,
        w_zp:     Vec<f32>,
        bit_rate: i64,
        q_scheme: QScheme,
        version:  i64) -> Self {
    
        todo!();
        /*


            : packed_w(move(packed_w)),
            w_scale(move(w_scale)),
            w_zp(move(w_zp)),
            bit_rate_(bit_rate),
            q_scheme(q_scheme),
            version_(version) 
        if (!packed_w.is_contiguous()) {
          packed_w = packed_w.contiguous();
        }
        */
    }
    
    pub fn unpack(&mut self) -> Tensor {
        
        todo!();
        /*
        
        */
    }
    
    pub fn prepack(weight: Tensor) -> IntrusivePtr<EmbeddingPackedParamsBase> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn bit_rate(&self) -> i64 {
        
        todo!();
        /*
            return bit_rate_;
        */
    }
    
    pub fn version(&self) -> i64 {
        
        todo!();
        /*
            return version_;
        */
    }
    
    pub fn embeddingbag_byte(&mut self, 
        indices:                    &Tensor,
        offsets:                    &Option<Tensor>,
        pruned_weights:             bool,
        per_sample_weights:         &Option<Tensor>,
        compressed_indices_mapping: &Option<Tensor>,
        include_last_offset:        bool,
        is_embedding_op:            bool) -> Tensor {
        
        todo!();
        /*
        
        */
    }
    
    pub fn embeddingbag_4bit(&mut self, 
        indices:                    &Tensor,
        offsets:                    &Option<Tensor>,
        pruned_weights:             bool,
        per_sample_weights:         &Option<Tensor>,
        compressed_indices_mapping: &Option<Tensor>,
        include_last_offset:        bool) -> Tensor {
        
        todo!();
        /*
        
        */
    }
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/fbgemm_utils.cpp]

#[cfg(feature = "fbgemm")]
pub fn is_channels_last3d(tensor: &Tensor) -> bool {
    
    todo!();
        /*
            if (tensor.dim() != 5) {
        return false;
      }
      const i64 C = tensor.size(1);
      const i64 D = tensor.size(2);
      const i64 H = tensor.size(3);
      const i64 W = tensor.size(4);
      return tensor.stride(0) == D * H * W * C && tensor.stride(1) == 1 &&
          tensor.stride(2) == H * W * C && tensor.stride(3) == W * C &&
          tensor.stride(4) == C;
        */
}

#[cfg(feature = "fbgemm")]
pub fn copy_to_channels_last3d_tensor<T>(
        N:   i64,
        C:   i64,
        D:   i64,
        H:   i64,
        W:   i64,
        src: *const T,
        dst: *mut T)  {

    todo!();
        /*
            const i64 inner_size = D * H * W;
      for (i64 i = 0; i < N; ++i) {
        for (i64 j = 0; j < inner_size; ++j) {
          for (i64 k = 0; k < C; ++k) {
            dst[(i * inner_size + j) * C + k] = src[(i * C + k) * inner_size + j];
          }
        }
      }
        */
}

#[cfg(feature = "fbgemm")]
pub fn copy_icf_irst3d_tensor_to_channels_last3d_tensor<T>(
        G:    i64,
        IC_G: i64,
        OC_G: i64,
        D:    i64,
        H:    i64,
        W:    i64,
        src:  *const T,
        dst:  *mut T)  {

    todo!();
        /*
            // IC OC/G THW -> G OC/G THW IC/G
      const i64 inner_size = D * H * W;
      for (i64 i = 0; i < G * OC_G; ++i) {
        for (i64 j = 0; j < inner_size; ++j) {
          for (i64 ic = 0; ic < IC_G; ++ic) {
            int g = i / OC_G;
            int oc = i % OC_G;
            dst[(i * inner_size + j) * IC_G + ic] =
                src[((g * IC_G + ic) * OC_G + oc) * inner_size + j];
          }
        }
      }
        */
}

#[cfg(feature = "fbgemm")]
pub fn make_fbgemm_conv_param<const kSpatialDim: i32>(
        N:              i32,
        C:              i32,
        M:              i32,
        image_shape:    &Vec<i32>,
        groups:         i32,
        kernels:        &Vec<i32>,
        strides:        &Vec<i32>,
        pads:           &Vec<i32>,
        dilations:      &Vec<i32>,
        output_padding: &Vec<i32>,
        transposed:     bool) -> FbgemmConvParam<SpatialDim> {

    todo!();
        /*
      array<int, kSpatialDim> image_shape_;
      array<int, kSpatialDim> kernels_;
      array<int, kSpatialDim> strides_;
      array<int, kSpatialDim * 2> pads_;
      array<int, kSpatialDim> dilations_;
      array<int, kSpatialDim> output_padding_;
      move(image_shape.begin(), image_shape.begin() + image_shape.size(), image_shape_.begin());
      move(
          kernels.begin(), kernels.begin() + kernels.size(), kernels_.begin());
      move(
          strides.begin(), strides.begin() + strides.size(), strides_.begin());
      move(
          dilations.begin(),
          dilations.begin() + dilations.size(),
          dilations_.begin());
      move(
          output_padding.begin(),
          output_padding.begin() + output_padding.size(),
          output_padding_.begin());
      copy(pads.begin(), pads.begin() + pads.size(), pads_.begin());
      move(pads.begin(), pads.begin() + pads.size(), pads_.begin() + pads.size());

      return fbgemm::conv_param_t<kSpatialDim>(
          N, // batch size
          C, // input channels
          M, // output channels
          image_shape_, // feature map size
          groups, // groups
          kernels_, // kernels
          strides_, // strides
          pads_, // paddings
          dilations_, // dilations
          output_padding_, // output paddings for conv transpose
          transposed);
        */
}

#[cfg(feature = "fbgemm")]
pub fn make_strided_qtensor_cpu(
        sizes:     &&[i32],
        strides:   &&[i32],
        options:   &TensorOptions,
        quantizer: QuantizerPtr) -> Tensor {
    
    todo!();
        /*
            AT_ASSERT(options.device().is_cpu());
      native::check_size_nonnegative(sizes);
      auto* allocator = getCPUAllocator();
      const i64 nelements = multiply_integers(sizes);
      auto dtype = options.dtype();
      TORCH_CHECK(
          isQIntType(typeMetaToScalarType(dtype)),
          "ScalarType is not supported in new_qtensor_cpu.");
      i64 size_bytes = nelements * dtype.itemsize();
      auto storage = make_intrusive<StorageImpl>(
          StorageImpl::use_byte_size_t(),
          size_bytes,
          allocator->allocate(size_bytes),
          allocator,
          /* resizable = */ true);
      auto tensor = make_tensor<QTensorImpl>(
          storage,
          DispatchKeySet(DispatchKey::QuantizedCPU),
          dtype,
          quantizer);
      get_qtensorimpl(tensor)->set_sizes_and_strides(sizes, strides);
      return tensor;
        */
}

#[cfg(feature = "fbgemm")]
pub fn make_empty_affine_quantized_channels_last3d_tensor(
        N:          i64,
        C:          i64,
        D:          i64,
        H:          i64,
        W:          i64,
        options:    &TensorOptions,
        scale:      f64,
        zero_point: i64) -> Tensor {
    
    todo!();
        /*
            return MakeStridedQTensorCPU(
          {N, C, D, H, W},
          {D * H * W * C, 1, H * W * C, W * C, C},
          options,
          make_per_tensor_affine_quantizer(
              scale, zero_point, typeMetaToScalarType(options.dtype())));
        */
}

#[cfg(feature = "fbgemm")]
pub fn make_empty_per_channel_affine_quantized_channels_last3d_tensor(
        N:           i64,
        C:           i64,
        D:           i64,
        H:           i64,
        W:           i64,
        options:     &TensorOptions,
        scales:      &Tensor,
        zero_points: &Tensor) -> Tensor {
    
    todo!();
        /*
            return MakeStridedQTensorCPU(
          {N, C, D, H, W},
          {D * H * W * C, 1, H * W * C, W * C, C},
          options,
          make_per_channel_affine_quantizer(
              scales,
              zero_points,
              0, // axis
              typeMetaToScalarType(options.dtype())));
        */
}

#[cfg(feature = "fbgemm")]
pub fn convert_to_channels_last3d_tensor(src: &Tensor) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(src.dim() == 5);
      Tensor dst;
      if (IsChannelsLast3d(src)) {
        dst = src;
      } else {
        const i64 N = src.size(0);
        const i64 C = src.size(1);
        const i64 D = src.size(2);
        const i64 H = src.size(3);
        const i64 W = src.size(4);
        dst = MakeStridedQTensorCPU(
            {N, C, D, H, W},
            {D * H * W * C, 1, H * W * C, W * C, C},
            src.options(),
            src.quantizer());
        AT_DISPATCH_QINT_TYPES(
            src.scalar_type(), "ConvertToChannelsLast3dTensor", [&]() {
              const Tensor src_contig = src.contiguous();
              CopyToChannelsLast3dTensor<Scalar>(
                  N,
                  C,
                  D,
                  H,
                  W,
                  src_contig.data_ptr<Scalar>(),
                  dst.data_ptr<Scalar>());
            });
      }
      return dst;
        */
}

#[cfg(feature = "fbgemm")]
pub fn transpose_conv_tensor_unpack_conversion2(
        src:    &Tensor,
        groups: i32) -> Tensor {
    
    todo!();
        /*
            // OC IC/G HW -> IC OC/G HW logically
      auto oc_g_ic_g_hw_tensors = src.chunk(groups);
      auto fused_tensor = cat(oc_g_ic_g_hw_tensors, 1);
      set_quantizer_(fused_tensor, src.quantizer());
      return fused_tensor.permute({1, 0, 2, 3});
        */
}

#[cfg(feature = "fbgemm")]
pub fn make_fbgemm_conv_param1(
        N:              i32,
        C:              i32,
        M:              i32,
        image_shape:    &Vec<i32>,
        groups:         i32,
        kernels:        &Vec<i32>,
        strides:        &Vec<i32>,
        pads:           &Vec<i32>,
        dilations:      &Vec<i32>,
        output_padding: &Vec<i32>,
        transposed:     bool) -> FbgemmConvParam1 {
    
    todo!();
        /*
        
        */
}

#[cfg(feature = "fbgemm")]
pub fn make_fbgemm_conv_param2(
        N:              i32,
        C:              i32,
        M:              i32,
        image_shape:    &Vec<i32>,
        groups:         i32,
        kernels:        &Vec<i32>,
        strides:        &Vec<i32>,
        pads:           &Vec<i32>,
        dilations:      &Vec<i32>,
        output_padding: &Vec<i32>,
        transposed:     bool) -> FbgemmConvParam2 {
    
    todo!();
        /*
        
        */
}

#[cfg(feature = "fbgemm")]
pub fn make_fbgemm_conv_param3(
        N:              i32,
        C:              i32,
        M:              i32,
        image_shape:    &Vec<i32>,
        groups:         i32,
        kernels:        &Vec<i32>,
        strides:        &Vec<i32>,
        pads:           &Vec<i32>,
        dilations:      &Vec<i32>,
        output_padding: &Vec<i32>,
        transposed:     bool) -> FbgemmConvParam3 {
    
    todo!();
        /*
        
        */
}

#[cfg(feature = "fbgemm")]
pub fn transpose_conv_tensor_unpack_conversion3(
        src:    &Tensor,
        groups: i32) -> Tensor {
    
    todo!();
        /*
            // OC IC/G DHW -> IC OC/G DHW logically
      auto oc_g_ic_g_hw_tensors = src.chunk(groups);
      auto fused_tensor = cat(oc_g_ic_g_hw_tensors, 1);
      set_quantizer_(fused_tensor, src.quantizer());
      return fused_tensor.permute({1, 0, 2, 3, 4});
        */
}

#[cfg(feature = "fbgemm")]
pub fn convert_conv_weights_to_channel_last_tensor2(
        src:       &Tensor,
        groups:    i32,
        transpose: bool) -> Tensor {
    
    todo!();
        /*
            return transpose ?
                       // 2D conv transpose weight transform
                       // IC OC/G KH KW -> G OC/G KH KW IC/G
          [&]() {
            auto ic_g_oc_g_hw_tensors = src.chunk(groups);
            for (auto& tensor : ic_g_oc_g_hw_tensors) {
              tensor = tensor.unsqueeze(0);
            }
            auto fused_tensor = cat(ic_g_oc_g_hw_tensors);
            set_quantizer_(fused_tensor, src.quantizer());
            return fused_tensor.permute({0, 2, 3, 4, 1})
                .contiguous(MemoryFormat::Contiguous);
          }()
                       // 2d conv weight transform
                       : src.contiguous(MemoryFormat::ChannelsLast);
        */
}

#[cfg(feature = "fbgemm")]
pub fn convert_conv_weights_to_channel_last_tensor3(
    src:       &Tensor,
    groups:    i32,
    transpose: bool) -> Tensor {
    
    todo!();
        /*
            if (!transpose) {
        return ConvertToChannelsLast3dTensor(src);
      } else {
        TORCH_CHECK(src.dim() == 5);
        Tensor dst;
        const i64 N = src.size(0);
        const i64 IC_G = N / groups;
        const i64 OC_G = src.size(1);
        const i64 D = src.size(2);
        const i64 H = src.size(3);
        const i64 W = src.size(4);
        dst = MakeStridedQTensorCPU(
            {groups * OC_G, IC_G, D, H, W},
            {D * H * W * IC_G, 1, H * W * IC_G, W * IC_G, IC_G},
            src.options(),
            src.quantizer());
        AT_DISPATCH_QINT_TYPES(
            src.scalar_type(), "CopyICFirst3dTensorToChannelsLast3dTensor", [&]() {
              const Tensor src_contig = src.contiguous();
              CopyICFirst3dTensorToChannelsLast3dTensor<Scalar>(
                  groups,
                  IC_G,
                  OC_G,
                  D,
                  H,
                  W,
                  src_contig.data_ptr<Scalar>(),
                  dst.data_ptr<Scalar>());
            });
        return dst;
      }
        */
}

pub fn register_conv_params<const SPATIAL_DIM: usize = 2>() -> TorchClass<ConvPackedParamsBase<SpatialDim>> {
    
    todo!();
        /*
            static auto register_conv_params =
        Torchclass_<ConvPackedParamsBase<kSpatialDim>>(
            "quantized", "Conv" + to_string(kSpatialDim) + "dPackedParamsBase")
        .def_pickle(
            [](const intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& params)
            -> ConvParamsSerializationType { // __getstate__
              return serialize_conv<kSpatialDim>(params);
            },
            // __setstate__ takes IValue because we support parsing historical
            // serialization versions.
            [](IValue v)
            -> intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> { // __setstate__
              ConvParamsSerializationType state = parse_conv_serialized_state<kSpatialDim>(v);
              return deserialize_conv<kSpatialDim>(state);
            })
        .def("weight", [](const intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& self) {
                         Tensor weight;
                         optional<Tensor> bias;
                         tie(weight, bias) = self->unpack();
                         return weight;
                       })
        .def("bias", [](const intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& self) {
                       Tensor weight;
                       optional<Tensor> bias;
                       tie(weight, bias) = self->unpack();
                       return bias;
                     })
        .def("unpack", &ConvPackedParamsBase<kSpatialDim>::unpack)
        .def("stride", &ConvPackedParamsBase<kSpatialDim>::stride)
        .def("padding", &ConvPackedParamsBase<kSpatialDim>::padding)
        .def("output_padding", &ConvPackedParamsBase<kSpatialDim>::output_padding)
        .def("dilation", &ConvPackedParamsBase<kSpatialDim>::dilation)
        .def("groups", &ConvPackedParamsBase<kSpatialDim>::groups)
        .def("transpose", &ConvPackedParamsBase<kSpatialDim>::transpose);
      return register_conv_params;
        */
}

pub fn register_embedding_params() -> TorchClass<EmbeddingPackedParamsBase> {
    
    todo!();
        /*
            // Type for __getstate__/__setstate__ serialization
      //
      // Element 0 is the version of the PackedParam structure
      // Element 1 is the Tensors contained in the Param instance
      // Element 2 is the double values (if any) contained in the Param instance
      // Element 3 is the int values (if any) contained in the Param instance

      using EmbeddingParamsSerializationType = tuple<
        i64, // version
        vector<Tensor>,
        vector<double>,
        vector<i64>>;

      static auto register_embedding_params =
        Torchclass_<EmbeddingPackedParamsBase>(
          "quantized", "EmbeddingPackedParamsBase")
          .def_pickle(
              [](const intrusive_ptr<EmbeddingPackedParamsBase>& params)
                  -> EmbeddingParamsSerializationType { // __getstate__ call
                Tensor weight = params->unpack();
                vector<Tensor> tensors_to_serialize = {weight};
                vector<double> doubles_to_serialize = {};
                i64 bit_rate = params->bit_rate();
                i64 version = params->version();
                vector<i64> longs_to_serialize = {bit_rate};
                return EmbeddingParamsSerializationType(
                  version,
                  move(tensors_to_serialize),
                  move(doubles_to_serialize),
                  move(longs_to_serialize));
              },
              [](EmbeddingParamsSerializationType state)
                  -> intrusive_ptr<EmbeddingPackedParamsBase> { // __setstate__ call

                vector<Tensor> tensors;
                vector<double> doubles;
                vector<i64> longs;
                i64 version;
                tie(version, tensors, doubles, longs) = move(state);

                TORCH_INTERNAL_ASSERT(tensors.size() == 1, "EmbeddingPackedParams: Expected weight tensor to be serialized");
                TORCH_INTERNAL_ASSERT(longs.size() == 1, "EmbeddingPackedParams: Expected bit_rate to be serialized");
                TORCH_CHECK(version == 1, "EmbeddingPackedParams: Currently only version 1 supported.");

                Tensor weight = move(tensors[0]);
                return PackedEmbeddingBagWeight::prepack(weight);
              })
          .def("bit_rate", &EmbeddingPackedParamsBase::bit_rate)
          .def("version", &EmbeddingPackedParamsBase::version);

      return register_embedding_params;
        */
}

lazy_static!{
    /*
    static auto conv2d_params = register_conv_params<2>();      
    static auto conv3d_params = register_conv_params<3>();      
    static auto linear_params = register_linear_params();       
    static auto embedding_params = register_embedding_params(); 
    */
}
