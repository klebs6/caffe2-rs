crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/quantized/Quantizer.h]

/**
  | UniformQuantizer is the parent class
  | for all uniform quantizers.
  | 
  | These quantization scheme will map
  | float value uniformly to the quantized
  | value. For example, affine quantizer
  | is the most commonly used scheme in this
  | category.
  |
  */
pub struct UniformQuantizer {
    base: Quantizer,
}

impl UniformQuantizer {

    pub fn new(scalar_type: ScalarType) -> Self {
    
        todo!();
        /*
        : quantizer(scalar_type),
        */
    }
}

/**
  | NonUniformQuantizer is the parent
  | class for all non-uniform quantizers.
  | 
  | These quantization scheme may map float
  | value non-uniformly to the quantized
  | value. K-means quantization is a representative
  | example in this category.
  |
  */
pub struct NonUniformQuantizer {
    base: Quantizer,
}

impl NonUniformQuantizer {
    
    pub fn new(scalar_type: ScalarType) -> Self {
    
        todo!();
        /*
        : quantizer(scalar_type),
        */
    }
}

/**
  | There is also StochasticQuantizer which is
  | uniform but not affine
  |
  | AffineQuantizer uses affine transformation
  | to do quantization.
  | 
  | For quantize:
  | 
  | Y = clamp(round(X / scale + zero_point),
  | min, max)
  | 
  | For dequantize:
  | 
  | X = (Y - zero_point) * scale
  |
  */
pub struct AffineQuantizer {
    base: UniformQuantizer,
}

impl AffineQuantizer {
    
    pub fn new(scalar_type: ScalarType) -> Self {
    
        todo!();
        /*
        : uniform_quantizer(scalar_type),
        */
    }
}

// Note that we will not have Symmetric Quantizer in backend to reduce
// complications in quantized kernel implementation.

/**
  | PerTensorAffineQuantizer stores
  | a scale and a zero_point, which is used
  | for all the values in the Tensor.
  |
  */
pub struct PerTensorAffineQuantizer {
    base:       AffineQuantizer,
    scale:      f64,

    /**
      | We use i64 for consistency with
      | Python
      |
      */
    zero_point: i64,
}

impl PerTensorAffineQuantizer {
    
    pub fn new(
        scalar_type: ScalarType,
        scale:       f64,
        zero_point:  i64) -> Self {
    
        todo!();
        /*
        : affine_quantizer(scalar_type),
        : scale(scale),
        : zero_point(zero_point),

        
        */
    }
   
    pub fn qscheme(&self) -> QScheme {
        
        todo!();
        /*
            return kPerTensorAffine;
        */
    }
    
    pub fn scale(&self) -> f64 {
        
        todo!();
        /*
            return scale_;
        */
    }
    
    pub fn zero_point(&self) -> i64 {
        
        todo!();
        /*
            return zero_point_;
        */
    }
    
    pub fn equal_to(&mut self, other: QuantizerPtr) -> bool {
        
        todo!();
        /*
            if (!other.get() || other->qscheme() != kPerTensorAffine) {
          return false;
        }
        auto* other_per_tensor_affine =
            static_cast<PerTensorAffineQuantizer*>(other.get());
        return scalar_type() == other_per_tensor_affine->scalar_type() &&
            scale() == other_per_tensor_affine->scale() &&
            zero_point() == other_per_tensor_affine->zero_point();
        */
    }
}

/**
  | PerChannelAffineQuantizer is the
  | same as PerTensorAffineQuantizer
  | except that we have an independent scale
  | and zero_point parameter for each channel.
  | 
  | Also note that per channel quantization
  | is mostly applied to output channels
  | of weights since per-input channel
  | of weight quantization or per-channel
  | quantization for activations can't
  | be efficiently supported in most of
  | processors since it requires each multiplication
  | result within a single dot-product
  | to have a different scale.
  |
  */
pub struct PerChannelAffineQuantizer {
    base:        AffineQuantizer,
    scales:      Tensor,
    zero_points: Tensor,
    axis:        i64,
}

impl PerChannelAffineQuantizer {
    
    pub fn new(
        scalar_type: ScalarType,
        scales:      Tensor,
        zero_points: Tensor,
        axis:        i64) -> Self {
    
        todo!();
        /*
        : affine_quantizer(scalar_type),
        : scales(scales),
        : zero_points(zero_points),
        : axis(axis),
        */
    }
    
    pub fn qscheme(&self) -> QScheme {
        
        todo!();
        /*
            return kPerChannelAffine;
        */
    }
    
    pub fn scales(&self) -> Tensor {
        
        todo!();
        /*
            return scales_;
        */
    }
    
    pub fn zero_points(&self) -> Tensor {
        
        todo!();
        /*
            return zero_points_;
        */
    }
    
    pub fn axis(&self) -> i64 {
        
        todo!();
        /*
            return axis_;
        */
    }
    
    pub fn equal_to(&mut self, other: QuantizerPtr) -> bool {
        
        todo!();
        /*
            if (!other.get() || other->qscheme() != kPerChannelAffine) {
          return false;
        }
        auto* other_per_channel_affine =
            static_cast<PerChannelAffineQuantizer*>(other.get());
        return scalar_type() == other_per_channel_affine->scalar_type() &&
            scales().equal(other_per_channel_affine->scales()) &&
            zero_points().equal(other_per_channel_affine->zero_points()) &&
            axis() == other_per_channel_affine->axis();
        */
    }
}

/**
  | PerChannelAffineFloatQParamsQuantizer
  | is the same as PerChannelAffineQuantizer
  | except that it expects both scale and
  | zero point to be floating point values.
  | 
  | This quantizer uses the kPerChannelAffineFloatQParams
  | qscheme which is a variant of kPerChannelAffine.
  | 
  | The quantize equation in this case looks
  | like -
  | 
  | Xq = (Xf - zero_point) * inv_scale, where
  | inv_scale = 1.0/scale
  | 
  | -----------
  | @note
  | 
  | Usage of floating point zero point is
  | useful in cases where 0 doesn't need
  | to be exactly represented in the quantized
  | space. We can get additional precision
  | by using floating point values for zero
  | point.
  |
  */
pub struct PerChannelAffineFloatQParamsQuantizer {
    base: PerChannelAffineQuantizer,
}

impl PerChannelAffineFloatQParamsQuantizer {
    
    pub fn new(
        scalar_type: ScalarType,
        scales:      Tensor,
        zero_points: Tensor,
        axis:        i64) -> Self {
    
        todo!();
        /*
        : per_channel_affine_quantizer(scalar_type,
                scales,
                zero_points,
                axis),
        */
    }
    
    pub fn qscheme(&self) -> QScheme {
        
        todo!();
        /*
            return kPerChannelAffineFloatQParams;
        */
    }
    
    pub fn equal_to(&mut self, other: QuantizerPtr) -> bool {
        
        todo!();
        /*
            if (!other.get() || other->qscheme() != kPerChannelAffineFloatQParams) {
          return false;
        }
        auto* other_per_channel_float_qparams =
            static_cast<PerChannelAffineFloatQParamsQuantizer*>(other.get());
        return scalar_type() == other_per_channel_float_qparams->scalar_type() &&
            scales().equal(other_per_channel_float_qparams->scales()) &&
            zero_points().equal(other_per_channel_float_qparams->zero_points()) &&
            axis() == other_per_channel_float_qparams->axis();
        */
    }
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/quantized/Quantizer.cpp]

pub fn check_per_channel_param_dims(
    scales:      &Tensor,
    zero_points: &Tensor)  {
    
    todo!();
        /*
            TORCH_CHECK(scales.dim() == 1, "scale tensor must have dimension 1");
        TORCH_CHECK(
            zero_points.dim() == 1, "zero_points tensor must have dimension 1");
        TORCH_CHECK(
            scales.numel() == zero_points.numel(),
            "number of elements in scales and zero_points must match");
        */
}

impl Tensor {
    
    /**
      | -----------
      | @note
      | 
      | this is not a native function as
      | 
      | Quantizer is not exposed to python yet
      |
      */
    pub fn quantizer(&self) -> QuantizerPtr {
        
        todo!();
        /*
            // This is a terrible hack to emulate what VariableType is doing
      AutoDispatchBelowAutograd mode;
      return get_qtensorimpl(*this)->quantizer();
        */
    }
}

/**
  | double and i64 are because of the native
  | function API, we only have these argument types
  | right now in native functions
  |
  */
pub fn make_per_tensor_affine_quantizer(
        scale:       f64,
        zero_point:  i64,
        scalar_type: ScalarType) -> QuantizerPtr {
    
    todo!();
        /*
            return make_intrusive<PerTensorAffineQuantizer>(scalar_type,
          scale, zero_point);
        */
}

pub fn make_per_channel_affine_quantizer(
    scales:      &Tensor,
    zero_points: &Tensor,
    axis:        i64,
    scalar_type: ScalarType) -> QuantizerPtr {
    
    todo!();
        /*
            checkPerChannelParamDims(scales, zero_points);
      TORCH_CHECK(
          isFloatingType(scales.scalar_type()),
          "scale tensor must be floating point");

      if (isFloatingType(zero_points.scalar_type())) {
        Tensor scales_float = scales.to(kFloat).contiguous();
        Tensor zero_points_float = zero_points.to(kFloat).contiguous();
        return make_intrusive<PerChannelAffineFloatQParamsQuantizer>(scalar_type,
                                                                          scales_float,
                                                                          zero_points_float,
                                                                          axis);
      }
      else {
        Tensor scales_double = scales.to(kDouble).contiguous();
        Tensor zero_points_int64 = zero_points.to(kLong).contiguous();
        return make_intrusive<PerChannelAffineQuantizer>(scalar_type,
                                                              scales_double,
                                                              zero_points_int64,
                                                              axis);
      }
        */
}

/**
  | This is an internal utility function for
  | getting at the QTensorImpl,
  |
  | You should only use this for writing low level
  | setters/getters for QTensorImpl fields;
  |
  | otherwise, you should use the low level
  | setters/getters that were implemented using
  | this.
  |
  | This may be called repeatedly, so make sure
  | it's pretty cheap.
  |
  */
pub fn get_qtensorimpl(self_: &Tensor) -> *mut QTensorImpl {
    
    todo!();
        /*
            TORCH_CHECK(
          !self.requires_grad(),
          "quantized tensors do not support autograd");
      TORCH_INTERNAL_ASSERT(self.is_quantized(), "get_qtensorimpl: not a quantized tensor");
      return static_cast<QTensorImpl*>(self.unsafeGetTensorImpl());
        */
}

pub fn get_sub_byte_tensor_size(
        size_bytes: i64,
        t:          ScalarType) -> i64 {
    
    todo!();
        /*
      i64 new_size_bytes;
      switch(t) {
        case ScalarType::QUInt4x2:
          new_size_bytes = ceil(size_bytes * 0.5);
          break;
        default:
          new_size_bytes = size_bytes;
      }
      return new_size_bytes;
        */
}

/**
  | Create a Quantized Tensor given arguments
  | for normal Tensor and a quantizer
  |
  */
#[inline] pub fn new_qtensor(
    sizes:     &[i32],
    options:   &TensorOptions,
    quantizer: QuantizerPtr) -> Tensor {

    todo!();
        /*
            auto memory_format = options.memory_format_opt().value_or(MemoryFormat::Contiguous);
      Allocator* allocator = options.device().is_cuda()
        ? getCUDAHooks().getCUDADeviceAllocator()
        : getCPUAllocator();

    #ifdef USE_PYTORCH_QNNPACK
      if (globalContext().qEngine() == QEngine::QNNPACK) {
        allocator = GetDefaultMobileCPUAllocator();
      }
    #endif

      DispatchKey tensorDispatchKey = options.computeDispatchKey();
      native::check_size_nonnegative(sizes);
      i64 nelements = multiply_integers(sizes);
      auto dtype = options.dtype();
      TORCH_CHECK(
          isQIntType(typeMetaToScalarType(dtype)),
          "ScalarType is not supported in new_qtensor.");
      auto scalar_type = typeMetaToScalarType(dtype);
      i64 size_bytes = get_sub_byte_tensor_size(nelements * dtype.itemsize(), scalar_type);

      auto storage = make_intrusive<StorageImpl>(
          StorageImpl::use_byte_size_t(),
          size_bytes,
          allocator->allocate(size_bytes),
          allocator,
          /*resizable=*/true);
      auto tensor = make_tensor<QTensorImpl>(
          storage, DispatchKeySet(tensorDispatchKey), dtype, quantizer);
      get_qtensorimpl(tensor)->set_sizes_contiguous(sizes);
      get_qtensorimpl(tensor)->empty_tensor_restride(memory_format);
      return tensor;
        */
}

impl PerTensorAffineQuantizer {
    
    pub fn quantize(&mut self, rtensor: &Tensor) -> Tensor {
        
        todo!();
        /*
            TORCH_CHECK(
          rtensor.scalar_type() == kFloat, "quantize only works on Float Tensor.");
      // Here we need a intrusive_ptr<Quantizer>.. but actually "this" is the
      // quantizer that can be reused, so I'm using intrusive_from_this here
      Tensor qtensor = new_qtensor(
          rtensor.sizes(),
          rtensor.options()
              .dtype(scalar_type_)
              .memory_format(rtensor.suggest_memory_format()),
          intrusive_from_this());

      auto rtensor_contig = rtensor.expect_contiguous(rtensor.suggest_memory_format());
      native::quantize_tensor_per_tensor_affine(
          *rtensor_contig, qtensor, scale_, zero_point_);
      return qtensor;
        */
    }
    
    pub fn dequantize(&mut self, qtensor: &Tensor) -> Tensor {
        
        todo!();
        /*
            Tensor rtensor = empty(
          qtensor.sizes(),
          qtensor.options()
              .dtype(kFloat)
              .memory_format(qtensor.suggest_memory_format()));
      auto qtensor_contig = qtensor.expect_contiguous(qtensor.suggest_memory_format());
      native::dequantize_tensor_per_tensor_affine(
          *qtensor_contig, rtensor, scale_, zero_point_);
      return rtensor;
        */
    }
}

impl PerChanneAffineQuantizer {
    
    pub fn quantize(&mut self, rtensor: &Tensor) -> Tensor {
        
        todo!();
        /*
            // Here we need a intrusive_ptr<Quantizer>.. but actually "this" is the
      // quantizer that can be reused, so I'm using intrusive_from_this here
      Tensor qtensor = new_qtensor(
          rtensor.sizes(),
          rtensor.options()
              .dtype(scalar_type_)
              .memory_format(rtensor.suggest_memory_format()),
          intrusive_from_this());
      auto rtensor_contig = rtensor.expect_contiguous(rtensor.suggest_memory_format());
      native::quantize_tensor_per_channel_affine(
          *rtensor_contig, qtensor, scales_, zero_points_, axis_);
      return qtensor;
        */
    }
    
    pub fn dequantize(&mut self, qtensor: &Tensor) -> Tensor {
        
        todo!();
        /*
            Tensor rtensor = empty(
          qtensor.sizes(),
          qtensor.options()
              .dtype(kFloat)
              .memory_format(qtensor.suggest_memory_format()));
      auto qtensor_contig = qtensor.expect_contiguous(qtensor.suggest_memory_format());
      native::dequantize_tensor_per_channel_affine(
          *qtensor_contig, rtensor, scales_, zero_points_, axis_);
      return rtensor;
        */
    }
}

impl PerChannelAffineFloatQParamsQuantizer {

    pub fn quantize(&mut self, rtensor: &Tensor) -> Tensor {
        
        todo!();
        /*
            TORCH_CHECK(
          rtensor.scalar_type() == kFloat, "quantize only works on Float Tensor.");
     Tensor qtensor = new_qtensor(
          rtensor.sizes(),
          rtensor.options().dtype(scalar_type_),
          intrusive_from_this());
     auto rtensor_contig = rtensor.expect_contiguous();
     native::quantize_tensor_per_channel_float_qparams(
       *rtensor_contig, qtensor, scales_, zero_points_, axis_);
      return qtensor;
        */
    }
    
    pub fn dequantize(&mut self, qtensor: &Tensor) -> Tensor {
        
        todo!();
        /*
            Tensor rtensor = empty(qtensor.sizes(), qtensor.options().dtype(kFloat));
      auto qtensor_contig = qtensor.expect_contiguous();
      native::dequantize_tensor_per_channel_float_qparams(
        *qtensor_contig, rtensor, scales_, zero_points_, axis_);
      return rtensor;
        */
    }
}

pub fn set_quantizer(
    self_:     &Tensor,
    quantizer: ConstQuantizerPtr)  {
    
    todo!();
        /*
            get_qtensorimpl(self)->set_quantizer_(quantizer);
        */
}
