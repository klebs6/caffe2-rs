crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qlinear_unpack.cpp]

pub fn register_linear_params() -> TorchClass<LinearPackedParamsBase> {
    
    todo!();
        /*
        
        */
}

#[cfg(feature = "fbgemm")]
impl PackedLinearWeight {
    
    pub fn unpack(&mut self) -> (Tensor,Option<Tensor>) {
        
        todo!();
        /*
            auto packB = w.get();

      i64 N = static_cast<i64>(packB->numCols());
      i64 K = static_cast<i64>(packB->numRows());

      Tensor weight_origin;
      if (q_scheme == kPerTensorAffine) {
        weight_origin = _empty_affine_quantized(
            {N, K}, device(kCPU).dtype(kQInt8), w_scale[0], w_zp[0]);
      } else if (q_scheme == kPerChannelAffine) {
        auto scales = from_blob(
            w_scale.data(), w_scale.size(), device(kCPU).dtype(kFloat));
        auto zero_points = from_blob(
            w_zp.data(), w_zp.size(), device(kCPU).dtype(kInt));

        weight_origin = _empty_per_channel_affine_quantized(
            {N, K},
            scales.toType(kDouble),
            zero_points.toType(kLong),
            0, // The output channel axis is 0
            device(kCPU).dtype(kQInt8));
      }

      i8* weight_ptr_int8 =
          reinterpret_cast<i8*>(weight_origin.data_ptr<qint8>());

      // packB->printPackedMatrix("packedB inside fbgemm_unpack
      // (QLinearUnpackWeightInt8): ");
      packB->unpack(weight_ptr_int8);

      return tuple<Tensor, optional<Tensor>>(
          weight_origin, bias_);
        */
    }
}

#[cfg(USE_PYTORCH_QNNPACK)]
impl PackedLinearWeightsQnnp {
    
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
}

#[cfg(feature = "fbgemm")]
impl PackedLinearWeightFp16 {
    
    pub fn unpack(&mut self) -> (Tensor,Option<Tensor>) {
        
        todo!();
        /*
            auto& packed_weight_ptr = w;

      auto nrows = packed_weight_ptr->numRows();
      auto ncols = packed_weight_ptr->numCols();

      Tensor unpacked_weight =
          empty({ncols, nrows}, kHalf, MemoryFormat::Contiguous);
      packed_weight_ptr->unpack(
          static_cast<fbgemm::float16*>(unpacked_weight.data_ptr()),
          fbgemm::matrix_op_t::Transpose);

      return make_tuple(unpacked_weight.to(kFloat), bias_);
        */
    }
}

pub struct QLinearUnpackWeightInt8 {

}

impl QLinearUnpackWeightInt8 {
    
    pub fn run(packed_weight: &IntrusivePtr<LinearPackedParamsBase>) -> (Tensor,Option<Tensor>) {
        
        todo!();
        /*
            return packed_weight->unpack();
        */
    }
}

pub struct QLinearUnpackWeightFp16 {

}

impl QLinearUnpackWeightFp16 {
    
    pub fn run(packed_weight: &IntrusivePtr<LinearPackedParamsBase>) -> (Tensor,Option<Tensor>) {
        
        todo!();
        /*
            auto& ctx = globalContext();

        TORCH_CHECK(
            ctx.qEngine() != QEngine::QNNPACK,
            "quantized::linear_unpack_fp16 is currently "
            "not supported by QNNPACK");

        return packed_weight->unpack();
        */
    }
}

pub struct QLinearUnpackWeightInt8Legacy {

}

impl QLinearUnpackWeightInt8Legacy {
    
    pub fn run(packed_weight: &Tensor) -> (Tensor,Option<Tensor>) {
        
        todo!();
        /*
            TORCH_WARN_ONCE(
            "quantized.linear_unpack(Tensor) is deprecated! Please "
            "upgrade your model to use the newer quantized.linear_"
            "unpack(LinearPackedParamsBase) overload");
        return cpp_custom_type_hack::cast<
                   intrusive_ptr<LinearPackedParamsBase>>(packed_weight)
            ->unpack();
        */
    }
}

pub struct QLinearUnpackWeightFp16Legacy {

}

impl QLinearUnpackWeightFp16Legacy {
    
    pub fn run(packed_weight: &Tensor) -> (Tensor,Option<Tensor>) {
        
        todo!();
        /*
            TORCH_WARN_ONCE(
            "quantized.linear_unpack(Tensor) is deprecated! Please "
            "upgrade your model to use the newer quantized.linear_"
            "unpack(LinearPackedParamsBase) overload");
        auto& ctx = globalContext();

        TORCH_CHECK(
            ctx.qEngine() != QEngine::QNNPACK,
            "quantized::linear_unpack_fp16 is currently "
            "not supported by QNNPACK");

        return cpp_custom_type_hack::cast<
                   intrusive_ptr<LinearPackedParamsBase>>(packed_weight)
            ->unpack();
        */
    }
}

lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(quantized, CPU, m) {
      m.impl(TORCH_SELECTIVE_NAME("quantized::linear_unpack.legacy"), TORCH_FN(QLinearUnpackWeightInt8Legacy::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::linear_unpack_fp16.legacy"), TORCH_FN(QLinearUnpackWeightFp16Legacy::run));
    }
    */
}

lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(quantized, CatchAll, m) {
      m.impl(TORCH_SELECTIVE_NAME("quantized::linear_unpack"), TORCH_FN(QLinearUnpackWeightInt8::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::linear_unpack_fp16"), TORCH_FN(QLinearUnpackWeightFp16::run));
    }
    */
}
