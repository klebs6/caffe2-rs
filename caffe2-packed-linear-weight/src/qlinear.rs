crate::ix!();

pub struct QLinearPackWeightInt8 {

}

impl QLinearPackWeightInt8 {
    
    pub fn run(
        weight:                  &Tensor,
        bias:                    &Option<Tensor>,
        out_features_block_size: i64,
        in_features_block_size:  i64) -> IntrusivePtr<LinearPackedParamsBase> {
        
        todo!();
        /*
            auto& ctx = globalContext();

    #ifdef USE_FBGEMM
        if (ctx.qEngine() == QEngine::FBGEMM) {
          return PackedLinearWeight::prepack(
              weight, bias, out_features_block_size, in_features_block_size);
        }
    #endif
    #ifdef USE_PYTORCH_QNNPACK
        if (ctx.qEngine() == QEngine::QNNPACK) {
          return PackedLinearWeightQnnp::prepack(
              weight, bias, out_features_block_size, in_features_block_size);
        }
    #endif
        TORCH_CHECK(
            false,
            "Didn't find engine for operation ao::sparse::qlinear_prepack ",
            toString(ctx.qEngine()));
        */
    }
}

lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(sparse, QuantizedCPU, m) {
      m.impl(
          TORCH_SELECTIVE_NAME("sparse::qlinear_prepack"),
          TORCH_FN(QLinearPackWeightInt8::run));
    }
    */
}

pub struct QLinearPackWeightInt8 {

}

impl QLinearPackWeightInt8 {
    
    pub fn run(
        weight: Tensor,
        bias:   Option<Tensor>) -> IntrusivePtr<LinearPackedParamsBase> {
        
        todo!();
        /*
            auto& ctx = globalContext();

    #ifdef USE_FBGEMM
        if (ctx.qEngine() == QEngine::FBGEMM) {
          return PackedLinearWeight::prepack(move(weight), move(bias));
        }
    #endif
    #ifdef USE_PYTORCH_QNNPACK
        if (ctx.qEngine() == QEngine::QNNPACK) {
          return PackedLinearWeightsQnnp::prepack(
              move(weight), move(bias));
        }
    #endif
        TORCH_CHECK(
            false,
            "Didn't find engine for operation quantized::linear_prepack ",
            toString(ctx.qEngine()));
        */
    }
}

pub struct QLinearPackWeightFp16 {

}

impl QLinearPackWeightFp16 {
    
    pub fn run(
        weight: Tensor,
        bias:   Option<Tensor>) -> IntrusivePtr<LinearPackedParamsBase> {
        
        todo!();
        /*
            auto& ctx = globalContext();
    #ifdef USE_FBGEMM
        if (ctx.qEngine() == QEngine::FBGEMM) {
          return PackedLinearWeightFp16::prepack(
              move(weight), move(bias));
        }
    #endif // USE_FBGEMM
    #ifdef USE_PYTORCH_QNNPACK
        if (ctx.qEngine() == QEngine::QNNPACK) {
          TORCH_CHECK(
              false,
              "quantized::linear_prepack_fp16 is currently "
              "not supported by QNNPACK");
        }
    #endif // USE_PYTORCH_QNNPACK
        TORCH_CHECK(
            false,
            "Didn't find engine for operation quantized::linear_prepack_fp16 ",
            toString(ctx.qEngine()));
        */
    }
}

pub struct QLinearPackWeightInt8Legacy {

}

impl QLinearPackWeightInt8Legacy {
    
    pub fn run(
        weight: Tensor,
        bias:   Option<Tensor>) -> Tensor {
        
        todo!();
        /*
            auto& ctx = globalContext();
        auto options = weight.options();

    #ifdef USE_FBGEMM
        if (ctx.qEngine() == QEngine::FBGEMM) {
          auto prepacked =
              PackedLinearWeight::prepack(move(weight), move(bias));
          auto wrapped =
              make_unique<intrusive_ptr<LinearPackedParamsBase>>(
                  move(prepacked));
          return cpp_custom_type_hack::create(move(wrapped), options);
        }
    #endif // USE_FBGEMM
    #ifdef USE_PYTORCH_QNNPACK
        if (ctx.qEngine() == QEngine::QNNPACK) {
          auto prepacked =
              PackedLinearWeightsQnnp::prepack(move(weight), move(bias));
          auto wrapped =
              make_unique<intrusive_ptr<LinearPackedParamsBase>>(
                  move(prepacked));
          return cpp_custom_type_hack::create(move(wrapped), options);
        }
    #endif // USE_PYTORCH_QNNPACK
        TORCH_CHECK(
            false,
            "Didn't find engine for operation quantized::linear_prepack ",
            toString(ctx.qEngine()));
        */
    }
}

pub struct QLinearPackWeightFp16Legacy {

}

impl QLinearPackWeightFp16Legacy {
    
    pub fn run(
        weight: Tensor,
        bias:   Option<Tensor>) -> Tensor {
        
        todo!();
        /*
            auto& ctx = globalContext();
        auto options = weight.options();
    #ifdef USE_FBGEMM
        if (ctx.qEngine() == QEngine::FBGEMM) {
          auto prepacked =
              PackedLinearWeightFp16::prepack(move(weight), move(bias));
          auto wrapped =
              make_unique<intrusive_ptr<LinearPackedParamsBase>>(
                  move(prepacked));
          return cpp_custom_type_hack::create(move(wrapped), options);
        }
    #endif // USE_FBGEMM
    #ifdef USE_PYTORCH_QNNPACK
        if (ctx.qEngine() == QEngine::QNNPACK) {
          TORCH_CHECK(
              false,
              "quantized::linear_prepack_fp16 is currently "
              "not supported by QNNPACK");
        }
    #endif // USE_PYTORCH_QNNPACK
        TORCH_CHECK(
            false,
            "Didn't find engine for operation quantized::linear_prepack_fp16 ",
            toString(ctx.qEngine()));
        */
    }
}


//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/ao_sparse/quantized/cpu/qlinear_unpack.cpp]

pub struct QLinearUnpackWeightInt8 {

}

impl QLinearUnpackWeightInt8 {
    
    pub fn run(packed_weight: &IntrusivePtr<LinearPackedParamsBase>) -> LinearPackedSerializationType {
        
        todo!();
        /*
            return packed_weight->unpack();
        */
    }
}

lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(sparse, QuantizedCPU, m) {
      m.impl(
          TORCH_SELECTIVE_NAME("sparse::qlinear_unpack"),
          TORCH_FN(QLinearUnpackWeightInt8::run));
    }
    */
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/ao_sparse/library.cpp]

/// Register operators
lazy_static!{
    /*
    TORCH_LIBRARY(sparse, m) {
      ao::sparse::register_linear_params();

      m.def(TORCH_SELECTIVE_SCHEMA(
          "sparse::qlinear(Tensor X, __torch__.torch.classes.sparse.LinearPackedParamsBase W_prepack, float Y_scale_i, int Y_zero_point_i) -> Tensor Y"));
      m.def(TORCH_SELECTIVE_SCHEMA(
          "sparse::qlinear_relu(Tensor X, __torch__.torch.classes.sparse.LinearPackedParamsBase W_prepack, float Y_scale_i, int Y_zero_point_i) -> Tensor Y"));

      m.def(TORCH_SELECTIVE_SCHEMA(
          "sparse::qlinear_dynamic(Tensor X, __torch__.torch.classes.sparse.LinearPackedParamsBase W_prepack) -> Tensor Y"));
      m.def(TORCH_SELECTIVE_SCHEMA(
          "sparse::qlinear_relu_dynamic(Tensor X, __torch__.torch.classes.sparse.LinearPackedParamsBase W_prepack) -> Tensor Y"));

      m.def(TORCH_SELECTIVE_SCHEMA(
          "sparse::qlinear_prepack(Tensor W, Tensor? B, int out_features_block_size, int in_features_block_size) -> __torch__.torch.classes.sparse.LinearPackedParamsBase W_prepack"));

      m.def(TORCH_SELECTIVE_SCHEMA(
          "sparse::qlinear_unpack(__torch__.torch.classes.sparse.LinearPackedParamsBase W_prepack) -> (Tensor W_origin, Tensor? B_origin, int[] block_pattern)"));
    }
    */
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/ao_sparse/quantized/cpu/qlinear.cpp]

pub struct QLinearInt8<const ReluFused: bool> {

}

impl<const ReluFused: bool> QLinearInt8<ReluFused> {
    
    pub fn run(
        input:             &Tensor,
        packed_weight:     &IntrusivePtr<LinearPackedParamsBase>,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor {
        
        todo!();
        /*
            if (ReluFused) {
          return packed_weight->apply_relu(input, output_scale, output_zero_point);
        } else {
          return packed_weight->apply(input, output_scale, output_zero_point);
        }
        */
    }
}

lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(sparse, QuantizedCPU, m) {
      m.impl(
          TORCH_SELECTIVE_NAME("sparse::qlinear"),
          TORCH_FN(QLinearInt8<false>::run));
      m.impl(
          TORCH_SELECTIVE_NAME("sparse::qlinear_relu"),
          TORCH_FN(QLinearInt8<true>::run));
    }
    */
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/ao_sparse/quantized/cpu/qlinear_dynamic.cpp]

pub struct QLinearDynamicInt8<const ReluFused: bool> {

}

impl<const ReluFused: bool> QLinearDynamicInt8<ReluFused> {
    
    pub fn run(
        input:         &Tensor,
        packed_weight: &IntrusivePtr<LinearPackedParamsBase>) -> Tensor {
        
        todo!();
        /*
            auto& ctx = globalContext();
    #ifdef USE_PYTORCH_QNNPACK
        if (ctx.qEngine() == QEngine::QNNPACK) {
          if (ReluFused) {
            return packed_weight->apply_dynamic_relu(input);
          } else {
            return packed_weight->apply_dynamic(input);
          }
        }
    #endif
        TORCH_CHECK(
            false,
            "Didn't find engine for operation ao::sparse::qlinear_dynamic",
            toString(ctx.qEngine()));
        */
    }
}

lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(sparse, CPU, m) {
      m.impl(
          TORCH_SELECTIVE_NAME("sparse::qlinear_dynamic"),
          TORCH_FN(QLinearDynamicInt8<false>::run));
      m.impl(
          TORCH_SELECTIVE_NAME("sparse::qlinear_relu_dynamic"),
          TORCH_FN(QLinearDynamicInt8<true>::run));
    }
    */
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/ao_sparse/quantized/cpu/qlinear_prepack.cpp]


/**
  | Calculate the column offsets.
  |
  | Note this includes the sum of the columns as
  | well as the scalar term B_zero_point * K,
  | whereas the row_offsets created by packing of
  | activation is only the sum of the A rows.
  */
#[cfg(feature = "fbgemm")]
pub fn calc_col_offsets_transpose(
        K:            i32,
        N:            i32,
        bint8:        *const i8,
        b_zero_point: *mut i32,
        col_offsets:  *mut i32,
        qtype:        QScheme)  {
    
    todo!();
        /*
      for (const auto i : irange(N)) {
        i32 sum = 0;
        for (const auto j : irange(K)) {
          sum += Bint8[i * K + j];
        }
        if (qtype == kPerTensorAffine) {
          col_offsets[i] = sum - B_zero_point[0] * K;
        } else {
          col_offsets[i] = sum - B_zero_point[i] * K;
        }
      }
        */
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qlinear_unpack.cpp]

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

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qlinear_dynamic.cpp]

pub struct QLinearDynamicInt8<const ReluFused: bool> {

}

impl<const ReluFused: bool> QLinearDynamicInt8<ReluFused> {
    
    pub fn run(
        input:         Tensor,
        packed_weight: &IntrusivePtr<LinearPackedParamsBase>,
        reduce_range:  bool) -> Tensor {
        
        todo!();
        /*
            if (ReluFused) {
          return packed_weight->apply_dynamic_relu(move(input), reduce_range);
        } else {
          return packed_weight->apply_dynamic(move(input), reduce_range);
        }
        */
    }
}

pub struct QLinearDynamicFp16<const ReluFused: bool> {

}

impl<const ReluFused: bool> QLinearDynamicFp16<ReluFused> {

    #[cfg(feature = "fbgemm")]
    pub fn run(
        input:         Tensor,
        packed_weight: &IntrusivePtr<LinearPackedParamsBase>) -> Tensor {
        
        todo!();
        /*
            // We make a strong guarantee that models using these operators will have
        // the same numerics across different machines. Therefore, we do not provide
        // a fallback path and rather fail loudly if we cannot run FBGEMM.
        TORCH_CHECK(
            fbgemm::fbgemmSupportedCPU(), "Your CPU doesn't support FBGEMM.");

        TORCH_INTERNAL_ASSERT(!ReluFused);
        return packed_weight->apply_dynamic(move(input));
        */
    }

    #[cfg(not(feature = "fbgemm"))]
    pub fn run(
        input:         Tensor,
        packed_weight: &IntrusivePtr<LinearPackedParamsBase>) -> Tensor {
        
        todo!();
        /*
            // We make a strong guarantee that models using these operators will have
        // the same numerics across different machines. Therefore, we do not provide
        // a fallback path and rather fail loudly if we cannot run FBGEMM.
        TORCH_CHECK(
            false, "This PyTorch installation was not built with FBGEMM operators");
        */
    }
}

lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(quantized, CPU, m) {
      m.impl(TORCH_SELECTIVE_NAME("quantized::linear_dynamic"), TORCH_FN(QLinearDynamicInt8<false>::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::linear_relu_dynamic"), TORCH_FN(QLinearDynamicInt8<true>::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::linear_dynamic_fp16"), TORCH_FN(QLinearDynamicFp16<false>::run));
    }
    */
}

lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(_quantized, CPU, m) {
      m.impl(TORCH_SELECTIVE_NAME("_quantized::linear_dynamic"), TORCH_FN(QLinearDynamicInt8<false>::run));
    }
    */
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qlinear.cpp]

pub struct QLinearInt8<const ReluFused: bool> {

}

impl<const ReluFused: bool> QLinearInt8<ReluFused> {
    
    pub fn run(
        input:             Tensor,
        packed_weight:     &IntrusivePtr<LinearPackedParamsBase>,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor {
        
        todo!();
        /*
            if (ReluFused) {
          return packed_weight->apply_relu(
              move(input), output_scale, output_zero_point);
        } else {
          return packed_weight->apply(
              move(input), output_scale, output_zero_point);
        }
        */
    }
}

lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
      m.impl(TORCH_SELECTIVE_NAME("quantized::linear"), TORCH_FN(QLinearInt8<false>::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::linear_relu"), TORCH_FN(QLinearInt8<true>::run));
    }
    */
}

lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(_quantized, QuantizedCPU, m) {
      m.impl(TORCH_SELECTIVE_NAME("_quantized::linear"), TORCH_FN(QLinearInt8<false>::run));
    }
    */
}
