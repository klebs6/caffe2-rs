crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qlinear_prepack.cpp]

pub fn register_linear_params() -> TorchClass<LinearPackedParamsBase> {
    
    todo!();
        /*
        
        */
}

/**
  | Calculate the column offsets.
  |
  | Note this includes the sum of the columns as
  | well as the scalar term B_zero_point * K,
  | whereas the row_offsets created by
  | PackAWithQuantRowOffset is only the sum of the
  | A rows.
  |
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
      for (usize i = 0; i < N; ++i) {
        i32 sum = 0;
        for (usize j = 0; j < K; ++j) {
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

#[cfg(feature = "fbgemm")]
impl PackedLinearWeight {
    
    pub fn prepack(&mut self, 
        weight: Tensor,
        bias:   Option<Tensor>) -> IntrusivePtr<LinearPackedParamsBase> {
        
        todo!();
        /*
            TORCH_CHECK(
          weight.dim() == 2,
          "The weight tensor for quantized::linear_prepack (fbgemm) should"
          " be 2-dimensional.");

      auto N = weight.size(0);
      auto K = weight.size(1);

      // TODO: contiguous is called for further JIT optimizations.
      auto weight_contig = weight.contiguous();
      const auto qtype = weight.qscheme();
      vector<i32> weight_zero_points_int32(1, 0);
      if (qtype == kPerTensorAffine) {
        weight_zero_points_int32[0] = weight.q_zero_point();
      } else if (qtype == kPerChannelAffine) {
        weight_zero_points_int32.resize(N, 0);
        for (const auto i : irange(N)) {
          weight_zero_points_int32[i] =
              weight.q_per_channel_zero_points()[i].item<i32>();
        }
      }
      vector<float> weight_scales_float(1, 0.0);
      if (qtype == kPerTensorAffine) {
        weight_scales_float[0] = weight.q_scale();
      } else if (qtype == kPerChannelAffine) {
        weight_scales_float.resize(N, 0.0);
        for (const auto i : irange(N)) {
          weight_scales_float[i] = weight.q_per_channel_scales()[i].item<float>();
        }
      }

      i8* weight_ptr_int8 =
          reinterpret_cast<i8*>(weight_contig.data_ptr<qint8>());

      vector<i32> col_offsets(N);
      calc_col_offsets_transpose(
          /*K=*/K,
          /*N=*/N,
          /*Bint8=*/weight_ptr_int8,
          /*B_zero_point=*/weight_zero_points_int32.data(),
          /*col_offsets=*/col_offsets.data(),
          /*qtype=*/qtype);

      optional<Tensor> bias_contig;
      if (bias.has_value()) {
        Tensor bias_vec = bias.value();
        TORCH_CHECK(bias_vec.dim() == 1, "bias should be a vector (1D Tensor)");
        TORCH_CHECK(
            bias_vec.size(0) == N,
            "bias should have N elements: " + to_string(N));
        bias_contig = bias->contiguous();
      }
      auto ret_ptr = make_intrusive<PackedLinearWeight>(
          make_unique<fbgemm::PackBMatrix<i8>>(
              /*trans=*/fbgemm::matrix_op_t::Transpose,
              /*nRow=*/K,
              /*nCol=*/N,
              /*smat=*/weight_ptr_int8,
              /*ld=*/K,
              /*pmat=*/nullptr, // PackBMatrix manages ownership of pmat
              /*groups=*/1),
          bias_contig,
          col_offsets,
          weight_scales_float,
          weight_zero_points_int32,
          qtype);
      return ret_ptr;
        */
    }
}


impl PackedLinearWeightsQnnp {
    
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
}

impl PackedLinearWeightFp16 {
    
    #[cfg(feature = "fbgemm")]
    pub fn prepack(&mut self, 
        weight: Tensor,
        bias:   Option<Tensor>) -> IntrusivePtr<LinearPackedParamsBase> {
        
        todo!();
        /*
            weight = _saturate_weight_to_fp16(weight);

      const i64 K = weight.size(1);
      const i64 N = weight.size(0);
      Tensor weight_contig = weight.contiguous();
      float* weight_contig_ptr = weight_contig.data_ptr<float>();

      // TODO(mingzhe09088):
      // Consider using a functor here in PackedGemmMatrixFP16
      // Comments from (XQ): Not entirely sure this make_unique is safe.
      // make_unique is created with regular "new", and freed through
      // TypeMetaData::deleteFn in this function. This is perfectly fine if the
      // tensors are created and freed within this translation unit. It might be
      // very problematic if that tensor flows across dll boundaries.
      auto ptr = make_intrusive<PackedLinearWeightFp16>(
          make_unique<fbgemm::PackedGemmMatrixFP16>(
              fbgemm::matrix_op_t::Transpose, K, N, 1, weight_contig_ptr),
          bias);
      return ptr;
        */
    }
}

pub fn saturate_weight_to_fp16(weight: &Tensor) -> Tensor {
    
    todo!();
        /*
            Tensor weight_contig = weight.contiguous();
      float* weight_contig_ptr = weight_contig.data_ptr<float>();
      quant_utils::HandleWeightsSaturation(weight.size(0) * weight.size(1), weight_contig_ptr);
      return weight;
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

lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
      m.impl(TORCH_SELECTIVE_NAME("quantized::linear_prepack"), TORCH_FN(QLinearPackWeightInt8::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::linear_prepack_legacy"), TORCH_FN(QLinearPackWeightInt8Legacy::run));
    }
    */
}

lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(quantized, CPU, m) {
      m.impl(TORCH_SELECTIVE_NAME("quantized::linear_prepack_fp16"), TORCH_FN(QLinearPackWeightFp16::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::linear_prepack_fp16_legacy"), TORCH_FN(QLinearPackWeightFp16Legacy::run));
    }
    */
}

lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(_quantized, QuantizedCPU, m) {
      m.impl(TORCH_SELECTIVE_NAME("_quantized::linear_prepack"), TORCH_FN(QLinearPackWeightInt8::run));
    }
    */
}

lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(_quantized, CPU, m) {
      m.impl(TORCH_SELECTIVE_NAME("_quantized::linear_prepack_fp16"), TORCH_FN(QLinearPackWeightFp16::run));
      m.impl(TORCH_SELECTIVE_NAME("_quantized::linear_prepack_fp16_legacy"), TORCH_FN(QLinearPackWeightFp16Legacy::run));
    }
    */
}
