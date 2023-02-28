crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/ao_sparse/quantized/cpu/qlinear_prepack.cpp]

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

impl PackedLinearWeight {
    
    #[cfg(feature = "fbgemm")]
    pub fn prepack(&mut self, 
        weight:                  &Tensor,
        bias:                    &Option<Tensor>,
        out_features_block_size: i64,
        in_features_block_size:  i64) -> IntrusivePtr<LinearPackedParamsBase> {
        
        todo!();
        /*
            TORCH_CHECK(
          weight.dim() == 2,
          "The weight tensor for ao::sparse::qlinear_prepack (fbgemm) should"
          " be 2-dimensional.");

      auto N = weight.size(0);
      auto K = weight.size(1);

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
      TORCH_CHECK(
          all_of(
              weight_zero_points_int32.cbegin(),
              weight_zero_points_int32.cend(),
              [](i32 i) { return i == 0; }),
          "zero point(s) should be 0 for the weight tensor of ao::sparse::qlinear op");
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
        const Tensor& bias_vec = bias.value();
        TORCH_CHECK(bias_vec.dim() == 1, "bias should be a vector (1D Tensor)");
        TORCH_CHECK(
            bias_vec.size(0) == N,
            "bias should have N elements: " + to_string(N));
        bias_contig = bias->contiguous();
      }

      auto bcsr = fbgemm::fbgemmDenseToBCSR<i8>(N, K, weight_ptr_int8);
      auto ret_ptr = make_intrusive<PackedLinearWeight>(
          move(bcsr),
          bias_contig,
          col_offsets,
          weight_scales_float,
          weight_zero_points_int32,
          qtype,
          out_features_block_size,
          in_features_block_size);
      return ret_ptr;
        */
    }
}

impl PackedLinearWeightQnnp {
    
    #[cfg(USE_PYTORCH_QNNPACK)]
    pub fn prepack(&mut self, 
        weight:                  &Tensor,
        bias:                    &Option<Tensor>,
        out_features_block_size: i64,
        in_features_block_size:  i64) -> IntrusivePtr<LinearPackedParamsBase> {
        
        todo!();
        /*
            native::initQNNPACK();
      return make_intrusive<PackedLinearWeightQnnp>(
          weight, bias, out_features_block_size, in_features_block_size);
        */
    }

    #[cfg(USE_PYTORCH_QNNPACK)]
    pub fn new(
        weight:                  &Tensor,
        bias:                    &Option<Tensor>,
        out_features_block_size: i64,
        in_features_block_size:  i64) -> Self {
    
        todo!();
        /*
        : linear_packed_params_base(out_features_block_size,
                  in_features_block_size),
        : orig_weight(weight),
        : orig_bias(bias),

            TORCH_CHECK(
          weight.dim() == 2,
          "ao::sparse::qlinear (qnnpack): Weight tensor rank should be == 2");
      TORCH_CHECK(out_features_block_size > 0, "Row block size must be > 0.");
      TORCH_CHECK(in_features_block_size > 0, "Row block size must be > 0.");

      i64 rows_w = weight.size(0);
      if (bias.has_value()) {
        bias_ = bias.value();
      } else {
        bias_ = zeros(rows_w, weight.options().dtype(kFloat));
      }
      TORCH_CHECK(
          (bias_.ndimension() == 1 && bias_.size(0) == rows_w),
          "ao::sparse::qlinear_prepack (qnnpack): Given weight of size ",
          weight.sizes(),
          ", expected bias to be 1-dimensional with ",
          rows_w,
          " elements",
          ", but got bias of size ",
          bias_.sizes(),
          " instead");

      // Given bias is supposed to be 1 dim, it is already contiguous,
      // but the weight might be non-contiguous.
      Tensor weight_contig = orig_weight_.contiguous();

      q_scheme_ = orig_weight_.qscheme();
      tie(w_zero_points_, w_scales_) =
          make_zero_points_and_scales_tensor(weight_contig);
      const float* weight_scales_data = w_scales_.data_ptr<float>();
      Tensor qnnp_weight = _empty_affine_quantized(
          weight_contig.sizes(),
          device(kCPU).dtype(kQUInt8),
          weight_scales_data[0],
          w_zero_points_[0]);
      auto* qnnp_w_data = qnnp_weight.data_ptr<quint8>();
      auto wt_numel = weight_contig.numel();
      i8* w_data =
          reinterpret_cast<i8*>(weight_contig.data_ptr<qint8>());
      for (const auto i : irange(wt_numel)) {
        qnnp_w_data[i] = static_cast<quint8>(w_data[i] + 128);
      }
      bcsr_matrix_ = qnnpack::generateBlockCSRMatrix(
          reinterpret_cast<u8*>(qnnp_w_data),
          orig_weight_.size(0), /* output_channels */
          orig_weight_.size(1), /* input_channels */
          out_features_block_size,
          in_features_block_size,
          w_zero_points_.data());
        */
    }
}

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
