crate::ix!();

/**
  | TODO: Refacto qnnpack_utils.h so as
  | to separate code needed for quantized
  | op from the generic qnnpack specific
  | quantization utilities.
  |
  */
#[cfg(feature = "qnnpack")]
pub struct PackedLinearWeightQnnp {

    base:                  LinearPackedParamsBase,

    orig_weight:           Tensor,
    orig_bias:             Option<Tensor>,

    /**
      | Seperate copy of bias exist so that we
      | can fill in zeros when optional bias
      | does not exist. This is to compy with
      | qnnpack operator that expects bias
      | to be present.
      | 
      | In case bias is present bias_ is just
      | a reference to orig_bias_
      |
      */
    bias:                  Tensor,

    q_scheme:              QScheme,
    input_scale:           f64,
    bcsr_matrix:           Box<QnnpackBCSRMatrix>,
    w_scales:              Tensor,
    w_zero_points:         Vec<u8>,
    requantization_scales: Vec<f32>,
    sparse_linear_op:      Box<PytorchQnnpOperator,QnnpackOperatorDeleter>, // default = { nullptr }
}

#[cfg(feature = "qnnpack")]
impl PackedLinearWeightQnnp {

    pub fn new(
        weight:                  &Tensor,
        bias:                    &Option<Tensor>,

        /* block sparsity size across output_features */
        out_features_block_size: i64,

        /* block sparsity size across input_features */
        in_features_block_size:  i64) -> Self {
    
        todo!();
        /*


        
        */
    }
    
    pub fn apply(&mut self, 
        input:             &Tensor,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor {
        
        todo!();
        /*
            TORCH_CHECK(
            false, "Static quantized sparse linear unimplemented on QNNPACK");
        */
    }
    
    pub fn apply_relu(&mut self, 
        input:             &Tensor,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor {
        
        todo!();
        /*
            TORCH_CHECK(
            false, "Static quantized sparse linear unimplemented on QNNPACK");
        */
    }
    
    pub fn bias(&mut self) -> Option<Tensor> {
        
        todo!();
        /*
            return orig_bias_;
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
    
    pub fn apply_dynamic_impl<const ReluFused: bool>(&mut self, input: &Tensor) -> Tensor {
    
        todo!();
        /*
        
        */
    }
    
    pub fn apply_dynamic_impl_true(&mut self, input: &Tensor) -> Tensor {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(
          false,
          "Sparse quantized dynamic linear with fused relu is not yet "
          "supported on qnnpack backend.");
      return Tensor();
        */
    }
    
    pub fn apply_dynamic_impl_false(&mut self, input: &Tensor) -> Tensor {
        
        todo!();
        /*
            TORCH_CHECK(
          input.dim() >= 2,
          "quantized_sparse_linear(): Input tensor rank should be >= 2");

      usize rows_input = 1;
      usize cols_input = input.size(input.dim() - 1);
      for (usize i = 0; i < input.dim() - 1; ++i) {
        rows_input *= input.size(i);
      }
      TORCH_CHECK(
          cols_input == orig_weight_.size(1),
          "quantized_sparse_lienar: Input tensor's last and weight tensor's"
          " second dimension must match.");

      float x_min;
      float x_max;
      if (input.numel() > 0) {
        x_min = input.min().item<float>();
        x_max = input.max().item<float>();
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

      // Quantize input
      Tensor q_input = quantize_per_tensor(
          input, q_params.scale, q_params.zero_point, kQUInt8);

      auto q_input_contig = q_input.contiguous();
      if (sparse_linear_op_ == nullptr) {
        // We calculate requant scale here as the vector holding the requant scale
        // is owned by this module. The pointer is then passed to qnnpack backend.
        generate_requantization_scales(
            w_scales_, q_input_contig.q_scale(), 1.f, requantization_scales_);
        input_scale_ = q_input_contig.q_scale();
        pytorch_qnnp_operator_t sparse_linear_op{nullptr};
        pytorch_qnnp_status status =
            pytorch_qnnp_create_fully_connected_sparse_dq_nc_q8(
                orig_weight_.size(1),
                orig_weight_.size(0),
                q_input_contig.q_zero_point(),
                w_zero_points_.data(),
                bcsr_matrix_->col_indices.data(),
                bcsr_matrix_->row_values.data(),
                bcsr_matrix_->values.data(),
                bcsr_matrix_->row_block_size, /* out_features_block_size */
                bcsr_matrix_->col_block_size, /* in_features_block_size */
                0, /* output zero point: not used */
                u8::min,
                u8::max,
                0, /* flags */
                requantization_scales_.data(),
                true, /* use prepacking kernel */
                &sparse_linear_op);
        TORCH_CHECK(
            status == pytorch_qnnp_status_success,
            "Failed to create sparse linear operator on"
            " qnnpack backend.");
        sparse_linear_op_ =
            unique_ptr<pytorch_qnnp_operator, QnnpackOperatorDeleter>(
                sparse_linear_op);
      }

      // Input on next iteration can be different, thus resulting in
      // different input scale. This will require us to recalculate requantization
      // scales.
      if (input_scale_ != q_input_contig.q_scale()) {
        generate_requantization_scales(
            w_scales_, q_input_contig.q_scale(), 1.f, requantization_scales_);
      }
      // Update input related quantization params in the operator.
      sparse_linear_op_->dynamic_conv_quantization_params.input_zero_point =
          q_input_contig.q_zero_point();
      sparse_linear_op_->dynamic_conv_quantization_params.multipliers =
          requantization_scales_.data();

      vector<i64> out_sizes = input.sizes().vec();
      usize rows_w = orig_weight_.size(0);
      out_sizes.back() = rows_w;

      auto output = empty(out_sizes, input.options().dtype(kFloat));

      pytorch_qnnp_status status =
          pytorch_qnnp_setup_fully_connected_sparse_dq_nc_q8(
              sparse_linear_op_.get(),
              rows_input, /* batch size */
              reinterpret_cast<u8*>(q_input_contig.data_ptr<quint8>()),
              cols_input, /* num input channels */
              bias_.data_ptr<float>(),
              output.data_ptr<float>(),
              rows_w /* num output channels */);
      TORCH_CHECK(
          status == pytorch_qnnp_status_success,
          "Failed to setup sparse linear operator on"
          " qnnpack backend.");

      status = pytorch_qnnp_run_operator(
          sparse_linear_op_.get(), pthreadpool_());
      TORCH_CHECK(
          status == pytorch_qnnp_status_success,
          "Failed to run sparse linear operator on"
          " qnnpack backend.");

      return output;
        */
    }
    
    pub fn apply_dynamic(&mut self, input: &Tensor) -> Tensor {
        
        todo!();
        /*
            return apply_dynamic_impl<false>(input);
        */
    }
    
    pub fn apply_dynamic_relu(&mut self, input: &Tensor) -> Tensor {
        
        todo!();
        /*
            return apply_dynamic_impl<true>(input);
        */
    }
    
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
    
    pub fn unpack(&mut self) -> LinearPackedSerializationType {
        
        todo!();
        /*
            vector<i64> block_pattern(
          {out_features_block_size_, in_features_block_size_});
      return make_tuple(orig_weight_, orig_bias_, move(block_pattern));
        */
    }
}
