crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/ao_sparse/quantized/cpu/qnnpack_utils.h]

#[cfg(feature = "fbgemm")]
pub struct PackedLinearWeightFp16 {
    base: LinearPackedParamsBase,
    w:    Box<FbgemmPackedGemmMatrixFP16>,
    bias: Option<Tensor>,
}

#[cfg(feature = "fbgemm")]
impl PackedLinearWeightFp16 {
    
    pub fn new(
        w:    Box<FbgemmPackedGemmMatrixFP16>,
        bias: Option<Tensor>) -> Self {
    
        todo!();
        /*


            : w(move(w)), bias_(move(bias))
        */
    }
    
    pub fn apply(&mut self, 
        input:             Tensor,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(false);
        */
    }
    
    pub fn apply_relu(&mut self, 
        input:             Tensor,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(false);
        */
    }
    
    pub fn apply_dynamic(&mut self, 
        input:        Tensor,
        reduce_range: bool) -> Tensor {
        let reduce_range: bool = reduce_range.unwrap_or(false);

        todo!();
        /*
        
        */
    }
    
    pub fn apply_dynamic_relu(&mut self, 
        input:        Tensor,
        reduce_range: bool) -> Tensor {
        let reduce_range: bool = reduce_range.unwrap_or(false);

        todo!();
        /*
        
        */
    }
    
    pub fn unpack(&mut self) -> (Tensor,Option<Tensor>) {
        
        todo!();
        /*
        
        */
    }
    
    pub fn bias(&mut self) -> Option<Tensor> {
        
        todo!();
        /*
            return bias_;
        */
    }
    
    pub fn prepack(
        weight: Tensor,
        bias:   Option<Tensor>) -> IntrusivePtr<LinearPackedParamsBase> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn set_bias(&mut self, bias: Option<Tensor>)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn apply_dynamic_impl<const ReluFused: bool>(&mut self, input: Tensor) -> Tensor {
    
        todo!();
        /*
        
        */
    }
    
    #[cfg(feature = "fbgemm")]
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
    
    #[cfg(feature = "fbgemm")]
    pub fn apply_dynamic_impl<const ReluFused: bool>(&mut self, input: Tensor) -> Tensor {
    
        todo!();
        /*
            const Tensor input_contig = input.contiguous();
      const float* input_ptr = input_contig.data_ptr<float>();

      auto& packed_weight_fp16 = *w;

      TORCH_CHECK(input.size(input.dim() - 1) == packed_weight_fp16.numRows())
      TORCH_CHECK(input.dim() >= 2);

      const i64 M = Sizeo_dim_(input.dim() - 1, input.sizes());
      const i64 N = packed_weight_fp16.numCols();
      vector<i64> output_size = input.sizes().vec();
      output_size.back() = N;
      Tensor output = empty(output_size, input.options().dtype(kFloat));

      // Call the fp16 gemm interface
      fbgemm::cblas_gemm_compute(
          fbgemm::matrix_op_t::NoTranspose,
          M,
          input_ptr,
          packed_weight_fp16,
          0.0f,
          output.data_ptr<float>());

      // Add bias term
      if (bias_.has_value()) {
        TORCH_CHECK(bias_->dim() == 1);
        output.add_(*bias_);
      }

      return output;
        */
    }
    
    pub fn apply_dynamic(&mut self, 
        input:        Tensor,
        reduce_range: bool) -> Tensor {
        
        todo!();
        /*
            return apply_dynamic_impl</*ReluFused=*/false>(move(input));
        */
    }
    
    pub fn apply_dynamic_relu(&mut self, 
        input:        Tensor,
        reduce_range: bool) -> Tensor {
        
        todo!();
        /*
            return apply_dynamic_impl</*ReluFused=*/true>(move(input));
        */
    }
    
    pub fn set_bias(&mut self, bias: Option<Tensor>)  {
        
        todo!();
        /*
            bias_ = move(bias);
        */
    }
}
