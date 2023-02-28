// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/QuantizedLinear.cpp]

caffe_known_type!(intrusive_ptr<LinearPackedParamsBase>);

// Required for cpp_custom_type_hack to work
#[cfg(feature = "fbgemm")]
caffe_known_type!(fbgemm::PackBMatrix<i8>);

#[cfg(feature = "fbgemm")]
caffe_known_type!(intrusive_ptr<PackedLinearWeightFp16>);

/// one be ten thousand if he be the best
///
#[cfg(feature = "fbgemm")]
pub fn fbgemm_linear_int8_weight_fp32_activation(
    input:             &Tensor,
    weight:            &Tensor,
    packed:            &Tensor,
    col_offsets:       &Tensor,
    weight_scale:      &Scalar,
    weight_zero_point: &Scalar,
    bias:              &Tensor) -> Tensor {

    todo!();
    /*
            // We make a strong guarantee that models using these operators will have the
      // same numerics across different machines. Therefore, we do not provide a
      // fallback path and rather fail loudly if we cannot run FBGEMM.
      TORCH_CHECK(fbgemm::fbgemmSupportedCPU(), "Your CPU doesn't support FBGEMM.");

      const Tensor input_contig = input.contiguous();
      const float* input_ptr = input_contig.data_ptr<float>();

      TORCH_CHECK(input.dim() >= 2);
      const i64 M = size_to_dim_(input.dim() - 1, input.sizes());
      const i64 K = input.size(input.dim() - 1);
      TORCH_CHECK(weight.dim() == 2);
      TORCH_CHECK(K == weight.size(1));
      const i64 N = weight.size(0);
      TORCH_CHECK(bias.dim() == 1);
      TORCH_CHECK(bias.size(0) == N);
      TORCH_CHECK(weight_scale.isFloatingPoint());
      TORCH_CHECK(weight_zero_point.isIntegral(false));

      // Calculate statistics for quantization of the input Tensor
      float x_min;
      float x_max;
      fbgemm::FindMinMax(
          /*m=*/input_ptr,
          /*min=*/&x_min,
          /*max=*/&x_max,
          /*len=*/input.numel());

      // Input tensor is quantized as 8-bit unsigned values
      constexpr int kPrecision = 8;
      constexpr bool kIsSigned = false;
      constexpr int kBound = (1 << (kPrecision - 1));

      // Calculate scale and zero point for quantization of input tensor
      auto q_params = fbgemm::ChooseQuantizationParams(
          /*min=*/x_min,
          /*max=*/x_max,
          /*qmin=*/kIsSigned ? -kBound : 0,
          /*qmax=*/kIsSigned ? (kBound - 1) : (1 << kPrecision) - 1,
          /*preserve_sparsity=*/false);
      q_params.precision = kPrecision;

      // ReQuantizeForFloat requires pointers to the scale and zero point values,
      // since in the case of rowwise quantization these will be arrays rather than
      // scalars. But in this case, we're doing whole-tensor quantization so we just
      // pass a pointer to the scale values (and internally ReQuantizeFor Float
      // won't index past 0
      const float weight_scale_float =
          static_cast<float>(weight_scale.to<double>());
      const i32 weight_zero_point_int32 =
          static_cast<i32>(weight_zero_point.to<i64>());

      const Tensor bias_contig = bias.contiguous();

      // Allocate output Tensor and a buffer for fbgemmPacked to use
      vector<i64> output_size = input.sizes().vec();
      output_size.back() = N;
      Tensor output = empty(output_size, input.options().dtype(kFloat), LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      Tensor buffer = empty(output_size, input.options().dtype(kInt), LEGACY_CONTIGUOUS_MEMORY_FORMAT);

      // Pull out the PackBMatrix instance from the owning tensor
      auto& pack_b =
          cpp_custom_type_hack::cast<fbgemm::PackBMatrix<i8>>(packed);

      const int num_tasks = get_num_threads();
      parallel_for(0, num_tasks, 1, [&](i64 begin, i64 end) {
        // This operation does the following:
        // 1) Quantizes the input matrix given the statistics we've calculated
        //    above.
        // 2) Creates a "row buffer" vector with offset values that must be added
        //    to the integer matrix multiplication operation to ensure correctness.
        // 3) Packs the resulting quantized matrix into vector-register and cache
        //    friendly tiles.
        //
        //  Note this is not executed eagerly, but rather within the fbgemmPacked
        //  call below.
        fbgemm::PackAWithQuantRowOffset<u8> pack_a(
            /*trans=*/fbgemm::matrix_op_t::NoTranspose,
            /*nRow=*/M,
            /*nCol=*/K,
            /*smat=*/input_ptr,
            /*ld=*/K,
            /*pmat=*/nullptr, // pack_a manages ownership of `pmat`
            /*scale=*/q_params.scale,
            /*zero_pt=*/q_params.zero_point);

        // This is the end of the pipeline, pass the resulting matrix through
        fbgemm::DoNothing<float, float> kDoNothingObj{};
        for (const auto task_id : irange(begin, end)) {
          // After the uint8 * int8 matrix multiplication is performed, this
          // operation does:
          //  1) Add in row and column offsets to the rows and columns, respectively
          //  2) Dequantize the results into floating point
          //  3) Add in the bias term
          fbgemm::ReQuantizeForFloat</* FUSE_RELU */ false> output_proc_obj(
              /*nextop=*/kDoNothingObj,
              /*Aq_scale=*/q_params.scale,
              /*Bq_scale=*/&weight_scale_float,
              /*Aq_zero_point=*/q_params.zero_point,
              /*Bq_zero_point=*/&weight_zero_point_int32,
              /*row_offsets=*/pack_a.getRowOffsetBuffer(),
              /*col_offsets=*/col_offsets.data_ptr<i32>(),
              /*bias=*/bias_contig.data_ptr<float>(),
              /*nCol=*/N);
          // Do the GEMM
          fbgemm::fbgemmPacked(
              /*packA=*/pack_a,
              /*packB=*/pack_b,
              /*C=*/output.data_ptr<float>(),
              /*C_buffer=*/buffer.data_ptr<i32>(),
              /*ldc=*/N,
              /*outProcess=*/output_proc_obj,
              /*thread_id=*/task_id,
              /*num_threads=*/num_tasks);
        }
      });

      return output;
        */
}

#[cfg(feature = "fbgemm")]
pub fn fbgemm_linear_int8_weight(
    input:             &Tensor,
    weight:            &Tensor,
    packed:            &Tensor,
    col_offsets:       &Tensor,
    weight_scale:      &Scalar,
    weight_zero_point: &Scalar,
    bias:              &Tensor) -> Tensor {

    todo!();
        /*
            // Replace after https://github.com/pytorch/pytorch/issues/24354 is fixed
      // TORCH_WARN(
      //     "fbgemm_linear_int8_weight will be deprecated soon."
      //     "Please use fbgemm_linear_int8_weight_fp32_activation instead.");

      return native::fbgemm_linear_int8_weight_fp32_activation(
          input,
          weight,
          packed,
          col_offsets,
          weight_scale,
          weight_zero_point,
          bias);
        */
}

/**
  | Calculate the column offsets
  |
  | Note this includes the sum of the columns as
  | well as the scalar term B_zero_point * K,
  | whereas the row_offsets created by
  | PackAWithQuantRowOffset is only the sum of the
  | A rows.
  */
#[cfg(feature = "fbgemm")]
pub fn calc_col_offsets_transpose(
    K:            i32,
    N:            i32,
    bint8:        *const i8,
    b_zero_point: i32,
    col_offsets:  *mut i32)  {

    todo!();
        /*
            for (int i = 0; i < N; ++i) {
        i32 sum = 0;
        for (int j = 0; j < K; ++j) {
          sum += Bint8[i * K + j];
        }
        col_offsets[i] = sum - B_zero_point * K;
      }
        */
}

#[cfg(feature = "fbgemm")]
pub fn fbgemm_linear_quantize_weight(weight: &Tensor) -> (Tensor,Tensor,f64,i64) {
    
    todo!();
        /*
            // We make a strong guarantee that models using these operators will have the
      // same numerics across different machines. Therefore, we do not provide a
      // fallback path and rather fail loudly if we cannot run FBGEMM.
      TORCH_CHECK(fbgemm::fbgemmSupportedCPU(), "Your CPU doesn't support FBGEMM.");
      const Tensor weight_contig = weight.contiguous();

      // Calculate weight statistics
      float w_min;
      float w_max;
      fbgemm::FindMinMax(
          /*m=*/weight_contig.data_ptr<float>(),
          /*min=*/&w_min,
          /*max=*/&w_max,
          /*len=*/weight_contig.numel());

      // Choose parameters for quantizing the weight as 8-bit signed integer
      constexpr bool kIsSigned = true;
      constexpr int kPrecision = 8;
      constexpr int kBound = (1 << (kPrecision - 1));
      auto q_params = fbgemm::ChooseQuantizationParams(
          /*min=*/w_min,
          /*max=*/w_max,
          /*qmin=*/kIsSigned ? -kBound : 0,
          /*qmax=*/kIsSigned ? (kBound - 1) : (1 << kPrecision) - 1,
          /*preserve_sparsity=*/false);
      q_params.precision = kPrecision;

      Tensor quantized = native::empty_like(
          weight_contig,
          kChar,
          weight_contig.options().layout_opt(),
          weight_contig.options().device_opt(),
          weight_contig.options().pinned_memory_opt(),
          LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      // Tensor quantized = native::empty_cpu(
      //     weight_contig.sizes(), weight_contig.options().dtype(kChar));
      fbgemm::Quantize<i8, false /*LEGACY*/>(
          /*src=*/weight_contig.data_ptr<float>(),
          /*dst=*/quantized.data_ptr<i8>(),
          /*len=*/weight_contig.numel(),
          /*qparams=*/q_params);

      // Calculate column offsets of the weight and store them away in a tensor.
      // Similarly to quantization, this can be done once and cached.
      Tensor col_offsets = empty(
          {weight_contig.size(0)},
          kInt,
          weight_contig.options().layout_opt(),
          weight_contig.options().device_opt(),
          weight_contig.options().pinned_memory_opt(),
          LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      CalcColOffsetsTranspose(
          /*K=*/quantized.size(1),
          /*N=*/quantized.size(0),
          /*Bint8=*/quantized.data_ptr<i8>(),
          /*B_zero_point=*/q_params.zero_point,
          /*col_offsets=*/col_offsets.data_ptr<i32>());

      return make_tuple(
          quantized, col_offsets, q_params.scale, q_params.zero_point);
        */
}

#[cfg(feature = "fbgemm")]
pub fn fbgemm_pack_quantized_matrix(weight: &Tensor) -> Tensor {
    
    todo!();
        /*
            // We make a strong guarantee that models using these operators will have the
      // same numerics across different machines. Therefore, we do not provide a
      // fallback path and rather fail loudly if we cannot run FBGEMM.
      TORCH_CHECK(fbgemm::fbgemmSupportedCPU(), "Your CPU doesn't support FBGEMM.");
      const i64 K = weight.size(1);
      const i64 N = weight.size(0);
      const Tensor weight_contig = weight.contiguous();
      const i8* weight_ptr = weight_contig.data_ptr<i8>();
      auto ptr = make_unique<fbgemm::PackBMatrix<i8>>(
          /*trans=*/fbgemm::matrix_op_t::Transpose,
          /*nRow=*/K,
          /*nCol=*/N,
          /*smat=*/weight_ptr,
          /*ld=*/K,
          /*pmat=*/nullptr, // PackBMatrix manages ownership of pmat
          /*groups=*/1);
      return cpp_custom_type_hack::create(move(ptr), weight.options());
        */
}

#[cfg(feature = "fbgemm")]
pub fn fbgemm_pack_quantized_matrix(
    weight: &Tensor,
    K:      i64,
    N:      i64) -> Tensor {

    todo!();
        /*
            // Replace after https://github.com/pytorch/pytorch/issues/24354 is fixed
      // TORCH_WARN(
      //     "fbgemm_pack_quantized_matrix(weight, K, N) will be deprecated soon."
      //     "Please use fbgemm_pack_quantized_matrix(weight) instead.");
      return native::fbgemm_pack_quantized_matrix(weight);
        */
}

#[cfg(feature = "fbgemm")]
pub fn raw_uint_16to_fp16(value: u16) -> f32 {
    
    todo!();
        /*
            // Convert raw 16 bits half precision floating point number
      // to single precision floating point number.
      const unsigned short sign_bits = value >> 15;
      const unsigned short exponent_bits = value >> 10 & 0x1f;
      const unsigned short significand_bits = value & 0x3ff;

      const float sign = sign_bits ? -1 : 1;
      const float significand =
          1 + significand_bits * 0.0009765625f; // 0.0009765625f = 0x1p-10 = 2^-10
      const float exponent = exponent_bits - 0xf;

      return sign * ldexp(significand, exponent);
        */
}


#[cfg(feature = "fbgemm")]
pub fn check_and_saturate<T>(
    max_val: T,
    element: *mut T) -> bool {

    todo!();
        /*
            if (*element > max_val) {
        *element = max_val;
        return true;
      }
      if (*element < -max_val) {
        *element = -max_val;
        return true;
      }
      return false;
        */
}

/**
  | The range for using FP16 quantization of
  | weights requires that the elements should be in
  | the range of [5.96e-8, 65504]. If it is out of
  | range, then the number will be saturated to max
  | or min representable values by FP16.
  |
  */
#[cfg(feature = "fbgemm")]
pub fn handle_weights_saturation(
    N:      i64,
    weight: *mut f32)  {
    
    todo!();
        /*
            const float kFp16Max = RawUint16ToFp16(0x7BFF);
      bool found_out_of_range = false;
      for (i64 i = 0; i < N; ++i) {
        if (CheckAndSaturate<float>(kFp16Max, weight + i)) {
          found_out_of_range = true;
        }
      }
      if (found_out_of_range) {
        TORCH_WARN("FOUND weight out of range ");
      }
        */
}

#[cfg(feature = "fbgemm")]
pub fn fbgemm_pack_gemm_matrix_fp16(weight: &Tensor) -> Tensor {
    
    todo!();
        /*
            // We make a strong guarantee that models using these operators will have the
      // same numerics across different machines. Therefore, we do not provide a
      // fallback path and rather fail loudly if we cannot run FBGEMM.
      TORCH_CHECK(fbgemm::fbgemmSupportedCPU(), "Your CPU doesn't support FBGEMM.");

      const i64 K = weight.size(1);
      const i64 N = weight.size(0);
      Tensor weight_contig = weight.contiguous();
      float* weight_contig_ptr = weight_contig.data_ptr<float>();
      HandleWeightsSaturation(K * N, weight_contig_ptr);

      // TODO(mingzhe09088):
      // Consider using a functor here in PackedGemmMatrixFP16
      // Comments from (XQ): Not entirely sure this make_unique is safe. make_unique
      // is created with regular "new", and freed through TypeMetaData::deleteFn in
      // this function. This is perfectly fine if the tensors are created and freed
      // within this translation unit. It might be very problematic if that tensor
      // flows across dll boundaries.
      auto ptr = make_unique<fbgemm::PackedGemmMatrixFP16>(
          fbgemm::matrix_op_t::Transpose, K, N, 1, weight_contig_ptr);
      intrusive_ptr<LinearPackedParamsBase> packed_weight =
          make_intrusive<PackedLinearWeightFp16>(move(ptr), nullopt);
      auto unique_ptr_wrapper =
          make_unique<decltype(packed_weight)>(move(packed_weight));
      return cpp_custom_type_hack::create(
          move(unique_ptr_wrapper), weight.options());
        */
}

#[cfg(feature = "fbgemm")]
pub fn fbgemm_linear_fp16_weight_fp32_activation(
    input:         &Tensor,
    packed_weight: &Tensor,
    bias:          &Tensor) -> Tensor {
    
    todo!();
        /*
            // We make a strong guarantee that models using these operators will have the
      // same numerics across different machines. Therefore, we do not provide a
      // fallback path and rather fail loudly if we cannot run FBGEMM.
      TORCH_CHECK(fbgemm::fbgemmSupportedCPU(), "Your CPU doesn't support FBGEMM.");

      const Tensor input_contig = input.contiguous();
      const float* input_ptr = input_contig.data_ptr<float>();

      // Pull out the PackedGemmMatrixFP16 instance from the owning tensor
      const fbgemm::PackedGemmMatrixFP16& packed_weight_fp16 =
          *dynamic_intrusive_pointer_cast<PackedLinearWeightFp16>(
               cpp_custom_type_hack::cast<
                   intrusive_ptr<LinearPackedParamsBase>>(packed_weight))
               ->w;

      TORCH_CHECK(input.size(input.dim() - 1) == packed_weight_fp16.numRows())
      TORCH_CHECK(input.dim() >= 2);
      TORCH_CHECK(bias.dim() == 1);

      const i64 M = size_to_dim_(input.dim() - 1, input.sizes());
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
      output.add_(bias);

      return output;
        */
}

#[cfg(feature = "fbgemm")]
pub fn fbgemm_linear_fp16_weight(
    input:         &Tensor,
    packed_weight: &Tensor,
    bias:          &Tensor) -> Tensor {
    
    todo!();
        /*
            // Replace after https://github.com/pytorch/pytorch/issues/24354 is fixed
      // TORCH_WARN(
      //     "fbgemm_linear_fp16_weight will be deprecated soon."
      //     "Please use fbgemm_linear_fp16_weight_fp32_activation instead.");
      return native::fbgemm_linear_fp16_weight_fp32_activation(
          input, packed_weight, bias);
        */
}


#[cfg(not(feature = "fbgemm"))]
pub fn fbgemm_is_cpu_supported() -> bool {
    
    todo!();
        /*
            return false;
        */
}
