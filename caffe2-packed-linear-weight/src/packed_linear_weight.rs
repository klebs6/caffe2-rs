crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/ao_sparse/quantized/cpu/fbgemm_utils.h]
//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/fbgemm_utils.h]

/**
  | The struct for the packed weight matrix
  | (PackBMatrix) and the corresponding column
  | offsets used for the fully connect layer, which
  | are both prepared in the prepacking step to
  | save the computations in the inference.
  |
  | Note the column offsets include the sum of the
  | B columns as well as the scalar term
  | B_zero_point * K, whereas the row offsets
  | created by
  | PackAWithQuantRowOffset/PackAWithIm2Col/PackAWithRowOffset
  | are only the sum of the A rows. The column
  | offsets are needed for the asymmetric
  | quantization (affine quantization) of input
  | matrix.
  |
  | Note that in JIT mode we can think of a way to
  | fuse col_offsets with bias.
  */
#[cfg(feature = "fbgemm")]
pub struct PackedLinearWeight {
    base:        LinearPackedParamsBase,

    //TODO: where are these boxed types?
    w:           Box<FbgemmPackBMatrix<i8>>,
    //w:           Box<FbgemmBCSRMatrix<i8>>,

    bias:        Option<Tensor>,
    col_offsets: Vec<i32>,
    w_scale:     Vec<f32>,
    w_zp:        Vec<i32>,
    q_scheme:    QScheme,
}

#[cfg(feature = "fbgemm")]
impl PackedLinearWeight {

    pub fn new(
        w:                       Box<FbgemmBCSRMatrix<i8>>,
        bias:                    Option<Tensor>,
        col_offsets:             Vec<i32>,
        w_scale:                 Vec<f32>,
        w_zp:                    Vec<i32>,
        q_scheme:                QScheme,

        /** block sparsity size across output_features */
        out_features_block_size: i64,

        /** block sparsity size across input_features */
        in_features_block_size:  i64) -> Self {
    
        todo!();
        /*
        : linear_packed_params_base(out_features_block_size,
                    in_features_block_size),
        : w(move(w)),
        : bias(move(bias)),
        : col_offsets(move(col_offsets)),
        : w_scale(move(w_scale)),
        : w_zp(move(w_zp)),
        : q_scheme(q_scheme),

        
        */
    }
    
    #[cfg(feature = "fbgemm")]
    pub fn apply_dynamic(&mut self, input: &Tensor) -> Tensor {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(
            false,
            "Sparse quantized dynamic linear with fused relu is not yet "
            "supported on qnnpack backend.");
        return Tensor();
        */
    }
    
    #[cfg(feature = "fbgemm")]
    pub fn apply_dynamic_relu(&mut self, input: &Tensor) -> Tensor {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(
            false,
            "Sparse quantized dynamic linear with fused relu is not yet "
            "supported on qnnpack backend.");
        return Tensor();
        */
    }
    
    #[cfg(feature = "fbgemm")]
    pub fn bias(&mut self) -> Option<Tensor> {
        
        todo!();
        /*
            return bias_;
        */
    }
    
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
    
    #[cfg(feature = "fbgemm")]
    pub fn unpack(&mut self) -> LinearPackedSerializationType {
        
        todo!();
        /*
            auto packW = w.get();

      i64 N = static_cast<i64>(packW->R);
      i64 K = static_cast<i64>(packW->C);

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

      // TODO: uncomment once unpack is implemented for BCSRMatrix
      // i8* weight_ptr_int8 =
      //     reinterpret_cast<i8*>(weight_origin.data_ptr<qint8>());
      // packW->unpack(weight_ptr_int8);
      vector<i64> block_pattern(
          {out_features_block_size_, in_features_block_size_});

      return make_tuple(weight_origin, bias_, move(block_pattern));
        */
    }
    
    #[cfg(feature = "fbgemm")]
    pub fn apply_impl<const ReluFused: bool>(&mut self, 
        input:             &Tensor,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor {
    
        todo!();
        /*
            // uint8 * int8 -> uint8 (no quantization/dequantization)

      // We make a strong guarantee that models using these operators will have
      // the same numerics across different machines. Therefore, we do not provide
      // a fallback path and rather fail loudly if we cannot run FBGEMM.
      TORCH_CHECK(
          fbgemm::fbgemmSupportedCPU(), "Your CPU does not support FBGEMM.");

      // TODO: contiguous is called for further jit optimizations.
      auto input_contig = input.contiguous();
      const auto* input_ptr =
          reinterpret_cast<u8*>(input_contig.data_ptr<quint8>());

      TORCH_CHECK(
          input.dim() >= 2,
          "The dimension of input tensor should be larger than or equal to 2");
      i64 batch_size = size_to_dim_(input.dim() - 1, input.sizes());

      auto packW = w.get();

      i64 out_channels = static_cast<i64>(packW->R);
      i64 K = input.size(input.dim() - 1);
      TORCH_CHECK(
          K == static_cast<i64>(packW->C),
          "The number of columns in the packW should be equal to K: " +
              to_string(K));

      float input_scale_float = input.q_scale();
      i32 input_zero_point_int32 = input.q_zero_point();

      vector<float> output_multiplier_float(1, 0.0);
      vector<float> act_times_w_scale(1, 0.0);
      TORCH_CHECK(
          w_scale.size() == w_zp.size(),
          "Weight scales and zero points vectors should have the same size.");
      if (q_scheme == kPerTensorAffine) {
        // Process the per tensor quantization.
        act_times_w_scale[0] = (input_scale_float * w_scale[0]);
        output_multiplier_float[0] =
            act_times_w_scale[0] / static_cast<float>(output_scale);
      } else if (q_scheme == kPerChannelAffine) {
        // Process the per channel quantization.
        output_multiplier_float.resize(out_channels, 0.0);
        act_times_w_scale.resize(out_channels, 1.0f);
        for (int i = 0; i < out_channels; ++i) {
          act_times_w_scale[i] = (input_scale_float * w_scale[i]);
          output_multiplier_float[i] =
              act_times_w_scale[i] / static_cast<float>(output_scale);
        }
      }
      i32 output_zero_point_int32 = static_cast<i32>(output_zero_point);

      const float* bias_ptr = nullptr;
      Tensor bias;
      if (this->bias_.has_value()) {
        bias = this->bias_.value();
        bias = bias.contiguous();
        TORCH_CHECK(bias.dim() == 1, "bias should be a vector (1D Tensor)");
        TORCH_CHECK(
            bias.size(0) == out_channels,
            "bias should have out_channels elements: " +
                to_string(out_channels));
        bias_ptr = reinterpret_cast<float*>(bias.data_ptr<float>());
      }

      // The resulting matrix here is 2-D, let's view it with the original
      // left hand dimensions of the input. Here are two examples:
      // 1. If the input tensor is {batch_size, K}, the output tensor is
      // {batch_size, out_channels}.
      // 2. If the input tensor is {x, batch_size, K}, the output tensor is {x,
      // batch_size, out_channels}.
      vector<i64> out_sizes = input.sizes().vec();
      out_sizes.back() = out_channels; // NOLINT
      // Allocate output Tensor and a buffer for fbgemmPacked to use
      auto output_tr = _empty_affine_quantized(
          out_sizes,
          device(kCPU).dtype(kQUInt8),
          output_scale,
          output_zero_point);
      auto output = _empty_affine_quantized(
          out_sizes,
          device(kCPU).dtype(kQUInt8),
          output_scale,
          output_zero_point);

      auto buffer = empty(out_sizes, output.options().dtype(kInt));

      // fbgemm kernel computes the following:
      // C(output) = A(weight) x B(input), where C, A, B are out_channels x
      // batch_size, out_channels x K, K x batch_size matrices, respectively.
      // Therefore we need to transpose input
      auto input_tr = _empty_affine_quantized(
          input.sizes(),
          device(kCPU).dtype(kQUInt8),
          input_scale_float,
          input_zero_point_int32);

      auto* input_tr_ptr =
          reinterpret_cast<u8*>(input_tr.data_ptr<quint8>());
      // TODO: Activation transpose before and after the kernel can be removed if we
      // keep activation tensor always tranposed.
      fbgemm::transpose_simd<u8>(
          batch_size, K, input_ptr, K, input_tr_ptr, batch_size);

      int num_tasks = get_num_threads();
      parallel_for(0, num_tasks, 1, [&](i64 begin, i64 end) {
        for (int task_id = begin; task_id < end; ++task_id) {
          fbgemm::trRequantizationParams_t reqParams = {
              input_zero_point_int32,
              w_zp.data(),
              output_zero_point_int32,
              static_cast<float>(output_scale),
              col_offsets.data(),
              /*activation offsets*/ nullptr,
              bias_ptr,
              act_times_w_scale.data()};

          if (q_scheme == kPerTensorAffine) {
            // Process the per tensor quantization.
            //
            // After the uint8 * int8 matrix multiplication is performed, this
            // operation does:
            //  1) Add in row and column offsets to the rows and columns,
            //  respectively.
            //  2) Add in the bias term.

            // Do the GEMM
            fbgemm::fbgemmSparseDenseInt8MM<
                ReluFused,
                fbgemm::QuantizationGranularity::TENSOR>(
                batch_size,
                w,
                input_tr_ptr,
                /*ldb=*/batch_size,
                /*C_i32=*/buffer.data_ptr<i32>(),
                /*C_u8=*/
                reinterpret_cast<u8*>(output_tr.data_ptr<quint8>()),
                /*ldc=*/batch_size,
                /*rParams=*/reqParams,
                /*accum=*/false,
                /*thread_id=*/task_id,
                /*num_threads=*/num_tasks);
          } else if (q_scheme == kPerChannelAffine) {
            // Process the per channel quantization.
            //
            // After the uint8 * int8 matrix multiplication is performed, this
            // operation does:
            //  1) Add in row and column offsets to the rows and columns,
            //  respectively.
            //  2) Add in the bias term.

            // Do the GEMM
            fbgemm::fbgemmSparseDenseInt8MM<
                ReluFused,
                fbgemm::QuantizationGranularity::OUT_CHANNEL>(
                batch_size,
                w,
                input_tr_ptr,
                /*ldb=*/batch_size,
                /*C_i32=*/buffer.data_ptr<i32>(),
                /*C_u8=*/
                reinterpret_cast<u8*>(output_tr.data_ptr<quint8>()),
                /*ldc=*/batch_size,
                /*rParams=*/reqParams,
                /*accum*/ false,
                /*thread_id=*/task_id,
                /*num_threads=*/num_tasks);
          }
        }
      });

      // transpose output_tr back to batch_size x out_channels
      fbgemm::transpose_simd<u8>(
          out_channels,
          batch_size,
          reinterpret_cast<u8*>(output_tr.data_ptr<quint8>()),
          batch_size,
          reinterpret_cast<u8*>(output.data_ptr<quint8>()),
          out_channels);

      return output;
        */
    }
    
    #[cfg(feature = "fbgemm")]
    pub fn apply(&mut self, 
        input:             &Tensor,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor {
        
        todo!();
        /*
            return apply_impl<false>(input, output_scale, output_zero_point);
        */
    }
    
    #[cfg(feature = "fbgemm")]
    pub fn apply_relu(&mut self, 
        input:             &Tensor,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor {
        
        todo!();
        /*
            return apply_impl<true>(input, output_scale, output_zero_point);
        */
    }
    
    #[cfg(feature = "fbgemm")]
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
    
    #[cfg(feature = "fbgemm")]
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
    
    #[cfg(feature = "fbgemm")]
    pub fn apply_impl<'a,const ReluFused: bool>(&mut self, 
        input:             &Tensor,
        output_scale:      f64,
        output_zero_point: i64,
        output:            &mut Tensor) -> &'a mut Tensor {
    
        todo!();
        /*
            // uint8 * int8 -> uint8 (no quantization/dequantization)

      // We make a strong guarantee that models using these operators will have
      // the same numerics across different machines. Therefore, we do not provide
      // a fallback path and rather fail loudly if we cannot run FBGEMM.
      TORCH_CHECK(
          fbgemm::fbgemmSupportedCPU(), "Your CPU does not support FBGEMM.");

      // TODO: contiguous is called for further jit optimizations.
      auto input_contig = input.expect_contiguous();
      const auto* input_ptr =
          reinterpret_cast<u8*>(input_contig->data_ptr<quint8>());

      TORCH_CHECK(
          input.dim() >= 2,
          "The dimension of input tensor should be larger than or equal to 2");
      // C(output) = A(input) x B(weight), where C, A, B are M x N, M x K, K x N
      // matrices, respectively.
      i64 M = Sizeo_dim_(input.dim() - 1, input.sizes());

      auto packB = w.get();

      i64 N = static_cast<i64>(packB->numCols());
      i64 K = input.sizes()[input.dim() - 1];
      TORCH_CHECK(
          K == static_cast<i64>(packB->numRows()),
          "The number of rows in the packB should be equal to K: " +
              to_string(K));

      float input_scale_float = input.q_scale();
      i32 input_zero_point_int32 = input.q_zero_point();

      vector<float> output_multiplier_float(1, 0.0);
      vector<float> act_times_w_scale(1, 0.0);
      TORCH_CHECK(
          w_scale.size() == w_zp.size(),
          "Weight scales and zero points vectors should have the same size.");
      if (q_scheme == kPerTensorAffine) {
        // Process the per tensor quantization.
        act_times_w_scale[0] = (input_scale_float * w_scale[0]);
        output_multiplier_float[0] =
            act_times_w_scale[0] / static_cast<float>(output_scale);
      } else if (q_scheme == kPerChannelAffine) {
        // Process the per channel quantization.
        output_multiplier_float.resize(N, 0.0);
        act_times_w_scale.resize(N, 1.0f);
        for (const auto i : irange(N)) {
          act_times_w_scale[i] = (input_scale_float * w_scale[i]);
          output_multiplier_float[i] =
              act_times_w_scale[i] / static_cast<float>(output_scale);
        }
      }
      i32 output_zero_point_int32 = static_cast<i32>(output_zero_point);

      const float* bias_ptr = nullptr;
      MaybeOwned<Tensor> bias_contig;
      if (this->bias_.has_value()) {
        auto& bias = this->bias_.value();
        bias_contig = bias.expect_contiguous();
        TORCH_CHECK(bias_contig->dim() == 1, "bias should be a vector (1D Tensor)");
        TORCH_CHECK(
            bias_contig->sizes()[0] == N, "bias should have N elements: " + to_string(N));
        bias_ptr = reinterpret_cast<float*>(bias_contig->data_ptr<float>());
      }

      // The resulting matrix here is 2-D, let's view it with the original
      // left hand dimensions of the input. Here are two examples:
      // 1. If the input tensor is {M, K}, the output tensor is {M, N}.
      // 2. If the input tensor is {b, M, K}, the output tensor is {b, M, N}.
      DimVector out_sizes(input.sizes());
      out_sizes.back() = N;
      // Resize output Tensor
      output.resize_(out_sizes);

      // Allocate a buffer for fbgemmPacked to use
      auto buffer = empty(out_sizes, output.options().dtype(kInt));

      int num_tasks = get_num_threads();
      parallel_for(0, num_tasks, 1, [&](i64 begin, i64 end) {
        for (const auto task_id : irange(begin, end)) {
          // This operation does the following:
          // 1) Creates a "row buffer" vector with offset values that must be
          //    added to the integer matrix multiplication operation to ensure
          //    correctness. This "row buffer" is also called the row offset, and
          //    it is needed when we use affine quantization for weights.
          // 2) Packs the resulting quantized matrix into vector-register and
          //    cache friendly tiles.
          //
          //  Note this is not executed eagerly, but rather within the
          //  fbgemmPacked call below.
          fbgemm::PackAWithRowOffset<u8> packA(
              /*trans=*/fbgemm::matrix_op_t::NoTranspose,
              /*nRow=*/M,
              /*nCol=*/K,
              /*smat=*/input_ptr,
              /*ld=*/K,
              /*pmat=*/nullptr); // Currently, packA manages ownership of `pmat`.
                                 // TODO: Consider a way to pre-allocate and reuse
                                 // pmat buffer.

          // ReQuantizeOutput requires pointers to the zero point values,
          // since in the case of rowwise quantization these will be arrays rather
          // than scalars. But in this case, we're doing whole-tensor quantization
          // so we just pass a pointer to the scale values (and internally
          // ReQuantizeOutput won't index past 0.

          // This is the end of the pipeline, pass the resulting matrix through.
          fbgemm::DoNothing<> doNothingObj{};

          if (q_scheme == kPerTensorAffine) {
            // Process the per tensor quantization.
            //
            // After the uint8 * int8 matrix multiplication is performed, this
            // operation does:
            //  1) Add in row and column offsets to the rows and columns,
            //  respectively.
            //  2) Add in the bias term.
            fbgemm::ReQuantizeOutput<
                ReluFused,
                fbgemm::QuantizationGranularity::TENSOR,
                float>
                outputProcObj(
                    doNothingObj,
                    output_multiplier_float.data(),
                    output_zero_point_int32,
                    input_zero_point_int32,
                    w_zp.data(),
                    packA.getRowOffsetBuffer(),
                    col_offsets.data(),
                    bias_ptr,
                    N, /* nCol */
                    1 /* groups */,
                    act_times_w_scale.data());

            // Do the GEMM
            fbgemm::fbgemmPacked(
                /*packA=*/packA,
                /*packB=*/*packB,
                /*C=*/reinterpret_cast<u8*>(output.data_ptr<quint8>()),
                /*C_buffer=*/buffer.data_ptr<i32>(),
                /*ldc=*/N,
                /*outProcess=*/outputProcObj,
                /*thread_id=*/task_id,
                /*num_threads=*/num_tasks);
          } else if (q_scheme == kPerChannelAffine) {
            // Process the per channel quantization.
            //
            // After the uint8 * int8 matrix multiplication is performed, this
            // operation does:
            //  1) Add in row and column offsets to the rows and columns,
            //  respectively.
            //  2) Add in the bias term.
            fbgemm::ReQuantizeOutput<
                ReluFused,
                fbgemm::QuantizationGranularity::OUT_CHANNEL,
                float>
                outputProcObj(
                    doNothingObj,
                    output_multiplier_float.data(),
                    output_zero_point_int32,
                    input_zero_point_int32,
                    w_zp.data(),
                    packA.getRowOffsetBuffer(),
                    col_offsets.data(),
                    bias_ptr,
                    N, /*nCol=*/
                    1, /* groups*/
                    act_times_w_scale.data());

            // Do the GEMM
            fbgemm::fbgemmPacked(
                /*packA=*/packA,
                /*packB=*/*packB,
                /*C=*/reinterpret_cast<u8*>(output.data_ptr<quint8>()),
                /*C_buffer=*/buffer.data_ptr<i32>(),
                /*ldc=*/N,
                /*outProcess=*/outputProcObj,
                /*thread_id=*/task_id,
                /*num_threads=*/num_tasks);
          }
        }
      });

      return output;
        */
    }
    
    #[cfg(feature = "fbgemm")]
    pub fn apply(&mut self, 
        input:             Tensor,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor {
        
        todo!();
        /*
            // Allocate output Tensor
      auto output = _empty_affine_quantized(
          {0},
          device(kCPU).dtype(kQUInt8),
          output_scale,
          output_zero_point);
      apply_impl<false>(input, output_scale, output_zero_point, output);
      return output;
        */
    }
    
    #[cfg(feature = "fbgemm")]
    pub fn apply_relu(&mut self, 
        input:             Tensor,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor {
        
        todo!();
        /*
            auto output = _empty_affine_quantized(
          {0},
          device(kCPU).dtype(kQUInt8),
          output_scale,
          output_zero_point);
      apply_impl<true>(input, output_scale, output_zero_point, output);
      return output;
        */
    }
    
    #[cfg(feature = "fbgemm")]
    pub fn apply_out<'a>(&mut self, 
        input:             &Tensor,
        output_scale:      f64,
        output_zero_point: i64,
        output:            &mut Tensor) -> &'a mut Tensor {
        
        todo!();
        /*
            TORCH_CHECK(
          (output.device() == kCPU) && (output.dtype() == kQUInt8) &&
          (output.q_scale() == output_scale) &&
          (output.q_zero_point() == output_zero_point));
      return apply_impl<false>(input, output_scale, output_zero_point, output);
        */
    }
    
    pub fn apply_relu_out<'a>(&mut self, 
        input:             &Tensor,
        output_scale:      f64,
        output_zero_point: i64,
        output:            &mut Tensor) -> &'a mut Tensor {
        
        todo!();
        /*
            TORCH_CHECK(
          (output.device() == kCPU) && (output.dtype() == kQUInt8) &&
          (output.q_scale() == output_scale) &&
          (output.q_zero_point() == output_zero_point));
      return apply_impl<true>(input, output_scale, output_zero_point, output);
        */
    }
    
    #[cfg(feature = "fbgemm")]
    pub fn apply_dynamic_impl<const ReluFused: bool>(
        &mut self, 
        input:        Tensor,
        reduce_range: Option<bool>

    ) -> Tensor {

        let reduce_range: bool = reduce_range.unwrap_or(false);
    
        todo!();
        /*
            using Tensor;
      // fp32 * int8 -> fp32 (with quantization on activation, and dequantization
      // on the result).

      // We make a strong guarantee that models using these operators will have
      // the same numerics across different machines. Therefore, we do not provide
      // a fallback path and rather fail loudly if we cannot run FBGEMM.
      TORCH_CHECK(
          fbgemm::fbgemmSupportedCPU(), "Your CPU does not support FBGEMM.");

      // TODO: contiguous is called for further jit optimizations.
      auto input_contig = input.contiguous();
      const auto* input_ptr = input_contig.data_ptr<float>();

      TORCH_CHECK(
          input.dim() >= 2,
          "The dimension of input tensor should be larger than or equal to 2");
      // C(output) = A(input) x B(weight), where C, A, B are M x N, M x K, K x N
      // matrices, respectively.
      i64 M = Sizeo_dim_(input.dim() - 1, input.sizes());

      auto packB = w.get();

      i64 N = static_cast<i64>(packB->numCols());
      i64 K = input.size(input.dim() - 1);
      TORCH_CHECK(
          K == static_cast<i64>(packB->numRows()),
          "The number of rows in the packB should be equal to K: " +
              to_string(K));

      // Calculate statistics for quantization of the input Tensor
      float x_min, x_max;
      fbgemm::FindMinMax(
          /*m=*/input_ptr,
          /*min=*/&x_min,
          /*max=*/&x_max,
          /*len=*/input.numel());

      // Input tensor is quantized as 8-bit unsigned values
      static constexpr int precision = 8;
      static constexpr bool is_signed = false;

      // Calculate scale and zero point for quantization of input tensor
      auto q_params = quant_utils::ChooseQuantizationParams(
          /*min=*/x_min,
          /*max=*/x_max,
          /*qmin=*/is_signed ? -(1 << (precision - 1)) : 0,
          /*qmax=*/
          is_signed ? ((1 << (precision - 1)) - 1) : (1 << precision) - 1,
          /*preserve_sparsity=*/false,
          /*force_scale_power_of_two=*/false,
          /*reduce_range=*/reduce_range);

      q_params.precision = precision;

      // ReQuantizeForFloat requires pointers to the zero point values,
      // since in the case of rowwise quantization these will be arrays rather
      // than scalars. But in this case, we're doing whole-tensor quantization so
      // we just pass a pointer to the scale values (and internally
      // ReQuantizeForFloat won't index past 0.

      const float* bias_ptr = nullptr;
      Tensor bias_vec;
      if (bias_.has_value()) {
        bias_vec = bias_.value();
        TORCH_CHECK(bias_vec.dim() == 1, "bias should be a vector (1D Tensor)");
        TORCH_CHECK(
            bias_vec.size(0) == N,
            "bias should have N elements: " + to_string(N));
        // TODO: contiguous is called for further jit optimizations.
        auto bias_contig = bias_vec.contiguous();
        bias_ptr = bias_contig.data_ptr<float>();
      }
      // The resulting matrix here is 2-D, let's view it with the original
      // left hand dimensions of the input. Here are two examples:
      // 1. If the input tensor is {M, K}, the output tensor is {M, N}.
      // 2. If the input tensor is {b, M, K}, the output tensor is {b, M, N}.
      vector<i64> out_sizes = input.sizes().vec();
      out_sizes.back() = N;
      // Allocate output Tensor and a buffer for fbgemmPacked to use
      auto output = empty(out_sizes, input.options().dtype(kFloat));
      auto buffer = empty_like(
          output,
          output.options().dtype(kInt),
          LEGACY_CONTIGUOUS_MEMORY_FORMAT);

      int num_tasks = get_num_threads();
      parallel_for(0, num_tasks, 1, [&](i64 begin, i64 end) {
        // This operation does the following:
        // 1) Quantizes the input matrix given the statistics we've calculated
        // above
        // 2) Creates a "row buffer" vector with offset values that must be
        // added
        //    to the integer matrix multiplication operation to ensure
        //    correctness. This "row buffer" is also called the row offset, and it
        //    is needed when we use affine quantization for weights.
        // 3) Packs the resulting quantized matrix into vector-register and cache
        //    friendly tiles.
        //
        //  Note this is not executed eagerly, but rather within the fbgemmPacked
        //  call below.

        fbgemm::PackAWithQuantRowOffset<u8> packA(
            /*trans=*/fbgemm::matrix_op_t::NoTranspose,
            /*nRow=*/M,
            /*nCol=*/K,
            /*smat=*/input_ptr,
            /*ld=*/K,
            /*pmat=*/nullptr, // Currently, packA manages ownership of `pmat`.
            /*scale=*/q_params.scale,
            /*zero_pt=*/q_params.zero_point);
        // TODO: Consider a way to pre-allocate and reuse
        // pmat buffer.

        // This is the end of the pipeline, pass the resulting matrix through.
        fbgemm::DoNothing<float, float> doNothingObj{};

        for (const auto task_id : irange(begin, end)) {
          if (q_scheme == kPerTensorAffine) {
            // Process the per tensor quantization.
            //
            // After the uint8 * int8 matrix multiplication is performed, this
            // operation does:
            //  1) Add in row and column offsets to the rows and columns,
            //  respectively.
            //  2) Dequantize the results into floating point.
            //  3) Add in the bias term.
            fbgemm::ReQuantizeForFloat<ReluFused> outputProcObj(
                /*nextop=*/doNothingObj,
                /*Aq_scale=*/q_params.scale,
                /*Bq_scale=*/w_scale.data(),
                /*Aq_zero_point=*/q_params.zero_point,
                /*Bq_zero_point=*/w_zp.data(),
                /*row_offsets=*/packA.getRowOffsetBuffer(),
                /*col_offsets=*/col_offsets.data(),
                /*bias=*/bias_ptr,
                /*nCol=*/N);

            // Do the GEMM
            fbgemm::fbgemmPacked(
                /*packA=*/packA,
                /*packB=*/*packB,
                /*C=*/output.data_ptr<float>(),
                /*C_buffer=*/buffer.data_ptr<i32>(),
                /*ldc=*/N,
                /*outProcess=*/outputProcObj,
                /*thread_id=*/task_id,
                /*num_threads=*/num_tasks);

          } else if (q_scheme == kPerChannelAffine) {
            // Process the per channel quantization.
            //
            // After the uint8 * int8 matrix multiplication is performed, this
            // operation does:
            //  1) Add in row and column offsets to the rows and columns,
            //  respectively.
            //  2) Dequantize the results into floating point.
            //  3) Add in the bias term.
            fbgemm::ReQuantizeForFloat<
                ReluFused,
                fbgemm::QuantizationGranularity::OUT_CHANNEL>
                outputProcObj(
                    /*nextop=*/doNothingObj,
                    /*Aq_scale=*/q_params.scale,
                    /*Bq_scale=*/w_scale.data(),
                    /*Aq_zero_point=*/q_params.zero_point,
                    /*Bq_zero_point=*/w_zp.data(),
                    /*row_offsets=*/packA.getRowOffsetBuffer(),
                    /*col_offsets=*/col_offsets.data(),
                    /*bias=*/bias_ptr,
                    /*nCol=*/N);

            // Do the GEMM
            fbgemm::fbgemmPacked(
                /*packA=*/packA,
                /*packB=*/*packB,
                /*C=*/output.data_ptr<float>(),
                /*C_buffer=*/buffer.data_ptr<i32>(),
                /*ldc=*/N,
                /*outProcess=*/outputProcObj,
                /*thread_id=*/task_id,
                /*num_threads=*/num_tasks);
          }
        }
      });

      return output;
        */
    }
    
    #[cfg(feature = "fbgemm")]
    pub fn apply_dynamic(
        &mut self, 
        input:        Tensor,
        reduce_range: Option<bool>
    ) -> Tensor {
        
        let reduce_range: bool = reduce_range.unwrap_or(false);

        todo!();
        /*
            return apply_dynamic_impl</*ReluFused=*/false>(move(input), reduce_range);
        */
    }
    
    #[cfg(feature = "fbgemm")]
    pub fn apply_dynamic_relu(
        &mut self, 
        input:        Tensor,
        reduce_range: Option<bool>

    ) -> Tensor {

        let reduce_range: bool = reduce_range.unwrap_or(false);
        
        todo!();
        /*
            return apply_dynamic_impl</*ReluFused=*/true>(move(input), reduce_range);
        */
    }
    
    #[cfg(feature = "fbgemm")]
    pub fn new(
        w:           Box<FbgemmPackBMatrix<i8>>,
        bias:        Option<Tensor>,
        col_offsets: Vec<i32>,
        w_scale:     Vec<f32>,
        w_zp:        Vec<i32>,
        q_scheme:    QScheme) -> Self {
    
        todo!();
        /*


            : w(move(w)),
            bias_(move(bias)),
            col_offsets(move(col_offsets)),
            w_scale(move(w_scale)),
            w_zp(move(w_zp)),
            q_scheme(move(q_scheme))
        */
    }
}
