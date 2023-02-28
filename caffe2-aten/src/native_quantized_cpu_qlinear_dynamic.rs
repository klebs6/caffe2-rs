crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qlinear_dynamic.cpp]

pub fn register_linear_params() -> TorchClass<LinearPackedParamsBase> {
    
    todo!();
        /*
        
        */
}

#[cfg(feature = "fbgemm")]
impl PackedLinearWeight {
    
    pub fn apply_dynamic_impl<const ReluFused: bool>(&mut self, 
        input:        Tensor,
        reduce_range: bool) -> Tensor {
    
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
    
    pub fn apply_dynamic(&mut self, 
        input:        Tensor,
        reduce_range: bool) -> Tensor {
        
        todo!();
        /*
            return apply_dynamic_impl</*ReluFused=*/false>(move(input), reduce_range);
        */
    }
    
    pub fn apply_dynamic_relu(&mut self, 
        input:        Tensor,
        reduce_range: bool) -> Tensor {
        
        todo!();
        /*
            return apply_dynamic_impl</*ReluFused=*/true>(move(input), reduce_range);
        */
    }
}

#[cfg(USE_PYTORCH_QNNPACK)]
impl PackedLinearWeightsQnnp {
    
    pub fn apply_dynamic_impl<const ReluFused: bool>(&mut self, input: Tensor) -> Tensor {
    
        todo!();
        /*
            using Tensor;
      TORCH_CHECK(
          input.dim() >= 2,
          "The dimension of input tensor should be larger than or equal to 2");
      auto input_contig = input.contiguous();
      // C(output) = A(input) x B(weight), where C, A, B are M x N, M x K, K x N
      // matrices, respectively.

      auto packB = w.get();
      usize rows_w = bias_.size(0);
      usize cols_w = input_contig.size(input_contig.dim() - 1);

      Tensor bias_vec = bias_;

      TORCH_CHECK(bias_vec.dim() == 1, "bias should be a vector (1D Tensor)");

      auto bias_contig = bias_vec.contiguous();
      const float* bias_ptr = bias_contig.data_ptr<float>();

      // Calculate statistics for quantization of input Tensor
      // TODO: optimized kernel
      float x_min;
      float x_max;
      if (input.numel() > 0) {
        x_min = input_contig.min().item<float>();
        x_max = input_contig.max().item<float>();
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
      float* weight_scales_data = w_scales.data_ptr<float>();
      if (!input_scale.has_value() || input_scale.value() != q_params.scale) {
        generate_requantization_scales(
            w_scales, q_params.scale, 1.f, requantization_scales);
      }

      if (!input_scale.has_value()) {
        // Get the original weight and adjust it to uint8 from int8
        auto weight_contig = orig_weight;

        // TODO(kimishpatel), we are allocating affine_quantized regardless of per channel or not.
        // This allocation is actually used only for packing weight and thus will be freed.
        // Still we should be consistent. Fix this.
        Tensor qnnp_weight = _empty_affine_quantized(
            weight_contig.sizes(),
            device(kCPU).dtype(kQUInt8),
            weight_scales_data[0],
            w_zero_points[0]);
        auto* qnnp_w_data = qnnp_weight.data_ptr<quint8>();
        i8* w_data = (i8*)weight_contig.data_ptr<qint8>();
        auto wt_numel = weight_contig.numel();
        for (int i = 0; i < wt_numel; ++i) {
          qnnp_w_data[i] = static_cast<quint8>(w_data[i] + 128);
        }

        // Pass in nullptr for bias, as we pass FP32 bias to run function.
        w.reset();
        w = make_unique<qnnpack::PackBMatrix>(
            cols_w /* input_channels */,
            rows_w /* output_channels */,
            w_zero_points.data(),
            requantization_scales.data(),
            (u8*)qnnp_w_data,
            nullptr);
        packB = w.get();
        if (globalContext().releaseWeightsWhenPrepacking()) {
          // On mobile, we release the original weight by resetting the intrusive_ptr.
          // Calling unpack after this will throw an assertion.
          orig_weight.reset();
        }
      }

      // Update the input scale to not pack weights again.
      // as well as to avoid repopulating requant scale if scale has not changed.
      input_scale = q_params.scale;

      // Quantize input
      Tensor q_input = quantize_per_tensor(
          input_contig, q_params.scale, q_params.zero_point, kQUInt8);

      // The resulting matrix here is 2-D, let's view it with the original
      // left hand dimensions of the input. Here are two examples:
      // 1. If the input tensor is {M, K}, the output tensor is {M, N}.
      // 2. If the input tensor is {b, M, K}, the output tensor is {b, M, N}.
      vector<i64> out_sizes = input.sizes().vec();
      out_sizes.back() = rows_w;

      auto output = empty(out_sizes, input.options().dtype(kFloat));

      usize rows_input = 1;
      usize cols_input = input_contig.size(input_contig.dim() - 1);
      for (usize i = 0; i < input_contig.dim() - 1; ++i) {
        rows_input *= input_contig.size(i);
      }
      pytorch_qnnp_status runStatus = qnnpack::qnnpackLinearDynamic(
          rows_input /* batch_size */,
          cols_input /* input_channels */,
          rows_w /* output_channels */,
          q_input.q_zero_point(),
          w_zero_points.data(),
          /* for dynamic should really be called dequant scale */
          requantization_scales.data(),
          (u8*)q_input.data_ptr<quint8>(),
          cols_input /* input_stride */,
          packB->getPackedWeights(),
          bias_ptr,
          output.data_ptr<float>(),
          rows_w /* output_stride */,
          pthreadpool_() /* threadpool */);

      TORCH_INTERNAL_ASSERT(
          runStatus == pytorch_qnnp_status_success,
          "failed to run QNNPACK Linear operator");
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
}

#[cfg(feature = "fbgemm")]
impl PackedLinearWeightFp16 {
    
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
