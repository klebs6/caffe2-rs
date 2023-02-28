crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qlinear.cpp]

pub fn register_linear_params() -> TorchClass<LinearPackedParamsBase> {
    
    todo!();
        /*
        
        */
}

#[cfg(feature = "fbgemm")]
impl PackedLinearWeight {
    
    pub fn apply_impl<const ReluFused: bool>(&mut self, 
        input:             &Tensor,
        output_scale:      f64,
        output_zero_point: i64,
        output:            &mut Tensor) -> &mut Tensor {
    
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
    
    pub fn apply_out(&mut self, 
        input:             &Tensor,
        output_scale:      f64,
        output_zero_point: i64,
        output:            &mut Tensor) -> &mut Tensor {
        
        todo!();
        /*
            TORCH_CHECK(
          (output.device() == kCPU) && (output.dtype() == kQUInt8) &&
          (output.q_scale() == output_scale) &&
          (output.q_zero_point() == output_zero_point));
      return apply_impl<false>(input, output_scale, output_zero_point, output);
        */
    }
    
    pub fn apply_relu_out(&mut self, 
        input:             &Tensor,
        output_scale:      f64,
        output_zero_point: i64,
        output:            &mut Tensor) -> &mut Tensor {
        
        todo!();
        /*
            TORCH_CHECK(
          (output.device() == kCPU) && (output.dtype() == kQUInt8) &&
          (output.q_scale() == output_scale) &&
          (output.q_zero_point() == output_zero_point));
      return apply_impl<true>(input, output_scale, output_zero_point, output);
        */
    }
}

#[cfg(USE_PYTORCH_QNNPACK)]
impl PackedLinearWeightsQnnp {
    
    pub fn apply_impl<const ReluFused: bool>(&mut self, 
        input:             Tensor,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor {
    
        todo!();
        /*
            TORCH_CHECK(
          input.dim() >= 2,
          "quantized::linear(): Input tensor rank should be >= 2");
      auto input_contig = input.contiguous();

      auto packB = w.get();
      usize rows_w = bias_.size(0);
      usize cols_w = input_contig.size(input_contig.dim() - 1);
      auto input_scale = input_contig.q_scale();

      if (!this->input_scale.has_value() ||
          this->input_scale.value() != input_scale) {
        // Get the original weight and adjust it to uint8 from int8
        auto weight_contig = orig_weight;
        auto bias_fp32 = bias_;
        i8* w_data = (i8*)weight_contig.data_ptr<qint8>();

        float* weight_scales_data = w_scales.data_ptr<float>();
        // We calculate requant scale here as the vector holding the requant scale
        // is owned by this module. The pointer is then passed to qnnpack backend.
        generate_requantization_scales(
            w_scales, input_scale, output_scale, requantization_scales);

        Tensor qnnp_weight = _empty_affine_quantized(
            weight_contig.sizes(),
            device(kCPU).dtype(kQUInt8),
            weight_scales_data[0],
            w_zero_points[0]);
        auto* qnnp_w_data = qnnp_weight.data_ptr<quint8>();
        auto wt_numel = weight_contig.numel();
        for (int i = 0; i < wt_numel; ++i) {
          qnnp_w_data[i] = static_cast<quint8>(w_data[i] + 128);
        }
        // Original bias was float, so we requantize it here.
        const bool is_per_channel = orig_weight.qscheme() == kPerChannelAffine;
        Tensor qbias;
        // Original bias was float, so we requantize it here.
        if (is_per_channel) {
          Tensor bias_quant_scales =
              weight_contig.q_per_channel_scales() * input_scale;
          Tensor bias_zp = zeros(bias_quant_scales.sizes(), kInt);
          qbias = native::quantize_per_channel_cpu(
              bias_fp32, bias_quant_scales, bias_zp, 0, kQInt32);
        } else {
          qbias = native::quantize_per_tensor(
              bias_fp32, weight_contig.q_scale() * input_scale, 0, kQInt32);
        }

        // Update the input scale to not pack again.
        this->input_scale = input_scale;
        w.reset();
        w = make_unique<qnnpack::PackBMatrix>(
            cols_w /* input_channels */,
            rows_w /* output_channels */,
            w_zero_points.data(),
            requantization_scales.data(),
            reinterpret_cast<u8*>(qnnp_w_data),
            reinterpret_cast<i32*>(qbias.data_ptr<qint32>()));
        packB = w.get();
        if (globalContext().releaseWeightsWhenPrepacking()) {
          // On mobile, we release the original weight by resetting the intrusive_ptr.
          // Calling unpack after this will throw an assertion.
          orig_weight.reset();
        }
      }

      usize rows_input = 1;
      usize cols_input = input_contig.size(input_contig.dim() - 1);
      for (usize i = 0; i < input_contig.dim() - 1; ++i) {
        rows_input *= input_contig.size(i);
      }

      TORCH_CHECK(
          cols_input == cols_w,
          "quantized::linear(): input size does not match weight dimension 1 size: \
             got ",
          cols_input,
          " but expected ",
          cols_w);

      // Allocate output Tensor and a buffer for QNNPACK to use
      // The resulting matrix here is 2-D, let's view it with the original
      // left hand dimensions of the input. Here are two examples:
      // 1. If the input tensor is {M, K}, the output tensor is {M, N}.
      // 2. If the input tensor is {b, M, K}, the output tensor is {b, M, N}.
      vector<i64> out_sizes = input.sizes().vec();
      out_sizes.back() = static_cast<long>(rows_w);
      Tensor output = _empty_affine_quantized(
          out_sizes,
          input.options(),
          output_scale,
          output_zero_point);

      auto output_min = ReluFused
          ? activationLimits(output_scale, output_zero_point, Activation::RELU)
                .first
          : u8::min;
      auto output_max = ReluFused
          ? activationLimits(output_scale, output_zero_point, Activation::RELU)
                .second
          : u8::max;
      TORCH_INTERNAL_ASSERT(packB != nullptr, "Packed Weights are NULL");
      const pytorch_qnnp_status runStatus = qnnpack::qnnpackLinear(
          rows_input /* batch_size */,
          cols_input /* input_channels */,
          rows_w /* output_channels */,
          input_contig.q_zero_point(),
          w_zero_points.data(),
          requantization_scales.data(),
          output_zero_point,
          output_min,
          output_max,
          (u8*)input_contig.data_ptr<quint8>(),
          cols_input /* input_stride */,
          packB->getPackedWeights(),
          (u8*)output.data_ptr<quint8>(),
          rows_w /* output_stride */,
          // TODO (Ashkan): Disabling temporarily.
          // Throws a floating point exception with OSS pthreadpool.
          pthreadpool_() /* threadpool */);

      TORCH_INTERNAL_ASSERT(
          runStatus == pytorch_qnnp_status_success,
          "failed to run QNNPACK Linear operator");

      return output;
        */
    }
    
    pub fn apply(&mut self, 
        input:             Tensor,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor {
        
        todo!();
        /*
            return apply_impl<false>(move(input), output_scale, output_zero_point);
        */
    }
    
    pub fn apply_relu(&mut self, 
        input:             Tensor,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor {
        
        todo!();
        /*
            return apply_impl<true>(move(input), output_scale, output_zero_point);
        */
    }
}

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
