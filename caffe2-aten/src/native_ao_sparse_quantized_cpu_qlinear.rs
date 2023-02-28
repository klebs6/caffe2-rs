crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/ao_sparse/quantized/cpu/qlinear.cpp]

pub fn register_linear_params() -> TorchClass<LinearPackedParamsBase> {
    
    todo!();
        /*
        
        */
}


impl PackedLinearWeight {
    
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
}

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
