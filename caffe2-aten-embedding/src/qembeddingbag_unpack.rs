crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qembeddingbag_unpack.cpp]

pub fn register_embedding_params() -> TorchClass<EmbeddingPackedParamsBase> {
    
    todo!();
        /*
        
        */
}

impl PackedEmbeddingBagWeight {
    
    pub fn unpack(&mut self) -> Tensor {
        
        todo!();
        /*
            auto packed_weight = packed_w;
      Tensor weight_origin;

      if (bit_rate_ == 8 || bit_rate_ == 4) {
        const auto input_rows = packed_weight.size(0);
        const auto input_columns = packed_weight.size(1);
        int scale_bias_bytes;
        const auto num_elem_per_byte = 8 / bit_rate_;
        if (bit_rate_ == 8) {
          // The last 2 values are used to store the FP32 scale and zero_point
          // values per row.
          scale_bias_bytes = 8;
        } else {
          scale_bias_bytes = 4;
        }

        const auto* input = packed_weight.data_ptr<u8>();
        // Calculate the output shape, accounting for the last n bytes to be used
        // for scale/bias rest of the entries are packed depending on the bit_width.
        vector<i64> output_shape = {
            input_rows,
            static_cast<i64>(input_columns - scale_bias_bytes) *
                num_elem_per_byte};

        auto scales = from_blob(
            w_scale.data(), w_scale.size(), device(kCPU).dtype(kFloat));
        auto zero_points = from_blob(
            w_zp.data(), w_zp.size(), device(kCPU).dtype(kFloat));

        auto output_columns = output_shape[1];
        u8* output_data;

        // Allocate output weight tensor based on the bit_width
        if (bit_rate_ == 8) {
          weight_origin = _empty_per_channel_affine_quantized(
              output_shape,
              scales.toType(kFloat),
              zero_points.toType(kFloat),
              0, // The output channel axis is 0
              device(kCPU).dtype(kQUInt8));
          output_data = static_cast<u8*>(weight_origin.data_ptr());
        } else {
          // We create empty qtensor with the full output shape, and dtype set to
          // quint4x2 This will internally allocate appropriate storage bytes to
          // account for the packed nature of this dtype.
          weight_origin = _empty_per_channel_affine_quantized(
              output_shape,
              scales.toType(kFloat),
              zero_points.toType(kFloat),
              0, // The output channel axis is 0
              device(kCPU).dtype(kQUInt4x2));
          output_data = static_cast<u8*>(weight_origin.data_ptr());
        }

        // Copy over the data from the packed weight to the output.
        // For sub-byte tensors this will copy the packed bytes over since the
        // sub_byte qtensors are expected to store data in packed format.
        parallel_for(0, input_rows, 1, [&](i32 start_idx, i32 end_idx) {
          for (i64 row = start_idx; row < end_idx; ++row) {
            const u8* input_row = input + row * input_columns;
            u8* output_row =
                output_data + row * output_columns / num_elem_per_byte;

            for (usize col = 0; col < output_columns / num_elem_per_byte;
                 ++col) {
              output_row[col] = input_row[col];
            } // output_columns
          }
        });

        return weight_origin;
      }
      TORCH_INTERNAL_ASSERT(
          false,
          "We currently only support 8-bit and 4-bit quantization of embedding_bag.");
      return weight_origin;
        */
    }
}

pub fn qembeddingbag_byte_unpack(packed_weight: &Tensor) -> Tensor {
    
    todo!();
        /*
            // The "last" dimension of an N-Dimensioned batch of embedding bags is
      // quantization channel. E.g. for a 2D embedding bag, this has
      // [ row, col ] dimensions, for batched of embedding bags, dimensions might be
      // [ batch, row, col ].
      //
      // Python Batched Embedding Example:
      // weights = torch.from_numpy((np.random.random_sample((
      //          2, 10, 3)).squeeze() + 1).astype(np.float32))
      // assert(weights.size() == torch.usize([2, 10, 3]))
      // # NOTE: 8 bytes (columns) are added due to fp32 zero_point and scales
      // packed_weights = torch.ops.quantized.embedding_bag_byte_prepack(weights)
      // assert(packed_weights.size() == torch.usize([2, 10, 11]))
      // unpacked_weights = torch.ops.quantized.embedding_bag_byte_unpack(packed_weights)
      // assert(unpacked_weights.size() == torch.usize([2, 10, 3]))
      const auto packed_weight_sizes = packed_weight.sizes();
      const auto col_dim = packed_weight_sizes.size() - 1;
      const i32 input_rows = Sizeo_dim_(col_dim, packed_weight_sizes);
      const i32 input_columns = packed_weight_sizes[col_dim];
      // The last 2 values are used to store the FP32 scale and zero_point values
      // per row.
      const i32 output_columns = input_columns - 2 * sizeof(float);
      const auto* input_data = packed_weight.data_ptr<u8>();

      vector<i64> output_shape = packed_weight_sizes.vec();
      output_shape[col_dim] = output_columns;
      Tensor output = empty(
          output_shape,
          packed_weight.options().dtype(kFloat),
          packed_weight.suggest_memory_format());
      float* output_data = output.data_ptr<float>();

    #ifdef USE_FBGEMM
        parallel_for(
          0, input_rows, 1, [&](i32 start_idx, i32 end_idx) {
            for (i64 row = start_idx; row < end_idx; ++row) {
              fbgemm::Fused8BitRowwiseQuantizedSBFloatToFloatOrHalf<float>(
                input_data + row * input_columns,
                1,
                input_columns,
                output_data + row * output_columns);
            }
          });
    #else
      for (usize row = 0; row < input_rows; ++row) {
        const u8* input_row = input_data + row * input_columns;
        const float* input_row_scale_zp =
            reinterpret_cast<const float*>(input_row + output_columns);
        float* output_row = output_data + row * output_columns;

        for (usize col = 0; col < output_columns; ++col) {
          output_row[col] =
              input_row[col] * input_row_scale_zp[0] + input_row_scale_zp[1];
        } // output_columns
      } // input_rows
    #endif // USE_FBGEMM
      return output;
        */
}

pub fn qembeddingbag_nbit_unpack_helper(
        packed_weight: &Tensor,
        BIT_RATE:      i32) -> Tensor {
    
    todo!();
        /*
            const auto input_rows = packed_weight.size(0);
      const auto input_columns = packed_weight.size(1);
      const auto* input_data = packed_weight.data_ptr<u8>();
      int NUM_ELEM_PER_BYTE = 8 / BIT_RATE;

      // The last 4 bytes per row are two fp16 scale and zero_point.
      // The rest of input_columns is the number of values in the original row.
      vector<i64> output_dimensions = {
          input_rows,
          static_cast<i64>(input_columns - 2 * sizeof(Half)) *
              NUM_ELEM_PER_BYTE};

      auto output = empty(
          output_dimensions,
          packed_weight.options().dtype(kFloat),
          packed_weight.suggest_memory_format());
      float* output_data = output.data_ptr<float>();
    #ifdef USE_FBGEMM
        parallel_for(
          0, input_rows, 1, [&](i32 start_idx, i32 end_idx) {
            for (i64 row = start_idx; row < end_idx; ++row) {
              fbgemm::FusedNBitRowwiseQuantizedSBHalfToFloatOrHalf<float>(BIT_RATE,
                input_data + row * input_columns,
                1,
                input_columns,
                output_data + row * output_dimensions[1]);
            }
          });
    #else
      auto output_columns = output_dimensions[1];
      for (usize row = 0; row < input_rows; ++row) {
        float* output_row = output_data + row * output_columns;
        const u8* input_row = input_data + row * input_columns;
        const Half* input_row_scale_zp = reinterpret_cast<const Half*>(
            input_row +
            (output_columns + NUM_ELEM_PER_BYTE - 1) / NUM_ELEM_PER_BYTE);
        float scale = input_row_scale_zp[0];
        float zero_point = input_row_scale_zp[1];

        for (int col = 0; col < output_columns; ++col) {
          u8 quantized = input_row[col / NUM_ELEM_PER_BYTE];
          quantized >>= (col % NUM_ELEM_PER_BYTE) * BIT_RATE;
          quantized &= (1 << BIT_RATE) - 1;
          output_row[col] = scale * quantized + zero_point;
        } // output_columns
      } // input_rows
    #endif // USE_FBGEMM

      return output;
        */
}

/**
  | De-quantizes the result of the qembeddingbag_4bit_prepack
  | operator.
  | 
  | The input is expected to first have quantized
  | values, then 2-byte fp16 scale and 2-byte
  | zero_offset.
  | 
  | The output is a matrix containing only
  | the values, but de-quantized.
  | 
  | De-quantization is performed by multiplying
  | each value by its row's scale and zero_point
  | parameters. The de-quantized values
  | will thus not be exactly equal to the
  | original, un-quantized floating point
  | values.
  |
  */
pub fn qembeddingbag_4bit_unpack(packed_weight: &Tensor) -> Tensor {
    
    todo!();
        /*
            return _qembeddingbag_nbit_unpack_helper(packed_weight, 4 /*BIT_RATE*/);
        */
}

/**
  | De-quantizes the result of the qembeddingbag_2bit_prepack
  | operator.
  | 
  | The input is expected to first have quantized
  | values, then 2-byte fp16 scale and 2-byte
  | zero_offset.
  | 
  | The output is a matrix containing only
  | the values, but de-quantized.
  | 
  | De-quantization is performed by multiplying
  | each value by its row's scale and zero_point
  | parameters. The de-quantized values
  | will thus not be exactly equal to the
  | original, un-quantized floating point
  | values.
  |
  */
pub fn qembeddingbag_2bit_unpack(packed_weight: &Tensor) -> Tensor {
    
    todo!();
        /*
            return _qembeddingbag_nbit_unpack_helper(packed_weight, 2 /*BIT_RATE*/);
        */
}

pub struct QEmbeddingUnpackWeights {

}

impl QEmbeddingUnpackWeights {
    
    pub fn run(packed_weight: &IntrusivePtr<EmbeddingPackedParamsBase>) -> Tensor {
        
        todo!();
        /*
            return packed_weight->unpack();
        */
    }
}

lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(quantized, CPU, m) {
      m.impl(
          TORCH_SELECTIVE_NAME("quantized::embedding_bag_byte_unpack"),
          qembeddingbag_byte_unpack);
      m.impl(
          TORCH_SELECTIVE_NAME("quantized::embedding_bag_4bit_unpack"),
          qembeddingbag_4bit_unpack);
      m.impl(
          TORCH_SELECTIVE_NAME("quantized::embedding_bag_2bit_unpack"),
          qembeddingbag_2bit_unpack);
    }
    */
}

lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(quantized, CatchAll, m) {
      // Unpack the packed embedding_bag weights using TorchBind custom class.
      // TODO extend to support 4-bit qtensor.
      m.impl(
          TORCH_SELECTIVE_NAME("quantized::embedding_bag_unpack"),
          TORCH_FN(QEmbeddingUnpackWeights::run));
    }
    */
}
