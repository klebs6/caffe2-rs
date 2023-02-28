crate::ix!();

#[inline] pub fn float_to_fused_8bit_rowwise_quantized_base(
    input:         *const f32,
    input_rows:    i32,
    input_columns: i32,
    output:        *mut u8)  {
    
    todo!();
    /*
        constexpr float kEpsilon = 1e-8f;

      int output_columns = input_columns + 2 * sizeof(float);
      for (size_t row = 0; row < input_rows; ++row) {
        const float* input_row = input + row * input_columns;
        uint8_t* output_row = output + row * output_columns;
        float* output_row_scale_bias =
            reinterpret_cast<float*>(output_row + input_columns);

        float minimum_element =
            *min_element(input_row, input_row + input_columns);
        float maximum_element =
            *max_element(input_row, input_row + input_columns);
        float range = maximum_element - minimum_element;

        output_row_scale_bias[0] = range / 255.0f;
        output_row_scale_bias[1] = minimum_element;
        const auto inverse_scale = 255.0f / (range + kEpsilon);
        for (size_t col = 0; col < input_columns; ++col) {
          output_row[col] =
              lrintf((input_row[col] - minimum_element) * inverse_scale);
        }
      }
    */
}

#[inline] pub fn fused_8bit_rowwise_quantized_to_float_base(
    input:         *const u8,
    input_rows:    i32,
    input_columns: i32,
    output:        *mut f32)  {
    
    todo!();
    /*
        int output_columns = input_columns - 2 * sizeof(float);

      for (size_t row = 0; row < input_rows; ++row) {
        const uint8_t* input_row = input + row * input_columns;
        const float* input_row_scale_bias =
            reinterpret_cast<const float*>(input_row + output_columns);
        float* output_row = output + row * output_columns;

        for (size_t col = 0; col < output_columns; ++col) {
          output_row[col] =
              input_row[col] * input_row_scale_bias[0] + input_row_scale_bias[1];
        }
      }
    */
}

#[inline] pub fn float_to_fused_8bit_rowwise_quantized(
    input:         *const f32,
    input_rows:    i32,
    input_columns: i32,
    output:        *mut u8)  {
    
    todo!();
    /*
        #ifdef USE_FBGEMM
      fbgemm::FloatToFused8BitRowwiseQuantizedSBFloat(
          input, input_rows, input_columns, output);
    #else
      FloatToFused8BitRowwiseQuantized__base(
          input, input_rows, input_columns, output);
    #endif
    */
}

#[inline] pub fn fused_8bit_rowwise_quantized_to_float(
    input:         *const u8,
    input_rows:    i32,
    input_columns: i32,
    output:        *mut f32)  {
    
    todo!();
    /*
        #ifdef USE_FBGEMM
      fbgemm::Fused8BitRowwiseQuantizedSBFloatToFloat(
          input, input_rows, input_columns, output);
    #else
      Fused8BitRowwiseQuantizedToFloat__base(
          input, input_rows, input_columns, output);
    #endif
    */
}

#[inline] pub fn float_to_fusedn_bit_rowwise_quantized_sb_half_base(
    bit_rate:      i32,
    input:         *const f32,
    input_rows:    i32,
    input_columns: i32,
    output:        *mut u8)  {
    
    todo!();
    /*
        int num_elem_per_byte = 8 / bit_rate;
      int output_columns =
          (input_columns + num_elem_per_byte - 1) / num_elem_per_byte +
          2 * sizeof(at::Half);
      for (size_t row = 0; row < input_rows; ++row) {
        const float* input_row = input + row * input_columns;
        uint8_t* output_row = output + row * output_columns;
        at::Half* output_row_scale_bias = reinterpret_cast<at::Half*>(
            output_row +
            (input_columns + num_elem_per_byte - 1) / num_elem_per_byte);

        float minimum_element =
            *min_element(input_row, input_row + input_columns);
        float maximum_element =
            *max_element(input_row, input_row + input_columns);

        minimum_element = static_cast<at::Half>(minimum_element);
        const float range = maximum_element - minimum_element;

        at::Half scale = range == 0 ? 1.0f : range / ((1 << bit_rate) - 1);
        if (scale == 0) {
          // Corner case handling when maximum_element == minimum_element
          // Any scale would work because X - minimum_element will be 0 for all X
          scale = 1.0f;
        }
        float inverse_scale = 1.0f / scale;
        if (isinf(inverse_scale)) {
          scale = 1.0f;
          inverse_scale = 1.0f;
        }

        output_row_scale_bias[0] = scale;
        output_row_scale_bias[1] = minimum_element;
        for (size_t col = 0; col < input_columns; ++col) {
          float X = input_row[col];
          uint8_t quantized = max(
              0,
              min<int>(
                  lrintf((X - minimum_element) * inverse_scale),
                  (1 << bit_rate) - 1));
          if (col % num_elem_per_byte == 0) {
            output_row[col / num_elem_per_byte] = quantized;
          } else {
            output_row[col / num_elem_per_byte] |=
                (quantized << ((col % num_elem_per_byte) * bit_rate));
          }
        }
      }
    */
}

#[inline] pub fn fusedn_bit_rowwise_quantized_sb_half_to_float_base(
    bit_rate:      i32,
    input:         *const u8,
    input_rows:    i32,
    input_columns: i32,
    output:        *mut f32)  {
    
    todo!();
    /*
        int num_elem_per_byte = 8 / bit_rate;
      int output_columns =
          (input_columns - 2 * sizeof(at::Half)) * num_elem_per_byte;

      for (size_t row = 0; row < input_rows; ++row) {
        const uint8_t* input_row = input + row * input_columns;
        const at::Half* input_row_scale_bias = reinterpret_cast<const at::Half*>(
            input_row +
            (output_columns + num_elem_per_byte - 1) / num_elem_per_byte);
        float scale = input_row_scale_bias[0];
        float bias = input_row_scale_bias[1];
        float* output_row = output + row * output_columns;

        for (size_t col = 0; col < output_columns; ++col) {
          uint8_t quantized = input_row[col / num_elem_per_byte];
          quantized >>= (col % num_elem_per_byte) * bit_rate;
          quantized &= (1 << bit_rate) - 1;
          output_row[col] = scale * quantized + bias;
        }
      }
    */
}

/**
  | Row-wise quantization with fp16 scale
  | and bias
  | 
  | -----------
  | @param bit_rate
  | 
  | can be 2, 4, or 8
  |
  */
#[inline] pub fn float_to_fusedn_bit_rowwise_quantized_sb_half(
    bit_rate:      i32,
    input:         *const f32,
    input_rows:    i32,
    input_columns: i32,
    output:        *mut u8)  {
    
    todo!();
    /*
        #ifdef USE_FBGEMM
      fbgemm::FloatToFusedNBitRowwiseQuantizedSBHalf(
          bit_rate, input, input_rows, input_columns, output);
    #else
      FloatToFusedNBitRowwiseQuantizedSBHalf__base(
          bit_rate, input, input_rows, input_columns, output);
    #endif
    */
}

#[inline] pub fn fusedn_bit_rowwise_quantized_sb_half_to_float(
    bit_rate:      i32,
    input:         *const u8,
    input_rows:    i32,
    input_columns: i32,
    output:        *mut f32)  {
    
    todo!();
    /*
        #ifdef USE_FBGEMM
      fbgemm::FusedNBitRowwiseQuantizedSBHalfToFloat(
          bit_rate, input, input_rows, input_columns, output);
    #else
      FusedNBitRowwiseQuantizedSBHalfToFloat__base(
          bit_rate, input, input_rows, input_columns, output);
    #endif
    */
}
