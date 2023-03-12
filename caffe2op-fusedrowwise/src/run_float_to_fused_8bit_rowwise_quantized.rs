crate::ix!();

impl<T, TypeForScaleAndBias, ConvertFn, const HAS_CONVERT: bool, Context> 
FloatToFused8BitRowwiseQuantizedOp<T, TypeForScaleAndBias, ConvertFn, HAS_CONVERT, Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            CAFFE_ENFORCE(IS_LITTLE_ENDIAN, "Unsupported endianness");

        const auto& input = Input(DATA_FLOAT);

        CAFFE_ENFORCE_GT(input.dim(), 0, "Input's dimension must be at least 1");
        const auto input_rows = input.size_to_dim(input.dim() - 1);
        const auto input_columns = input.size(input.dim() - 1);

        // The "fused" representation stores the scale and bias with the row-wise
        // quantized data in one tensor. Since we quantize with 8 bits (1 byte) and
        // represent the scale and bias with 32-bit floats, we'll use the last 8
        // bytes of each row for scale (4 bytes) and bias (4 bytes).
        // | ... int8 data ... | scale       | bias       |
        // | number_of_columns |  sizeof(Tsb)| sizeof(Tsb)|
        auto output_dimensions = input.sizes().vec();
        output_dimensions[input.dim() - 1] =
            input_columns + 2 * static_cast<std::int64_t>(sizeof(Tsb));
        auto* output = Output(
            DATA_FUSED_SCALE_BIAS_INT8,
            output_dimensions,
            at::dtype<std::uint8_t>());

        const auto* input_data = input.template data<T>();
        auto* output_data = output->template mutable_data<std::uint8_t>();
        const auto output_columns = output->size(output->dim() - 1);

        bool is_float = std::is_same<T, float>::value;
        bool out_sb_half = std::is_same<Tsb, at::Half>::value;

        if (!HAS_CONVERT) {
          CAFFE_ENFORCE(is_float, "convert can be nullptr only if T is float");
          if (out_sb_half) {
            FloatToFusedNBitRowwiseQuantizedSBHalf(
                8,
                reinterpret_cast<const float*>(input_data),
                input_rows,
                input_columns,
                output_data);
          } else {
            FloatToFused8BitRowwiseQuantized(
                reinterpret_cast<const float*>(input_data),
                input_rows,
                input_columns,
                output_data);
          }
        } else {
          bool is_half = std::is_same<T, at::Half>::value;
          CAFFE_ENFORCE(is_half);

          vector<float> tmp(input_columns);
          for (size_t row = 0; row < input_rows; ++row) {
            convert(tmp.data(), input_data + row * input_columns, input_columns);
            if (out_sb_half) {
              FloatToFusedNBitRowwiseQuantizedSBHalf(
                  8,
                  tmp.data(),
                  1,
                  input_columns,
                  output_data + row * output_columns);
            } else {
              FloatToFused8BitRowwiseQuantized(
                  tmp.data(), 1, input_columns, output_data + row * output_columns);
            }
          }
        }

        return true;
        */
    }
}

