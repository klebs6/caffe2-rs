crate::ix!();

impl<T, TypeForScaleAndBias, ConvertFn, const HAS_CONVERT: bool, Context> 
Fused8BitRowwiseQuantizedToFloatOp<T, TypeForScaleAndBias, ConvertFn, HAS_CONVERT, Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            CAFFE_ENFORCE(IS_LITTLE_ENDIAN, "Unsupported endianness");

        const auto& input = Input(DATA_FUSED_SCALE_BIAS_INT8);

        CAFFE_ENFORCE_GT(input.dim(), 0, "Input's dimension must be at least 1");
        const auto input_rows = input.size_to_dim(input.dim() - 1);
        const auto input_columns = input.size(input.dim() - 1);

        // The last 2*sizeof(Tsb) bytes per row are the scale and the bias.
        // The rest of input_columns is the number of values in the original row.
        auto output_dimensions = input.sizes().vec();
        output_dimensions[input.dim() - 1] =
            input_columns - 2 * static_cast<std::int64_t>(sizeof(Tsb));
        auto* output = Output(DATA_FLOAT, output_dimensions, at::dtype<T>());
        const auto output_columns = output->size(output->dim() - 1);

        const auto* input_data = input.template data<std::uint8_t>();
        T* output_data = output->template mutable_data<T>();

        bool is_float = std::is_same<T, float>::value;
        bool in_sb_half = std::is_same<Tsb, at::Half>::value;

        if (!HAS_CONVERT) {
          CAFFE_ENFORCE(is_float, "convert can be nullptr only if T is float");

          if (in_sb_half) {
            FusedNBitRowwiseQuantizedSBHalfToFloat(
                8,
                input_data,
                input_rows,
                input_columns,
                reinterpret_cast<float*>(output_data));
          } else {
            Fused8BitRowwiseQuantizedToFloat(
                input_data,
                input_rows,
                input_columns,
                reinterpret_cast<float*>(output_data));
          }
        } else {
          bool is_half = std::is_same<T, at::Half>::value;
          CAFFE_ENFORCE(is_half);

          vector<float> tmp(input_columns);
          for (size_t row = 0; row < input_rows; ++row) {
            if (in_sb_half) {
              FusedNBitRowwiseQuantizedSBHalfToFloat(
                  8,
                  input_data + row * input_columns,
                  1,
                  input_columns,
                  tmp.data());
            } else {
              Fused8BitRowwiseQuantizedToFloat(
                  input_data + row * input_columns, 1, input_columns, tmp.data());
            }
            convert(output_data + row * output_columns, tmp.data(), output_columns);
          }
        }

        return true;
        */
    }
}
