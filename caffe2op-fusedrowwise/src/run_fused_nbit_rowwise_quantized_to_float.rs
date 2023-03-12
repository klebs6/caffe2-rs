crate::ix!();

impl<const BIT_RATE: i32, T, ConvertFn> FusedNBitRowwiseQuantizedToFloatOp<BIT_RATE, T, ConvertFn> {
    
    pub fn new(def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator<CPUContext>(def, ws)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            CAFFE_ENFORCE(internal::is_little_endian(), "Unsupported endianness");

        const auto& input = Input(DATA_FUSED_SCALE_BIAS);

        CAFFE_ENFORCE_GT(input.dim(), 0, "Input's dimension must be at least 1");
        const auto input_rows = input.size_to_dim(input.dim() - 1);
        const auto input_columns = input.size(input.dim() - 1);

        static_assert(8 % BIT_RATE == 0, "BIT_RATE must divide 8");
        constexpr int NUM_ELEM_PER_BYTE = 8 / BIT_RATE;

        // The last 4 bytes per row are two fp16 scale and bias.
        // The rest of input_columns is the number of values in the original row.
        auto output_dimensions = input.sizes().vec();
        output_dimensions[input.dim() - 1] =
            static_cast<std::int64_t>(input_columns - 2 * sizeof(at::Half)) *
            NUM_ELEM_PER_BYTE;
        auto* output = Output(DATA_FLOAT, output_dimensions, at::dtype<T>());
        const auto output_columns = output->size(output->dim() - 1);

        const auto* input_data = input.template data<std::uint8_t>();
        T* output_data = output->template mutable_data<T>();

        if (std::is_same<T, float>::value) {
          // fast path
          CAFFE_ENFORCE(
              reinterpret_cast<void (*)(float*, const float*, std::size_t)>(
                  convert) == internal::convertfp32fp32,
              "When T == float, convert must be convertfp32fp32");
          FusedNBitRowwiseQuantizedSBHalfToFloat(
              BIT_RATE,
              input_data,
              input_rows,
              input_columns,
              reinterpret_cast<float*>(output_data));
        } else {
          std::vector<float> tmp(output_columns);

          for (size_t row = 0; row < input_rows; ++row) {
            const std::uint8_t* input_row = input_data + row * input_columns;
            float scale = *reinterpret_cast<const at::Half*>(
                input_row +
                (output_columns + NUM_ELEM_PER_BYTE - 1) / NUM_ELEM_PER_BYTE);
            float bias = *reinterpret_cast<const at::Half*>(
                input_row +
                (output_columns + NUM_ELEM_PER_BYTE - 1) / NUM_ELEM_PER_BYTE +
                sizeof(at::Half));

            for (int col = 0; col < output_columns; ++col) {
              std::uint8_t quantized = input_row[col / NUM_ELEM_PER_BYTE];
              quantized >>= (col % NUM_ELEM_PER_BYTE) * BIT_RATE;
              quantized &= (1 << BIT_RATE) - 1;
              tmp[col] = scale * quantized + bias;
            }

            convert(output_data + row * output_columns, tmp.data(), output_columns);
          }
        }

        return true;
        */
    }
}
