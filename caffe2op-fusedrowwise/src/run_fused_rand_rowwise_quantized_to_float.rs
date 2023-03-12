crate::ix!();

impl<Context> FusedRandRowwiseQuantizedToFloatOp<Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            CAFFE_ENFORCE(IS_LITTLE_ENDIAN, "Unsupported endianness");

      const auto& input = Input(DATA_FUSED_QUANTIZED);

      CAFFE_ENFORCE_EQ(input.dim(), 2, "Expect input to be a matrix.");
      CAFFE_ENFORCE_GE(
          input.numel(),
          4,
          "Expect input to have size greater than or equal to 4.");

      const auto input_rows = input.size(0);
      const auto input_columns = input.size(1);
      const auto* input_data = input.template data<uint8_t>();
      const size_t bitwidth = input_data[0];
      CAFFE_ENFORCE(
          bitwidth == 1 || bitwidth == 2 || bitwidth == 4 || bitwidth == 8,
          "Unsupported bitwidth");
      const size_t tail = input_data[1];
      const size_t output_columns = (input_columns - 10) * (8 / bitwidth) - tail;
      const std::vector<int64_t> output_dimensions = {
          input_rows, static_cast<int64_t>(output_columns)};
      auto* output = Output(DATA_FLOAT, output_dimensions, at::dtype<float>());
      auto* output_data = output->template mutable_data<float>();
      for (size_t row = 0; row < input_rows; ++row) {
        math::decompress_and_dequantize(
            input_data + row * input_columns,
            output_data + row * output_columns,
            input_columns);
      }

      return true;
        */
    }
}
