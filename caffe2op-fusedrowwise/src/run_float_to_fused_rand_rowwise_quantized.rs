crate::ix!();

impl<T,Context> FloatToFusedRandRowwiseQuantizedOp<T,Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            bitwidth_(OperatorStorage::GetSingleArgument<int32_t>("bitwidth", 8)),
            random_(OperatorStorage::GetSingleArgument<bool>("random", true)) 

        CAFFE_ENFORCE(
            bitwidth_ == 1 || bitwidth_ == 2 || bitwidth_ == 4 || bitwidth_ == 8,
            "Unsupported bitwidth");
        if (random_) {
    #ifdef FUSED_ROWWISE_RANDOM_QUANTIZATION_USE_MKL
          int status = vslNewStream(
              &vslStream_,
              VSL_BRNG_MT19937,
              std::chrono::system_clock::now().time_since_epoch().count());
          if (status != VSL_STATUS_OK) {
            LOG(WARNING) << "vslNewStream returns " << status;
          }
    #else
          gen_.seed(std::chrono::system_clock::now().time_since_epoch().count());
          dis_.reset(new std::uniform_real_distribution<float>(0.0f, 1.0f));
    #endif
        }
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            CAFFE_ENFORCE(IS_LITTLE_ENDIAN, "Unsupported endianness");

      const auto& input = Input(DATA_FLOAT);

      CAFFE_ENFORCE_EQ(
          input.dim(),
          2,
          "Expect input to be a matrix. Reshape the input tensor to a matrix for usage.");

      const auto input_rows = input.size(0);
      const auto input_columns = input.size(1);

      // The "fused" representation stores the [bitwidth][tail][min][max]
      // with the row-wise quantized data in one tensor. Since we store 8/bitwidth
      // quantized data in one byte, the last buckets of some bytes may have
      // unused bits. There are totally tail buckets are unused.
      // We encode *bitwidth* and *tail* at the beginning of
      // each row, following by 32-bit floating data respresenting min and max.
      // | bitwidth | tail | min | max | ... int8 data ... |
      // |    1B    |  1B  |  4B |  4B | ...output_data....|
      // In output_data: the b-th bucket of the i-th byte stores
      // the i-th data of the b-th segment of input row
      size_t data_per_byte = 8 / bitwidth_;
      // How many bytes in the output
      size_t segment_size = (input_columns + data_per_byte - 1) / data_per_byte;
      const std::vector<int64_t> output_dimensions = {
          input_rows, 10 + static_cast<int64_t>(segment_size)};
      auto* output =
          Output(DATA_FUSED_QUANTIZED, output_dimensions, at::dtype<uint8_t>());

      const auto* input_data = input.template data<float>();
      auto* output_data = output->template mutable_data<uint8_t>();
      const size_t output_columns = static_cast<size_t>(output->size(1));
      memset(output_data, 0, output->numel());

      if (random_) {
        random_buffer_.resize(input_columns);
      }

      for (size_t row = 0; row < input_rows; ++row) {
        if (random_) {
    #ifdef FUSED_ROWWISE_RANDOM_QUANTIZATION_USE_MKL
          int status = vsRngUniform(
              VSL_RNG_METHOD_UNIFORM_STD,
              vslStream_,
              input_columns,
              random_buffer_.data(),
              0.0f,
              1.0f);
          if (status != VSL_ERROR_OK) {
            LOG(WARNING) << "vsRngUniform returns " << status;
          }
    #else
          for (int i = 0; i < input_columns; ++i) {
            random_buffer_[i] = (*dis_)(gen_);
          }
    #endif
        }

        math::quantize_and_compress(
            input_data + row * input_columns,
            output_data + row * output_columns,
            input_columns,
            bitwidth_,
            random_,
            random_buffer_.data());
      }

      return true;
        */
    }
}

