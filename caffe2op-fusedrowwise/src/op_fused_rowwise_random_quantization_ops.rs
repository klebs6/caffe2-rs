crate::ix!();

/**
  | Applies row-wise stochastic/random
  | quantization by determining the range
  | of each row in the input matrix, and then
  | quantize each element to one of two closest
  | discrete levels by randomly drawing
  | 
  | Bernoulli distribution.
  | 
  | The method is extended from TernGrad
  | [1], which randomly quantizes gradients
  | to three levels to reduce communication
  | in distributed training.
  | 
  | The format of each row (x) in the output
  | matrix is [bitwidth][tail][min][max][data]:
  | 
  | bitwidth[1 Byte]: bitwidth per data
  | [1, 2, 4 or 8];
  | 
  | tail[1 Byte]: the number of unused buckets
  | [1-8] (One byte is split to 8/bitwidth
  | buckets and each bucket stores one low-precision
  | data in bitwidth bits);
  | 
  | min[4 Bytes]: the minimum floating
  | value min(x);
  | 
  | max[4 Bytes]: the maximum floating
  | value max(x);
  | 
  | data: quantized data.
  | 
  | The quantization is uniform with levels
  | q = min + (max-min)/(2^bitwidth - 1)*[0:1:2^bitwidth].
  | 
  | During stochastic/random quantization
  | x'=Quantize(x), for q_j < x_i <= q_{j+1},
  | we draw quantization x'_i from Bernoulli
  | distributions with
  | 
  | P(x'_i = q_{j+1}) = (x_i - q_j)/(q_{j+1}
  | - q_j), and
  | 
  | P(x'_i = q_j) = (q_{j+1} - x_i)/(q_{j+1}
  | - q_j) where x'_i is the quantized value
  | of x_i.
  | 
  | [1] proved E{x'_i}=x_i, which is an
  | unbiased approximation.
  | 
  | More details are in the paper.
  | 
  | For example, suppose targeted bitwidth
  | = 2 and x = [0.3, -1.4, -0.6, 0.9, 1.0],
  | then tail = 3, min = -1.4, max = 1.0 and
  | q = [-1.4, -0.6, 0.2, 1.0]. x_1 = 0.3 will
  | be quantized to x'_1 = 0.2 with probability
  | 7/8 and to x'_1 = 1.0 with probability
  | 1/8.
  | 
  | The storage format of quantized data
  | is: [x'_1|x'_3|x'_5|xxx]-[x'_2|x'_4|xxx|xxx].
  | 
  | In general, a input row is split to multiple
  | segments.
  | 
  | One segment is a continuous subarray
  | of the row, and its length is the number
  | of bytes storing quantized data in the
  | output matrix.
  | 
  | The b-th bucket of the i-th byte stores
  | the i-th data of the b-th segment of input
  | row.
  | 
  | [1] Wen, Wei, Cong Xu, Feng Yan, Chunpeng
  | Wu,
  | 
  | Yandan Wang, Yiran Chen, and Hai Li.
  | "Terngrad:
  | 
  | Ternary gradients to reduce communication
  | in distributed deep learning." In Advances
  | in Neural
  | 
  | Information Processing Systems, pp.
  | 1508-1518. 2017.
  |
  */
  #[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct FloatToFusedRandRowwiseQuantizedOp<T,Context> {

    storage:       OperatorStorage,
    context:       Context,

    bitwidth:      usize, //8

    random:        bool, //true
    random_buffer: Vec<f32>,

    #[cfg(fused_rowwise_random_quantization_use_mkl)]
    vsl_stream:    VSLStreamStatePtr,

    #[cfg(not(fused_rowwise_random_quantization_use_mkl))]
    dis:           Uniform,

    #[cfg(not(fused_rowwise_random_quantization_use_mkl))]
    gen:           StdRng,

    phantom:       PhantomData<T>,
}

register_cpu_operator![
    FloatToFusedRandRowwiseQuantized,
    FloatToFusedRandRowwiseQuantizedOp<CPUContext>
];

num_inputs!{FloatToFusedRandRowwiseQuantized, 1}

num_outputs!{FloatToFusedRandRowwiseQuantized, 1}

inputs!{FloatToFusedRandRowwiseQuantized, 
    0 => ("input", "Float32 input data")
}

outputs!{FloatToFusedRandRowwiseQuantized, 
    0 => ("output", "Fused bitwidth, tail, min, max and quantized data")
}

args!{FloatToFusedRandRowwiseQuantized, 
    0 => ("bitwidth", "How many bits to quantize per data (defaults to 8)."),
    1 => ("random", "random or not (True). False is set up for unittest.")
}

tensor_inference_function!{FloatToFusedRandRowwiseQuantized, /* [](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      ArgumentHelper helper(def);
      auto bitwidth = helper.GetSingleArgument<int32_t>("bitwidth", 8);
      size_t data_per_byte = 8 / bitwidth;
      vector<TensorShape> out;
      TensorShape X = in[0];
      X.set_dims(1, 10 + (X.dims(1) + data_per_byte - 1) / data_per_byte);
      out.push_back(std::move(X));
      out[0].set_data_type(TensorProto_DataType_UINT8);
      return out;
    } */}

no_gradient!{FloatToFusedRandRowwiseQuantized}

input_tags!{
    FloatToFusedRandRowwiseQuantizedOp {
        DataFloat
    }
}

output_tags!{
    FloatToFusedRandRowwiseQuantizedOp {
        DataFusedQuantized
    }
}

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

impl<T, Context> Drop for FloatToFusedRandRowwiseQuantizedOp<T, Context> {
    fn drop(&mut self) {
        todo!();
        /*
        if (random_) {
    #ifdef FUSED_ROWWISE_RANDOM_QUANTIZATION_USE_MKL
          int status = vslDeleteStream(&vslStream_);
          if (status != VSL_STATUS_OK) {
            LOG(WARNING) << "vslDeleteStream returns " << status;
          }
    #endif
        }
        */
    }
}

/**
  | De-quantizes the result of the
  | FloatToFusedRandRowwiseQuantized
  | operator.
  | 
  | Refer FloatToFusedRandRowwiseQuantized
  | operator for details.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct FusedRandRowwiseQuantizedToFloatOp<Context> {

    storage: OperatorStorage,
    context: Context,
}

register_cpu_operator!{
    FusedRandRowwiseQuantizedToFloat,
    FusedRandRowwiseQuantizedToFloatOp<CPUContext>
}

num_inputs!{FusedRandRowwiseQuantizedToFloat, 1}

num_outputs!{FusedRandRowwiseQuantizedToFloat, 1}

inputs!{FusedRandRowwiseQuantizedToFloat, 
    0 => ("quantized_input", "Fused bitwidth, tail, min, max and quantized data")
}

outputs!{FusedRandRowwiseQuantizedToFloat, 
    0 => ("float_input", "Float32 data")
}

tensor_inference_function!{FusedRandRowwiseQuantizedToFloat, 
    |def: &OperatorDef, input: &Vec<TensorShape>| {
        todo!();
        /*
          vector<TensorShape> out;
          for (int i = 0; i < def.output_size(); i++) {
            TensorShape ts;
            ts.set_unknown_shape(true);
            ts.set_data_type(TensorProto_DataType_FLOAT);
            out.push_back(ts);
          }
          return out;
        */
    }
}

no_gradient!{FusedRandRowwiseQuantizedToFloat}

input_tags!{
    FusedRandRowwiseQuantizedToFloatOp {
        DataFusedQuantized
    }
}

output_tags!{
    FusedRandRowwiseQuantizedToFloatOp {
        DataFloat
    }
}

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

