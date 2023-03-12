crate::ix!();

/**
  | Fake 2/4 bit quantization
  | 
  | Creates a 2/4bit rowwise quantized
  | blob with scales and biases in fp16
  | 
  | The storage format is 8 bit rowwise with
  | scales and biases in fp32
  |
  */
pub struct FloatToFusedNBitFakeRowwiseQuantizedOp<Context,
const BIT_RATE: i32, T, ConvertFn, const GREEDY: bool>
{
    storage: OperatorStorage,
    context: Context,
    phantom: PhantomData<T>,
    phantomCFN: PhantomData<ConvertFn>,
}

input_tags!{
    FloatToFusedNBitFakeRowwiseQuantizedOp {
        DataFloat
    }
}

output_tags!{
    FloatToFusedNBitFakeRowwiseQuantizedOp {
        // INT8 suffix because this is a fake quantization operator whose output
        // type is always 8-bit regardless of BIT_RATE.
        DataFusedScaleBiasInt8
    }
}

impl<Context, const BIT_RATE: i32, T, ConvertFn, const GREEDY: bool>
FloatToFusedNBitFakeRowwiseQuantizedOp<Context, BIT_RATE, T, ConvertFn, GREEDY>
{
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

        const auto& input = Input(DATA_FLOAT);

        const auto input_rows = input.size(0);
        const auto input_columns = input.size(1);
        CAFFE_ENFORCE_EQ(input.dim(), 2, "Expect input to be a matrix");

        const std::vector<int64_t> output_dimensions = {input_rows,
                                                        input_columns + 8};
        auto* output = Output(
            DATA_FUSED_SCALE_BIAS_INT8, output_dimensions, at::dtype<uint8_t>());

        const auto* input_data = input.template data<T>();
        auto* output_data = output->template mutable_data<uint8_t>();
        const auto output_columns = output->size(1);

        if (!std::is_same<T, float>::value && !std::is_same<T, at::Half>::value) {
          CAFFE_THROW("Unsupported data type");
        }

        bool use_openmp = GREEDY;
    #ifdef _OPENMP
        vector<float> tmp_vec(input_columns * (GREEDY ? omp_get_max_threads() : 1));
    #else
        vector<float> tmp_vec(input_columns);
    #endif

    #pragma omp parallel for if (GREEDY)
        for (int row = 0; row < input_rows; ++row) {
          float* tmp = tmp_vec.data();
    #ifdef _OPENMP
          if (GREEDY) {
            tmp = &tmp_vec[omp_get_thread_num() * input_columns];
          }
    #endif
          convert(tmp, input_data + row * input_columns, input_columns);
          uint8_t* output_row = output_data + row * output_columns;
          float* output_row_scale_bias =
              reinterpret_cast<float*>(output_row + input_columns);

          float minimum_element = *std::min_element(tmp, tmp + input_columns);
          float maximum_element = *std::max_element(tmp, tmp + input_columns);

          if (GREEDY) {
            internal::param_search_greedy(
                tmp,
                input_columns,
                200,
                0.16,
                minimum_element,
                maximum_element,
                BIT_RATE);
          }

          minimum_element = static_cast<at::Half>(minimum_element);
          const float range = maximum_element - minimum_element;

          const float scale = range == 0
              ? 1.0f
              : static_cast<float>(static_cast<at::Half>(
                    range / static_cast<float>((1 << BIT_RATE) - 1)));
          const float inverse_scale = 1.0f / scale;

          output_row_scale_bias[0] = scale;
          output_row_scale_bias[1] = minimum_element;

          for (size_t col = 0; col < input_columns; ++col) {
            output_row[col] = std::max(
                0,
                std::min<int>(
                    std::lrintf((tmp[col] - minimum_element) * inverse_scale),
                    (1 << BIT_RATE) - 1));
          }
        }

        return true;
        */
    }
}
