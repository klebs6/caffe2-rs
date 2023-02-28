crate::ix!();

use crate::{
    OperatorStorage,
    CPUContext,
    OperatorDef,
    Workspace
};

pub struct FloatToFusedNBitRowwiseQuantizedOp< const BIT_RATE: i32, T, ConvertFn, const Greedy: bool>
{
    storage: OperatorStorage,
    context: CPUContext,
    phantom: PhantomData<T>,
    phantomCFN: PhantomData<ConvertFn>,
}

input_tags!{
    FloatToFusedNBitRowwiseQuantizedOp {
        DataFloat
    }
}

output_tags!{
    FloatToFusedNBitRowwiseQuantizedOp {
        DataFusedScaleBias
    }
}

impl< const BIT_RATE: i32, T, ConvertFn, const Greedy: bool> 
FloatToFusedNBitRowwiseQuantizedOp<BIT_RATE,T,ConvertFn,Greedy> {
    
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

        CAFFE_ENFORCE_GT(input.dim(), 0, "Input's dimension must be at least 1");
        const auto input_rows = input.size_to_dim(input.dim() - 1);
        const auto input_columns = input.size(input.dim() - 1);
        static_assert(8 % BIT_RATE == 0, "BIT_RATE must divide 8");
        constexpr int NUM_ELEM_PER_BYTE = 8 / BIT_RATE;
        CAFFE_ENFORCE_EQ(
            input.dim(input.dim() - 1) % NUM_ELEM_PER_BYTE,
            0,
            "FloatToFused" + caffe2::to_string(BIT_RATE) +
                "BitRowwiseQuantizedOp only works for the number of "
                "columns a multiple of " +
                caffe2::to_string(NUM_ELEM_PER_BYTE));

        // The "fused" representation stores the scale and bias with the
        // row-wise quantized data in one tensor.
        // Since we represent the scale and bias in 16-bit float, we'll use the
        // last 4 bytes of each row for scale (2 bytes) and bias (2 bytes).
        // | ... quantized data ... | scale | bias |
        // |    number_of_columns   |  2B   |  2B  |
        auto output_dimensions = input.sizes().vec();
        output_dimensions[input.dim() - 1] = static_cast<std::int64_t>(
            (input_columns + NUM_ELEM_PER_BYTE - 1) / NUM_ELEM_PER_BYTE +
            2 * sizeof(at::Half));
        auto* output = Output(
            DATA_FUSED_SCALE_BIAS, output_dimensions, at::dtype<std::uint8_t>());

        const auto* input_data = input.template data<T>();
        auto* output_data = output->template mutable_data<std::uint8_t>();

        if (!GREEDY && std::is_same<T, float>::value) {
          // fast path
          CAFFE_ENFORCE(
              reinterpret_cast<void (*)(float*, const float*, std::size_t)>(
                  convert) == internal::convertfp32fp32,
              "When T == float, convert must be convertfp32fp32");
          FloatToFusedNBitRowwiseQuantizedSBHalf(
              BIT_RATE,
              reinterpret_cast<const float*>(input_data),
              input_rows,
              input_columns,
              output_data);
        } else {
          const auto output_columns = output->size(output->dim() - 1);

    #ifdef _OPENMP
          vector<float> tmp_vec(
              input_columns * (GREEDY ? omp_get_max_threads() : 1));
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

            std::uint8_t* output_row = output_data + row * output_columns;
            at::Half* output_row_scale = reinterpret_cast<at::Half*>(
                output_row +
                (input_columns + NUM_ELEM_PER_BYTE - 1) / NUM_ELEM_PER_BYTE);
            at::Half* output_row_bias = reinterpret_cast<at::Half*>(
                output_row +
                (input_columns + NUM_ELEM_PER_BYTE - 1) / NUM_ELEM_PER_BYTE +
                sizeof(at::Half));

            float Xmin = *std::min_element(tmp, tmp + input_columns);
            float Xmax = *std::max_element(tmp, tmp + input_columns);

            if (GREEDY) {
              internal::param_search_greedy(
                  tmp, input_columns, 200, 0.16, Xmin, Xmax, BIT_RATE);
            }

            // Round Xmin to fp16 to match with dequantization that will use fp16
            // for Xmin.
            Xmin = static_cast<at::Half>(Xmin);
            const float range = Xmax - Xmin;
            // Round scale to fp16 to match with dequantization that will use fp16
            // for scale.
            // Set scale to 1.0f for the corner case of Xmax == Xmin .
            // Any non-zero scale would work because during quantization
            // (X - Xmin) / scale will be 0 for all X unless scale is 0.
            at::Half scale = range == 0 ? 1.0f : range / ((1 << BIT_RATE) - 1);
            float inverse_scale = scale == 0 ? 1.0f : 1.0f / scale;
            if (scale == 0 || std::isinf(inverse_scale)) {
              // Corner case handling when Xmax == Xmin
              // Any scale would work because X - Xmin will be 0 for all X
              scale = 1.0f;
              inverse_scale = 1.0f;
            }

            *output_row_scale = scale;
            *output_row_bias = Xmin;

            for (int col = 0; col < input_columns; ++col) {
              float X = tmp[col];
              std::uint8_t quantized = std::max(
                  0,
                  std::min<int>(
                      std::lrintf((X - Xmin) * inverse_scale),
                      (1 << BIT_RATE) - 1));
              if (col % NUM_ELEM_PER_BYTE == 0) {
                output_row[col / NUM_ELEM_PER_BYTE] = quantized;
              } else {
                output_row[col / NUM_ELEM_PER_BYTE] |=
                    (quantized << ((col % NUM_ELEM_PER_BYTE) * BIT_RATE));
              }
            }
          }
        } // GREEDY || !std::is_same<T, float>::value

        return true;
        */
    }
}

///--------------------------------------------------
pub struct FusedNBitRowwiseQuantizedToFloatOp<const BIT_RATE: i32, T, ConvertFn> {
    storage:    OperatorStorage,
    context:    CPUContext,
    phantom:    PhantomData<T>,
    phantomCFN: PhantomData<ConvertFn>,
}

input_tags!{
    FusedNBitRowwiseQuantizedToFloatOp {
        DataFusedScaleBias
    }
}

output_tags!{
    FusedNBitRowwiseQuantizedToFloatOp {
        DataFloat
    }
}

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

#[inline] pub fn convertfp_32fp16(dst: *mut f16, src: *const f32, n: usize)  {
    
    todo!();
    /*
        for (size_t i = 0; i < N; i++) {
        dst[i] = src[i];
      }
    */
}

/**
  | Applies 4-bit row-wise quantization
  | by determining the range (maximum -
  | minimum) and offset (minimum value)
  | of each row in the input matrix, and then
  | scaling each element to an 4-bit number
  | between 0 and 15.
  | 
  | To later de-quantize values, the scale
  | (range / 15) and zero_point are stored
  | alongside the data. More precisely,
  | each row first has quantized values,
  | and then 2-byte fp16 scale and 2-byte
  | zero_offset.)
  |
  */
register_cpu_operator!{
    FloatToFused4BitRowwiseQuantized,
    FloatToFusedNBitRowwiseQuantizedOp<4, f32, internal::convertfp32fp32>
}

num_inputs!{FloatToFused4BitRowwiseQuantized, 1}

num_outputs!{FloatToFused4BitRowwiseQuantized, 1}

inputs!{FloatToFused4BitRowwiseQuantized, 
    0 => ("input", "Float32 input data")
}

outputs!{FloatToFused4BitRowwiseQuantized, 
    0 => ("output", "Fused scale, bias and quantized data")
}

tensor_inference_function!{FloatToFused4BitRowwiseQuantized, 
    |def: &OperatorDef, input: &Vec<TensorShape>| {
        todo!();
        /*
          vector<TensorShape> out;
          TensorShape X = in[0];
          // divide over 2 and round up, add 4 for the extra scale and bias
          X.set_dims(
              X.dims().size() - 1,
              (X.dims(X.dims().size() - 1) + 1) / 2 + 2 * sizeof(at::Half));
          out.push_back(std::move(X));
          out[0].set_data_type(TensorProto_DataType_UINT8);
          return out;
        */
    }
}

no_gradient!{FloatToFused4BitRowwiseQuantized}

/**
  | Applies 4-bit row-wise quantization
  | by determining the range (maximum -
  | minimum) and offset (minimum value)
  | of each row in the input matrix, and then
  | scaling each element to an 4-bit number
  | between 0 and 15.
  | 
  | To later de-quantize values, the scale
  | (range / 15) and zero_point are stored
  | alongside the data. More precisely,
  | each row first has quantized values,
  | and then 2-byte fp16 scale and 2-byte
  | zero_offset.)
  |
  */
register_cpu_operator!{
    HalfToFused4BitRowwiseQuantized,
    FloatToFusedNBitRowwiseQuantizedOp<4, f16, internal::convertfp16fp32>
}

num_inputs!{HalfToFused4BitRowwiseQuantized, 1}

num_outputs!{HalfToFused4BitRowwiseQuantized, 1}

inputs!{HalfToFused4BitRowwiseQuantized, 
    0 => ("input", "Float16 input data")
}

outputs!{HalfToFused4BitRowwiseQuantized, 
    0 => ("output", "Fused scale, bias and quantized data")
}

tensor_inference_function!{HalfToFused4BitRowwiseQuantized, 
    |def: &OperatorDef, input: &Vec<TensorShape>| {
        todo!();
        /*
          vector<TensorShape> out;
          TensorShape X = in[0];
          X.set_dims(
              X.dims().size() - 1,
              (X.dims(X.dims().size() - 1) + 1) / 2 + 2 * sizeof(at::Half));
          out.push_back(std::move(X));
          out[0].set_data_type(TensorProto_DataType_UINT8);
          return out;
        */
    }
}

no_gradient!{HalfToFused4BitRowwiseQuantized}

/**
  | De-quantizes the result of the
  | 
  | FloatToFused4BitRowwiseQuantized
  | operator.
  | 
  | The input is expected to first have quantized
  | values, then 2-byte fp16 scale and 1-byte
  | zero_offset. The output is a matrix
  | containing only the values, but de-quantized.
  | De-quantization is performed by multiplying
  | each value by its row's scale and zero_point
  | parameters.
  | 
  | The de-quantized values will thus not
  | be exactly equal to the original, un-quantized
  | floating point values.
  |
  */
register_cpu_operator!{
    Fused4BitRowwiseQuantizedToFloat,
    FusedNBitRowwiseQuantizedToFloatOp<4, f32, internal::convertfp32fp32>
}

num_inputs!{Fused4BitRowwiseQuantizedToFloat, 1}

num_outputs!{Fused4BitRowwiseQuantizedToFloat, 1}

inputs!{Fused4BitRowwiseQuantizedToFloat, 
    0 => ("scale_bias_quantized_input", "Fused scale, bias and quantized data")
}

outputs!{Fused4BitRowwiseQuantizedToFloat, 
    0 => ("float_output", "Float32 data")
}

tensor_inference_function!{Fused4BitRowwiseQuantizedToFloat, 
    |def: &OperatorDef, input: &Vec<TensorShape>| {
        todo!();
        /*
          vector<TensorShape> out;
          TensorShape X = in[0];
          X.set_dims(
              X.dims().size() - 1,
              (X.dims(X.dims().size() - 1) - 2 * sizeof(at::Half)) * 2);
          out.push_back(std::move(X));
          out[0].set_data_type(TensorProto_DataType_FLOAT);
          return out;
        */
    }
}

no_gradient!{Fused4BitRowwiseQuantizedToFloat}

/**
  | De-quantizes the result of the
  | 
  | FloatToFused4BitRowwiseQuantized
  | operator.
  | 
  | The input is expected to first have quantized
  | values, then 2-byte fp16 scale and 1-byte
  | zero_offset. The output is a matrix
  | containing only the values, but de-quantized.
  | 
  | De-quantization is performed by multiplying
  | each value by its row's scale and zero_point
  | parameters.
  | 
  | The de-quantized values will thus not
  | be exactly equal to the original, un-quantized
  | floating point values.
  |
  */
register_cpu_operator!{
    Fused4BitRowwiseQuantizedToHalf,
    FusedNBitRowwiseQuantizedToFloatOp<4, f16, internal::convertfp32fp16>
}

num_inputs!{Fused4BitRowwiseQuantizedToHalf, 1}

num_outputs!{Fused4BitRowwiseQuantizedToHalf, 1}

inputs!{Fused4BitRowwiseQuantizedToHalf, 
    0 => ("scale_bias_quantized_input", "Fused scale, bias and quantized data")
}

outputs!{Fused4BitRowwiseQuantizedToHalf, 
    0 => ("float16_output", "Float16 data")
}

tensor_inference_function!{Fused4BitRowwiseQuantizedToHalf, 
    |def: &OperatorDef, input: &Vec<TensorShape>| {

        todo!();
        /*
          vector<TensorShape> out;
          TensorShape X = in[0];
          X.set_dims(
              X.dims().size() - 1,
              (X.dims(X.dims().size() - 1) - 2 * sizeof(at::Half)) * 2);
          out.push_back(std::move(X));
          out[0].set_data_type(TensorProto_DataType_FLOAT16);
          return out;
        */
    }
}

no_gradient!{Fused4BitRowwiseQuantizedToHalf}

///----------------------------------
register_cpu_operator_with_engine!{
    FloatToFused4BitRowwiseQuantized,
    GREEDY,
    FloatToFusedNBitRowwiseQuantizedOp<
    { 4 },
    f32,
    convertfp32fp32,
    Greedy>
}

register_cpu_operator_with_engine!{
    HalfToFused4BitRowwiseQuantized,
    GREEDY,
    FloatToFusedNBitRowwiseQuantizedOp<
    { 4 },
        f16,
        convertfp16fp32,
        Greedy>
}

/**
  | Applies 2-bit row-wise quantization
  | by determining the range (maximum -
  | minimum) and offset (minimum value)
  | of each row in the input matrix, and then
  | scaling each element to an 2-bit number
  | between 0 and 3.
  | 
  | To later de-quantize values, the scale
  | (range / 3) and zero_point are stored
  | alongside the data.
  | 
  | More precisely, each row first has quantized
  | values, and then 2-byte fp16 scale and
  | 2-byte zero_offset.)
  |
  */
register_cpu_operator!{
    FloatToFused2BitRowwiseQuantized,
    FloatToFusedNBitRowwiseQuantizedOp<2, f32, convertfp32fp32>
}

num_inputs!{FloatToFused2BitRowwiseQuantized, 1}

num_outputs!{FloatToFused2BitRowwiseQuantized, 1}

inputs!{FloatToFused2BitRowwiseQuantized, 
    0 => ("input", "Float32 input data")
}

outputs!{FloatToFused2BitRowwiseQuantized, 
    0 => ("output", "Fused scale, bias and quantized data")
}

tensor_inference_function!{FloatToFused2BitRowwiseQuantized, 
    |def: &OperatorDef, input: &Vec<TensorShape>| {
        todo!();
        /*
          vector<TensorShape> out;
          TensorShape X = in[0];
          // divide over 4 and round up, add 4 for the extra scale and bias
          X.set_dims(
              X.dims().size() - 1,
              (X.dims(X.dims().size() - 1) + 3) / 4 + 2 * sizeof(at::Half));
          out.push_back(std::move(X));
          out[0].set_data_type(TensorProto_DataType_UINT8);
          return out;
        */
    }
}

no_gradient!{FloatToFused2BitRowwiseQuantized}

register_cpu_operator_with_engine!{
    FloatToFused2BitRowwiseQuantized,
    GREEDY,
    FloatToFusedNBitRowwiseQuantizedOp<
    {2},
    f32,
    internal::convertfp32fp32,
    Greedy>
}

/**
  | Applies 2-bit row-wise quantization
  | by determining the range (maximum -
  | minimum) and offset (minimum value)
  | of each row in the input matrix, and then
  | scaling each element to an 2-bit number
  | between 0 and 3.
  | 
  | To later de-quantize values, the scale
  | (range / 3) and zero_point are stored
  | alongside the data.
  | 
  | More precisely, each row first has quantized
  | values, and then 2-byte fp16 scale and
  | 2-byte zero_offset.)
  |
  */
register_cpu_operator!{
    HalfToFused2BitRowwiseQuantized,
    FloatToFusedNBitRowwiseQuantizedOp<2, f16, internal::convertfp16fp32>
}

num_inputs!{HalfToFused2BitRowwiseQuantized, 1}

num_outputs!{HalfToFused2BitRowwiseQuantized, 1}

inputs!{HalfToFused2BitRowwiseQuantized, 
    0 => ("input", "Float16 input data")
}

outputs!{HalfToFused2BitRowwiseQuantized, 
    0 => ("output", "Fused scale, bias and quantized data")
}

tensor_inference_function!{HalfToFused2BitRowwiseQuantized, 
    |def: &OperatorDef, input: &Vec<TensorShape>| {
        /*
          vector<TensorShape> out;
          TensorShape X = in[0];
          X.set_dims(
              X.dims().size() - 1,
              (X.dims(X.dims().size() - 1) + 3) / 4 + 2 * sizeof(at::Half));
          out.push_back(std::move(X));
          out[0].set_data_type(TensorProto_DataType_UINT8);
          return out;
        */
    }
}

no_gradient!{HalfToFused2BitRowwiseQuantized}

register_cpu_operator_with_engine!{
    HalfToFused2BitRowwiseQuantized,
    GREEDY,
    FloatToFusedNBitRowwiseQuantizedOp<
    { 2 },
    f16,
    convertfp16fp32,
    Greedy>
}

/**
  | De-quantizes the result of the
  | 
  | FloatToFused2BitRowwiseQuantized
  | operator.
  | 
  | The input is expected to first have quantized
  | values, then 2-byte fp16 scale and 1-byte
  | zero_offset.
  | 
  | The output is a matrix containing only
  | the values, but de-quantized.
  | 
  | De-quantization is performed by multiplying
  | each value by its row's scale and zero_point
  | parameters.
  | 
  | The de-quantized values will thus not
  | be exactly equal to the original, un-quantized
  | floating point values.
  |
  */
register_cpu_operator!{
    Fused2BitRowwiseQuantizedToFloat,
    FusedNBitRowwiseQuantizedToFloatOp<2, f32, internal::convertfp32fp32>
}

num_inputs!{Fused2BitRowwiseQuantizedToFloat, 1}

num_outputs!{Fused2BitRowwiseQuantizedToFloat, 1}

inputs!{Fused2BitRowwiseQuantizedToFloat, 
    0 => ("scale_bias_quantized_input", "Fused scale, bias and quantized data")
}

outputs!{Fused2BitRowwiseQuantizedToFloat, 
    0 => ("float_output", "Float32 data")
}

tensor_inference_function!{Fused2BitRowwiseQuantizedToFloat, 
    |def: &OperatorDef, input: &Vec<TensorShape>| {
        todo!();
        /*
          vector<TensorShape> out;
          TensorShape X = in[0];
          X.set_dims(
              X.dims().size() - 1,
              (X.dims(X.dims().size() - 1) - 2 * sizeof(at::Half)) * 4);
          out.push_back(std::move(X));
          out[0].set_data_type(TensorProto_DataType_FLOAT);
          return out;
        */
    }
}

no_gradient!{Fused2BitRowwiseQuantizedToFloat}

/**
  | De-quantizes the result of the
  | 
  | FloatToFused2BitRowwiseQuantized
  | operator.
  | 
  | The input is expected to first have quantized
  | values, then 2-byte fp16 scale and 1-byte
  | zero_offset. The output is a matrix
  | containing only the values, but de-quantized.
  | 
  | De-quantization is performed by multiplying
  | each value by its row's scale and zero_point
  | parameters.
  | 
  | The de-quantized values will thus not
  | be exactly equal to the original, un-quantized
  | floating point values.
  |
  */
register_cpu_operator!{
    Fused2BitRowwiseQuantizedToHalf,
    FusedNBitRowwiseQuantizedToFloatOp<2, f16, internal::convertfp32fp16>
}

num_inputs!{Fused2BitRowwiseQuantizedToHalf, 1}

num_outputs!{Fused2BitRowwiseQuantizedToHalf, 1}

inputs!{Fused2BitRowwiseQuantizedToHalf, 
    0 => ("scale_bias_quantized_input", "Fused scale, bias and quantized data")
}

outputs!{Fused2BitRowwiseQuantizedToHalf, 
    0 => ("float16_output", "Float16 data")
}

tensor_inference_function!{Fused2BitRowwiseQuantizedToHalf, 
    |def: &OperatorDef, input: &Vec<TensorShape>| {
        todo!();
        /*
          vector<TensorShape> out;
          TensorShape X = in[0];
          X.set_dims(
              X.dims().size() - 1,
              (X.dims(X.dims().size() - 1) - 2 * sizeof(at::Half)) * 4);
          out.push_back(std::move(X));
          out[0].set_data_type(TensorProto_DataType_FLOAT16);
          return out;
        */
    }
}

no_gradient!{Fused2BitRowwiseQuantizedToHalf}
