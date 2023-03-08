crate::ix!();

pub type ConvertFnType<T> = fn(
    dst: *mut f32, 
    src: *const T, 
    N: libc::size_t) -> ();

/**
  | Applies 8-bit row-wise quantization
  | by determining the range (maximum -
  | minimum) and offset (minimum value)
  | of each row in the input matrix, and then
  | scaling each element to an 8-bit number
  | between 0 and 255.
  | 
  | To later de-quantize values, the scale
  | (range / 255) and offset (bias) are stored
  | alongside the data.
  | 
  | More precisely, each row contains int8
  | elements for each quantized element,
  | and the last 8 bytes of each row in the
  | output matrix are a float storing the
  | scale followed by another float containing
  | the scale.
  | 
  | For N-dimensional input tensor, the
  | first N-1 dimensions are interpreted
  | as rows and the last dimension is interpreted
  | as a column.
  | 
  | For example, an input tensor with dimension
  | 5x2x4 is interpreted as 10 rows and 4
  | columns.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct FloatToFused8BitRowwiseQuantizedOp<T, TypeForScaleAndBias, ConvertFn, const HAS_CONVERT: bool, Context> {

    storage:                    OperatorStorage,
    context:                    Context,

    phantom:                    PhantomData<T>,
    phantomCFN:                 PhantomData<ConvertFn>,
    phantomTypeForScaleAndBias: PhantomData<TypeForScaleAndBias>,
}

num_inputs!{FloatToFused8BitRowwiseQuantized, 1}

num_outputs!{FloatToFused8BitRowwiseQuantized, 1}

inputs!{FloatToFused8BitRowwiseQuantized, 
    0 => ("input", "Float32 input data")
}

outputs!{FloatToFused8BitRowwiseQuantized, 
    0 => ("output", "Fused scale, bias and quantized data")
}

tensor_inference_function!{FloatToFused8BitRowwiseQuantized, /* [](const OperatorDef& /* def */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out;
      TensorShape X = in[0];
      X.set_dims(
          X.dims().size() - 1, X.dims(X.dims().size() - 1) + 2 * sizeof(float));
      out.push_back(std::move(X));
      out[0].set_data_type(TensorProto_DataType_UINT8);
      return out;
    } */
}

no_gradient!{FloatToFused8BitRowwiseQuantized}

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

input_tags!{
    FloatToFused8BitRowwiseQuantizedOp {
        DataFloat
    }
}

output_tags!{
    FloatToFused8BitRowwiseQuantizedOp {
        DataFusedScaleBiasInt8
    }
}

///------------------------------------
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct Fused8BitRowwiseQuantizedToFloatOp<T, TypeForScaleAndBias, ConvertFn, const HAS_CONVERT: bool, Context> {

    storage:                    OperatorStorage,
    context:                    Context,

    phantom:                    PhantomData<T>,
    phantomCFN:                 PhantomData<ConvertFn>,
    phantomTypeForScaleAndBias: PhantomData<TypeForScaleAndBias>,
}

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

input_tags!{
    Fused8BitRowwiseQuantizedToFloatOp {
        DataFusedScaleBiasInt8
    }
}

output_tags!{
    Fused8BitRowwiseQuantizedToFloatOp {
        DataFloat
    }
}

#[inline] pub fn convertfp_16fp32(dst: *mut f32, src: *const f16, n: usize)  {
    
    todo!();
    /*
        for (size_t i = 0; i < N; i++) {
        dst[i] = src[i];
      }
    */
}

#[inline] pub fn convertfp_32fp16(dst: *mut f16, src: *const f32, n: usize)  {
    
    todo!();
    /*
        for (size_t i = 0; i < N; i++) {
        dst[i] = src[i];
      }
    */
}

register_cpu_operator!{
    FloatToFused8BitRowwiseQuantized,
    FloatToFused8BitRowwiseQuantizedOp<
        f32,
        f32,
        nullptr,
        false,
        CPUContext>
}

/**
  | Applies 8-bit row-wise quantization
  | by determining the range (maximum -
  | minimum) and offset (minimum value)
  | of each row in the input matrix, and then
  | scaling each element to an 8-bit number
  | between 0 and 255.
  | 
  | To later de-quantize values, the scale
  | (range / 255) and offset (bias) are stored
  | alongside the data.
  | 
  | More precisely, each row contains int8
  | elements for each quantized element,
  | and the last 4 bytes of each row in the
  | output matrix are a half float storing
  | the scale followed by another half float
  | containing the scale.)
  |
  */
register_cpu_operator!{
    FloatToFused8BitRowwiseQuantizedHalfScaleBias,
    FloatToFused8BitRowwiseQuantizedOp<
        f32,
        f16,
        nullptr,
        false,
        CPUContext>
}

num_inputs!{FloatToFused8BitRowwiseQuantizedHalfScaleBias, 1}

num_outputs!{FloatToFused8BitRowwiseQuantizedHalfScaleBias, 1}

inputs!{FloatToFused8BitRowwiseQuantizedHalfScaleBias, 
    0 => ("input", "Float32 input data")
}

outputs!{FloatToFused8BitRowwiseQuantizedHalfScaleBias, 
    0 => ("output", "Fused scale, bias and quantized data")
}

tensor_inference_function!{FloatToFused8BitRowwiseQuantizedHalfScaleBias, /* [](const OperatorDef& /* def */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out;
      TensorShape X = in[0];
      X.set_dims(
          X.dims().size() - 1,
          X.dims(X.dims().size() - 1) + 2 * sizeof(at::Half));
      out.push_back(std::move(X));
      out[0].set_data_type(TensorProto_DataType_UINT8);
      return out;
    } */}

/**
  | Applies 8-bit row-wise quantization
  | by determining the range (maximum -
  | minimum) and offset (minimum value)
  | of each row in the input matrix, and then
  | scaling each element to an 8-bit number
  | between 0 and 255.
  | 
  | To later de-quantize values, the scale
  | (range / 255) and offset (bias) are stored
  | alongside the data.
  | 
  | More precisely, each row contains int8
  | elements for each quantized element,
  | and the last 8 bytes of each row in the
  | output matrix are a float storing the
  | scale followed by another float containing
  | the scale.)
  |
  */
register_cpu_operator!{
    HalfFloatToFused8BitRowwiseQuantized,
    FloatToFused8BitRowwiseQuantizedOp<
        f16,
        f32,
        convertfp16fp32,
        true,
        CPUContext>
}

num_inputs!{HalfFloatToFused8BitRowwiseQuantized, 1}

num_outputs!{HalfFloatToFused8BitRowwiseQuantized, 1}

inputs!{HalfFloatToFused8BitRowwiseQuantized, 
    0 => ("input", "Float16 input data")
}

outputs!{HalfFloatToFused8BitRowwiseQuantized, 
    0 => ("output", "Fused scale, bias and quantized data")
}

tensor_inference_function!{HalfFloatToFused8BitRowwiseQuantized, /* [](const OperatorDef& /* def */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out;
      TensorShape X = in[0];
      X.set_dims(
          X.dims().size() - 1, X.dims(X.dims().size() - 1) + 2 * sizeof(float));
      out.push_back(std::move(X));
      out[0].set_data_type(TensorProto_DataType_UINT8);
      return out;
    } */}

/**
  | Applies 8-bit row-wise quantization
  | by determining the range (maximum -
  | minimum) and offset (minimum value)
  | of each row in the input matrix, and then
  | scaling each element to an 8-bit number
  | between 0 and 255.
  | 
  | To later de-quantize values, the scale
  | (range / 255) and offset (bias) are stored
  | alongside the data.
  | 
  | More precisely, each row contains int8
  | elements for each quantized element,
  | and the last 4 bytes of each row in the
  | output matrix are a float storing the
  | scale followed by another float containing
  | the scale.)
  |
  */
register_cpu_operator!{
    HalfFloatToFused8BitRowwiseQuantizedHalfScaleBias,
    FloatToFused8BitRowwiseQuantizedOp<
        f16,
        f16,
        convertfp16fp32,
        true,
        CPUContext>
}

num_inputs!{HalfFloatToFused8BitRowwiseQuantizedHalfScaleBias,  1}

num_outputs!{HalfFloatToFused8BitRowwiseQuantizedHalfScaleBias, 1}

inputs!{HalfFloatToFused8BitRowwiseQuantizedHalfScaleBias, 
    0 => ("input", "Float16 input data")
}

outputs!{HalfFloatToFused8BitRowwiseQuantizedHalfScaleBias, 
    0 => ("output", "Fused scale, bias and quantized data")
}

tensor_inference_function!{HalfFloatToFused8BitRowwiseQuantizedHalfScaleBias, /* [](const OperatorDef& /* def */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out;
      TensorShape X = in[0];
      X.set_dims(
          X.dims().size() - 1,
          X.dims(X.dims().size() - 1) + 2 * sizeof(at::Half));
      out.push_back(std::move(X));
      out[0].set_data_type(TensorProto_DataType_UINT8);
      return out;
    } */
}

/**
  | De-quantizes the result of the
  | 
  | FloatToFused8BitRowwiseQuantized
  | operator.
  | 
  | The input is expected to encode the scale
  | as a 32-bit float in the second to the
  | last 4 bytes of each row, followed by
  | the bias as a 32-bit float in the next
  | 4 bytes, and the quantized values in
  | the preceding bytes of the row.
  | 
  | The output is a matrix containing only
  | the values, but de-quantized.
  | 
  | De-quantization is performed by multiplying
  | each value by its row's scale and bias
  | parameters.
  | 
  | The de-quantized values will thus not
  | be exactly equal to the original, un-quantized
  | floating point values.
  |
  */
register_cpu_operator!{
    Fused8BitRowwiseQuantizedToFloat,
    Fused8BitRowwiseQuantizedToFloatOp<
        f32,
        f32,
        nullptr,
        false,
        CPUContext>
}

num_inputs!{Fused8BitRowwiseQuantizedToFloat, 1}

num_outputs!{Fused8BitRowwiseQuantizedToFloat, 1}

inputs!{Fused8BitRowwiseQuantizedToFloat, 
    0 => ("scale_bias_quantized_input", "Fused scale, bias and quantized data")
}

outputs!{Fused8BitRowwiseQuantizedToFloat, 
    0 => ("float_output", "Float32 data")
}

tensor_inference_function!{Fused8BitRowwiseQuantizedToFloat, /* [](const OperatorDef& /* def */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out;
      TensorShape X = in[0];
      X.set_dims(
          X.dims().size() - 1, X.dims(X.dims().size() - 1) - 2 * sizeof(float));
      out.push_back(std::move(X));
      out[0].set_data_type(TensorProto_DataType_FLOAT);
      return out;
    } */
}

/**
  | De-quantizes the result of the
  | 
  | FloatToFused8BitRowwiseQuantized
  | operator.
  | 
  | The input is expected to encode the scale
  | as a 16-bit float in the second to the
  | last 2 bytes of each row, followed by
  | the bias as a 16-bit float in the next
  | 2 bytes, and the quantized values in
  | the preceding bytes of the row.
  | 
  | The output is a matrix containing only
  | the values, but de-quantized. De-quantization
  | is performed by multiplying each value
  | by its row's scale and bias parameters.
  | 
  | The de-quantized values will thus not
  | be exactly equal to the original, un-quantized
  | floating point values.
  |
  */
register_cpu_operator!{
    Fused8BitRowwiseQuantizedHalfScaleBiasToFloat,
    Fused8BitRowwiseQuantizedToFloatOp<
        f32,
        f16,
        nullptr,
        false,
        CPUContext>
}

num_inputs!{Fused8BitRowwiseQuantizedHalfScaleBiasToFloat, 1}

num_outputs!{Fused8BitRowwiseQuantizedHalfScaleBiasToFloat, 1}

inputs!{Fused8BitRowwiseQuantizedHalfScaleBiasToFloat, 
    0 => ("scale_bias_quantized_input", "Fused scale, bias and quantized data")
}

outputs!{Fused8BitRowwiseQuantizedHalfScaleBiasToFloat, 
    0 => ("float_output", "Float32 data")
}

tensor_inference_function!{Fused8BitRowwiseQuantizedHalfScaleBiasToFloat, /* [](const OperatorDef& /* def */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out;
      TensorShape X = in[0];
      X.set_dims(
          X.dims().size() - 1,
          X.dims(X.dims().size() - 1) - 2 * sizeof(at::Half));
      out.push_back(std::move(X));
      out[0].set_data_type(TensorProto_DataType_FLOAT);
      return out;
    } */}

///----------------------------------------
no_gradient!{FloatToFused8BitRowwiseQuantizedHalfScaleBias}
no_gradient!{HalfFloatToFused8BitRowwiseQuantized}
no_gradient!{HalfFloatToFused8BitRowwiseQuantizedHalfScaleBias}
no_gradient!{Fused8BitRowwiseQuantizedToFloat}
no_gradient!{Fused8BitRowwiseQuantizedHalfScaleBiasToFloat}
no_gradient!{Fused8BitRowwiseQuantizedToHalfFloat}
no_gradient!{Fused8BitRowwiseQuantizedHalfScaleBiasToHalfFloat}

/**
  | De-quantizes the result of the
  | 
  | HalfFloatToFused8BitRowwiseQuantized
  | operator.
  | 
  | The input is expected to encode the scale
  | as a 32-bit float in the second to the
  | last 4 bytes of each row, followed by
  | the bias as a 32-bit float in the next
  | 4 bytes, and the quantized values in
  | the preceding bytes of the row.
  | 
  | The output is a matrix containing only
  | the values, but de-quantized. De-quantization
  | is performed by multiplying each value
  | by its row's scale and bias parameters.
  | 
  | The de-quantized values will thus not
  | be exactly equal to the original, un-quantized
  | floating point values.
  |
  */
register_cpu_operator!{
    Fused8BitRowwiseQuantizedToHalfFloat,
    Fused8BitRowwiseQuantizedToFloatOp<
        f16,
        f32,
        convertfp32fp16,
        true,
        CPUContext>
}

num_inputs!{Fused8BitRowwiseQuantizedToHalfFloat, 1}

num_outputs!{Fused8BitRowwiseQuantizedToHalfFloat, 1}

inputs!{Fused8BitRowwiseQuantizedToHalfFloat, 
    0 => ("scale_bias_quantized_input", "Fused scale, bias and quantized data")
}

outputs!{Fused8BitRowwiseQuantizedToHalfFloat, 
    0 => ("float16_output", "Float16 data")
}

tensor_inference_function!{Fused8BitRowwiseQuantizedToHalfFloat, /* [](const OperatorDef& /* def */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out;
      TensorShape X = in[0];
      X.set_dims(
          X.dims().size() - 1, X.dims(X.dims().size() - 1) - 2 * sizeof(float));
      out.push_back(std::move(X));
      out[0].set_data_type(TensorProto_DataType_FLOAT16);
      return out;
    } */
}

/**
  | De-quantizes the result of the
  | 
  | FloatToFused8BitRowwiseQuantized
  | operator.
  | 
  | The input is expected to encode the scale
  | as a 16-bit float in the second to the
  | last 2 bytes of each row, followed by
  | the bias as a 16-bit float in the next
  | 2 bytes, and the quantized values in
  | the preceding bytes of the row.
  | 
  | The output is a matrix containing only
  | the values, but de-quantized. De-quantization
  | is performed by multiplying each value
  | by its row's scale and bias parameters.
  | 
  | The de-quantized values will thus not
  | be exactly equal to the original, un-quantized
  | floating point values.
  |
  */
register_cpu_operator!{
    Fused8BitRowwiseQuantizedHalfScaleBiasToHalfFloat,
    Fused8BitRowwiseQuantizedToFloatOp<
        f32,
        f16,
        nullptr,
        false,
        CPUContext>
}

num_inputs!{Fused8BitRowwiseQuantizedHalfScaleBiasToHalfFloat,  1}

num_outputs!{Fused8BitRowwiseQuantizedHalfScaleBiasToHalfFloat, 1}

inputs!{Fused8BitRowwiseQuantizedHalfScaleBiasToHalfFloat, 
    0 => ("scale_bias_quantized_input", "Fused scale, bias and quantized data")
}

outputs!{Fused8BitRowwiseQuantizedHalfScaleBiasToHalfFloat, 
    0 => ("float_output", "Float32 data")
}

tensor_inference_function!{Fused8BitRowwiseQuantizedHalfScaleBiasToHalfFloat, /* [](const OperatorDef& /* def */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out;
      TensorShape X = in[0];
      X.set_dims(
          X.dims().size() - 1,
          X.dims(X.dims().size() - 1) - 2 * sizeof(at::Half));
      out.push_back(std::move(X));
      out[0].set_data_type(TensorProto_DataType_FLOAT);
      return out;
    } */
}

pub type Fused8BitRowwiseQuantizedToFloatCPUOp = 
    Fused8BitRowwiseQuantizedToFloatOp<
        f32,
        f32,
        NoOpFunctor,
        false,
        CPUContext>;
