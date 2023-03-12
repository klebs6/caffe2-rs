crate::ix!();

/**
  | Applies 4-bit row-wise fake quantization
  | to a tensor of floats.
  | 
  | The output looks like an int8 rowwise
  | quantized blob with scale and biases
  | in half float.
  |
  */
register_cpu_operator!{
    FloatToFused4BitFakeRowwiseQuantized,
    FloatToFusedNBitFakeRowwiseQuantizedOp<
        4,
        f32,
        internal::convertfp32fp32>
}

register_cpu_operator_with_engine!{
    FloatToFused4BitFakeRowwiseQuantized,
    GREEDY,
    FloatToFusedNBitFakeRowwiseQuantizedOp<
        4,
        f32,
        internal::convertfp32fp32,
        true /* GREEDY */>
}

no_gradient!{FloatToFused4BitFakeRowwiseQuantized}

num_inputs!{FloatToFused4BitFakeRowwiseQuantized, 1}

num_outputs!{FloatToFused4BitFakeRowwiseQuantized, 1}

inputs!{FloatToFused4BitFakeRowwiseQuantized, 
    0 => ("input", "Float32 input data")
}

outputs!{FloatToFused4BitFakeRowwiseQuantized, 
    0 => ("output", "Fused scale, bias and quantized data")
}

tensor_inference_function!{FloatToFused4BitFakeRowwiseQuantized, /* [](const OperatorDef& /* def */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out;
      TensorShape X = in[0];
      X.set_dims(1, X.dims(1) + 8);
      out.push_back(std::move(X));
      out[0].set_data_type(TensorProto_DataType_UINT8);
      return out;
    } */
}

/**
  | Applies 4-bit row-wise fake quantization
  | to a tensor of half floats.
  | 
  | The output looks like an int8 rowwise
  | quantized blob with scale and biases
  | in half float.
  |
  */
register_cpu_operator!{
    HalfToFused4BitFakeRowwiseQuantized,
    FloatToFusedNBitFakeRowwiseQuantizedOp<
        4,
        f16,
        internal::convertfp16fp32>
}

register_cpu_operator_with_engine!{
    HalfToFused4BitFakeRowwiseQuantized,
    GREEDY,
    FloatToFusedNBitFakeRowwiseQuantizedOp<
        4,
        f16,
        internal::convertfp16fp32,
        true /* GREEDY */>
}

no_gradient!{HalfToFused4BitFakeRowwiseQuantized}

num_inputs!{HalfToFused4BitFakeRowwiseQuantized, 1}

num_outputs!{HalfToFused4BitFakeRowwiseQuantized, 1}

inputs!{HalfToFused4BitFakeRowwiseQuantized, 
    0 => ("input", "Float16 input data")
}

outputs!{HalfToFused4BitFakeRowwiseQuantized, 
    0 => ("output", "Fused scale, bias and quantized data")
}

tensor_inference_function!{HalfToFused4BitFakeRowwiseQuantized, /* [](const OperatorDef& /* def */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out;
      TensorShape X = in[0];
      X.set_dims(1, X.dims(1) + 8);
      out.push_back(std::move(X));
      out[0].set_data_type(TensorProto_DataType_UINT8);
      return out;
    } */
}

/**
  | Applies 2-bit row-wise fake quantization
  | to a tensor of floats.
  | 
  | The output looks like an int8 rowwise
  | quantized blob with scale and biases
  | in half float.
  |
  */
register_cpu_operator!{
    FloatToFused2BitFakeRowwiseQuantized,
    FloatToFusedNBitFakeRowwiseQuantizedOp<
        2,
        f32,
        internal::convertfp32fp32>
}

no_gradient!{FloatToFused2BitFakeRowwiseQuantized}

register_cpu_operator_with_engine!{
    FloatToFused2BitFakeRowwiseQuantized,
    GREEDY,
    FloatToFusedNBitFakeRowwiseQuantizedOp<
        2,
        f32,
        internal::convertfp32fp32,
        true /* GREEDY */>
}

num_inputs!{FloatToFused2BitFakeRowwiseQuantized, 1}

num_outputs!{FloatToFused2BitFakeRowwiseQuantized, 1}

inputs!{FloatToFused2BitFakeRowwiseQuantized, 
    0 => ("input", "Float32 input data")
}

outputs!{FloatToFused2BitFakeRowwiseQuantized, 
    0 => ("output", "Fused scale, bias and quantized data")
}

tensor_inference_function!{FloatToFused2BitFakeRowwiseQuantized, 
    |def: &OperatorDef, input: &Vec<TensorShape>| {
        todo!();
        /*
          vector<TensorShape> out;
          TensorShape X = in[0];
          X.set_dims(1, X.dims(1) + 8);
          out.push_back(std::move(X));
          out[0].set_data_type(TensorProto_DataType_UINT8);
          return out;
        */
    }
}

/**
  | Applies 2-bit row-wise fake quantization
  | to a tensor of half floats.
  | 
  | The output looks like an int8 rowwise
  | quantized blob with scale and biases
  | in half float.
  |
  */
register_cpu_operator!{
    HalfToFused2BitFakeRowwiseQuantized,
    FloatToFusedNBitFakeRowwiseQuantizedOp<
        2,
        f16,
        convertfp16fp32> 
}

no_gradient!{HalfToFused2BitFakeRowwiseQuantized}

register_cpu_operator_with_engine!{
    HalfToFused2BitFakeRowwiseQuantized,
    GREEDY,
    FloatToFusedNBitFakeRowwiseQuantizedOp<
        2,
        f16,
        convertfp16fp32,
        Greedy>
}

num_inputs!{HalfToFused2BitFakeRowwiseQuantized, 1}

num_outputs!{HalfToFused2BitFakeRowwiseQuantized, 1}

inputs!{HalfToFused2BitFakeRowwiseQuantized, 
    0 => ("input", "Float16 input data")
}

outputs!{HalfToFused2BitFakeRowwiseQuantized, 
    0 => ("output", "Fused scale, bias and quantized data")
}

tensor_inference_function!{HalfToFused2BitFakeRowwiseQuantized, 
    |def: &OperatorDef, input: &Vec<TensorShape>| {
        todo!();
        /*
          vector<TensorShape> out;
          TensorShape X = in[0];
          X.set_dims(1, X.dims(1) + 8);
          out.push_back(std::move(X));
          out[0].set_data_type(TensorProto_DataType_UINT8);
          return out;
        */
    }
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

no_gradient!{FloatToFused8BitRowwiseQuantizedHalfScaleBias}
no_gradient!{HalfFloatToFused8BitRowwiseQuantized}
no_gradient!{HalfFloatToFused8BitRowwiseQuantizedHalfScaleBias}
no_gradient!{Fused8BitRowwiseQuantizedToFloat}
no_gradient!{Fused8BitRowwiseQuantizedHalfScaleBiasToFloat}
no_gradient!{Fused8BitRowwiseQuantizedToHalfFloat}
no_gradient!{Fused8BitRowwiseQuantizedHalfScaleBiasToHalfFloat}

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

register_cpu_operator_with_engine!{
    HalfToFused2BitRowwiseQuantized,
    GREEDY,
    FloatToFusedNBitRowwiseQuantizedOp<
    { 2 },
    f16,
    convertfp16fp32,
    Greedy>
}
