crate::ix!();

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
