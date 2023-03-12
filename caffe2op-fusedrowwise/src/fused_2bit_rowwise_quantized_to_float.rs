crate::ix!();

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

