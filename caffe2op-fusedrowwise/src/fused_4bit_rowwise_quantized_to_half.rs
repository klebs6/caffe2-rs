crate::ix!();

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

