crate::ix!();

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
