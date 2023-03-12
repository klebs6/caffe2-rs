crate::ix!();

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

