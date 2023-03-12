crate::ix!();

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
    } */
}
