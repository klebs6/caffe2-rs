crate::ix!();

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
