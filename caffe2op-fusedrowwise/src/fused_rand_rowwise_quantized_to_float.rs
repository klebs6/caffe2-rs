crate::ix!();

/**
  | De-quantizes the result of the
  | FloatToFusedRandRowwiseQuantized
  | operator.
  | 
  | Refer FloatToFusedRandRowwiseQuantized
  | operator for details.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct FusedRandRowwiseQuantizedToFloatOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

register_cpu_operator!{
    FusedRandRowwiseQuantizedToFloat,
    FusedRandRowwiseQuantizedToFloatOp<CPUContext>
}

num_inputs!{FusedRandRowwiseQuantizedToFloat, 1}

num_outputs!{FusedRandRowwiseQuantizedToFloat, 1}

inputs!{FusedRandRowwiseQuantizedToFloat, 
    0 => ("quantized_input", "Fused bitwidth, tail, min, max and quantized data")
}

outputs!{FusedRandRowwiseQuantizedToFloat, 
    0 => ("float_input", "Float32 data")
}

tensor_inference_function!{FusedRandRowwiseQuantizedToFloat, 
    |def: &OperatorDef, input: &Vec<TensorShape>| {
        todo!();
        /*
          vector<TensorShape> out;
          for (int i = 0; i < def.output_size(); i++) {
            TensorShape ts;
            ts.set_unknown_shape(true);
            ts.set_data_type(TensorProto_DataType_FLOAT);
            out.push_back(ts);
          }
          return out;
        */
    }
}

no_gradient!{FusedRandRowwiseQuantizedToFloat}

input_tags!{
    FusedRandRowwiseQuantizedToFloatOp {
        DataFusedQuantized
    }
}

output_tags!{
    FusedRandRowwiseQuantizedToFloatOp {
        DataFloat
    }
}
