crate::ix!();

///-------------------------
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct HalfToFloatOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

num_inputs!{HalfToFloat, 1}

num_outputs!{HalfToFloat, 1}

tensor_inference_function!{HalfToFloat, /* [](const OperatorDef& /* unused */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out;
      const TensorShape& X = in[0];
      out.push_back(X);
      out[0].set_data_type(TensorProto_DataType_FLOAT);

      return out;
    } */
}
