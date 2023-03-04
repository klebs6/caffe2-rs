crate::ix!();

///------------------------------------
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct FloatToHalfOp<Context> {
    storage: OperatorStorage,
    context: Context,
    clip:    bool,
}

num_inputs!{FloatToHalf, 1}

num_outputs!{FloatToHalf, 1}

tensor_inference_function!{FloatToHalf, /* [](const OperatorDef& /* unused */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out;
      const TensorShape& X = in[0];
      out.push_back(X);
      out[0].set_data_type(TensorProto_DataType_FLOAT16);

      return out;
    } */
}

impl<Context> FloatToHalfOp<Context> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator<Context>(operator_def, ws),
            clip_(this->template GetSingleArgument<bool>("clip", false))
        */
    }
}
