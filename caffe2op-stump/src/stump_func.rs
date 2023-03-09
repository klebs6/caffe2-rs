crate::ix!();

/**
  | Converts each input element into either
  | high_ or low_value based on the given
  | threshold.
  | 
  | out[i] = low_value if in[i] <= threshold
  | else high_value
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct StumpFuncOp<TIN,TOUT,Context> {

    storage:     OperatorStorage,
    context:     Context,

    threshold:   TIN,
    low_value:   TOUT,
    high_value:  TOUT,

    /*
      | Input: label,
      | 
      | output: weight
      |
      */
}

register_cpu_operator!{StumpFunc, StumpFuncOp<float, float, CPUContext>}

num_inputs!{StumpFunc, 1}

num_outputs!{StumpFunc, 1}

inputs!{StumpFunc, 
    0 => ("X", "tensor of float")
}

outputs!{StumpFunc, 
    0 => ("Y", "tensor of float")
}

tensor_inference_function!{StumpFunc, /* ([](const OperatorDef&,
                                const vector<TensorShape>& input_types) {
      vector<TensorShape> out(1);
      out.at(0) = input_types.at(0);
      out.at(0).set_data_type(TensorProto_DataType::TensorProto_DataType_FLOAT);
      return out;
    }) */
}

no_gradient!{StumpFunc}

impl<TIN,TOUT,Context> StumpFuncOp<TIN,TOUT,Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            threshold_(this->template GetSingleArgument<TIN>("threshold", 0)),
            low_value_(this->template GetSingleArgument<TOUT>("low_value", 0)),
            high_value_(this->template GetSingleArgument<TOUT>("high_value", 0))
        */
    }
}
