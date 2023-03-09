crate::ix!();

/**
  | `SparseDropoutWithReplacement`
  | takes a 1-d input tensor and a lengths
  | tensor.
  | 
  | Values in the Lengths tensor represent
  | how many input elements consitute each
  | example in a given batch. The set of input
  | values for an example will be replaced
  | with the single dropout value with probability
  | given by the `ratio` argument.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SparseDropoutWithReplacementOp<Context> {
    context:            Context,
    ratio:              f32,
    replacement_value:  i64,
}

no_gradient!{SparseDropoutWithReplacement}

num_inputs!{SparseDropoutWithReplacement, 2}

inputs!{SparseDropoutWithReplacement, 
    0 => ("X",       "*(type: Tensor`<int64_t>`)* Input data tensor."),
    1 => ("Lengths", "*(type: Tensor`<int32_t>`)* Lengths tensor for input.")
}

outputs!{SparseDropoutWithReplacement, 
    0 => ("Y",             "*(type: Tensor`<int64_t>`)* Output tensor."),
    1 => ("OutputLengths", "*(type: Tensor`<int32_t>`)* Output tensor.")
}

args!{SparseDropoutWithReplacement, 
    0 => ("ratio",             "*(type: float; default: 0.0)* Probability of an element to be replaced."),
    1 => ("replacement_value", "*(type: int64_t; default: 0)* Value elements are replaced with.")
}

same_number_of_output!{SparseDropoutWithReplacement}

impl<Context> SparseDropoutWithReplacementOp<Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            ratio_(this->template GetSingleArgument<float>("ratio", 0.0)),
            replacement_value_(
                this->template GetSingleArgument<int64_t>("replacement_value", 0)) 
        // It is allowed to drop all or drop none.
        CAFFE_ENFORCE_GE(ratio_, 0.0, "Ratio should be a valid probability");
        CAFFE_ENFORCE_LE(ratio_, 1.0, "Ratio should be a valid probability");
        */
    }
}
