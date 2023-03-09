crate::ix!();

/**
  | Pad empty field given lengths and index
  | features,
  | 
  | Input(0) is a blob pointing to the lengths
  | of samples in one batch, [Input(1),...
  | Input(num_fields)] a list of tensors
  | containing the data for each field of
  | the features.
  | 
  | PadEmptySamples is thread safe.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct PadEmptySamplesOp<Context> {

    storage: OperatorStorage,
    context: Context,
}

num_inputs!{PadEmptySamples, (1,INT_MAX)}

num_outputs!{PadEmptySamples, (1,INT_MAX)}

inputs!{PadEmptySamples, 
    0 => ("lengths", "A blob containing a pointer to the lengths.")
}

outputs!{PadEmptySamples, 
    0 => ("out_lengths", "Tensor containing lengths with empty sample padded.")
}

impl<Context> PadEmptySamplesOp<Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
}
