crate::ix!();

/**
  | Greedy decoder for connectionist temporal
  | classification.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct CTCGreedyDecoderOp<Context> {

    storage: OperatorStorage,
    context: Context,

    merge_repeated: bool,

    /*
      | Input: X, 3D tensor; L, 1D tensor.
      | 
      | Output: Y sparse tensor
      |
      */
}

register_cpu_operator!{
    CTCGreedyDecoder, 
    CTCGreedyDecoderOp<CPUContext>
}

num_inputs!{CTCGreedyDecoder, (1,2)}

num_outputs!{CTCGreedyDecoder, 2}

inputs!{CTCGreedyDecoder, 
    0 => ("INPUTS", "3D float Tensor sized [max_time, batch_size, num_classes]"),
    1 => ("SEQ_LEN", "(optional) 1D int vector containing sequence lengths, having size [batch_size] seq_len will be set to max_time if not provided")
}

outputs!{CTCGreedyDecoder, 
    0 => ("OUTPUT_LEN", "Output_len matrix size (batch). The row store: [decoded_length]"),
    1 => ("VALUES", "Values vector, size (total_decoded_outputs). The vector stores the decoded classes")
}

args!{CTCGreedyDecoder, 
    0 => ("merge_repeated", "When merge_repeated is true, merge repeated classes in output.")
}

inherit_onnx_schema!{CTCGreedyDecoder}

should_not_do_gradient!{CTCGreedyDecoder}

input_tags!{
    CTCGreedyDecoderOp {
        Inputs,
        SeqLen
    }
}

output_tags!{
    CTCGreedyDecoderOp {
        OutputLen,
        Values
    }
}

impl<Context> CTCGreedyDecoderOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...) 
        merge_repeated_ =
            this->template GetSingleArgument<bool>("merge_repeated", true);
        */
    }
}
