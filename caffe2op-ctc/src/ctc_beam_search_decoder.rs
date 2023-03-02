crate::ix!();

/**
  | Prefix beam search decoder for connectionist
  | temporal classification.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct CTCBeamSearchDecoderOp<Context> {
    storage: OperatorStorage,
    context: Context,

    beam_width:       i32,
    num_candidates:   i32,
    prune_threshold:  f32,

    /*
      | Input: X, 3D tensor; L, 1D tensor.
      | 
      | Output: Y sparse tensor
      |
      */
}

register_cpu_operator!{
    CTCBeamSearchDecoder, 
    CTCBeamSearchDecoderOp<CPUContext>
}

num_inputs!{CTCBeamSearchDecoder, (1,2)}

num_outputs!{CTCBeamSearchDecoder, (2,3)}

inputs!{CTCBeamSearchDecoder, 
    0 => ("INPUTS", "3D float Tensor sized [max_activation_length, batch_size, alphabet_size] of network logits (before softmax application)."),
    1 => ("SEQ_LEN", "(optional) 1D int vector containing sequence lengths, having size [batch_size] seq_len will be set to max_time if not provided.")
}

outputs!{CTCBeamSearchDecoder, 
    0 => ("OUTPUT_LEN", "Output_len matrix size (batch_size * num_candidates). Each index stores lengths of candidates for its corresponding batch item."),
    1 => ("VALUES", "Values vector, size (total_decoded_outputs). The flattened vector of final output sequences, in batch order."),
    2 => ("OUTPUT_PROB", "Probability vector, size (total_decoded_outputs). Each index stores final output probability of its corresponding batch item.")
}

args!{CTCBeamSearchDecoder, 
    0 => ("beam_width", "Maximum number of candidates to carry over to next activation step."),
    1 => ("prune_threshold", "Probability threshold below which outputs are ignored.")
}

inherit_onnx_schema!{CTCBeamSearchDecoder}

should_not_do_gradient!{CTCBeamSearchDecoder}

input_tags!{
    CTCBeamSearchDecoderOp {
        Inputs,
        SeqLen
    }
}

output_tags!{
    CTCBeamSearchDecoderOp {
        OutputLen,
        Values,
        OutputProb
    }
}

impl<Context> CTCBeamSearchDecoderOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...) 

        beam_width_ = this->template GetSingleArgument<int32_t>("beam_width", 10);
        num_candidates_ =
            this->template GetSingleArgument<int32_t>("num_candidates", 1);
        prune_threshold_ =
            this->template GetSingleArgument<float>("prune_threshold", 0.001);
        DCHECK(beam_width_ >= num_candidates_);
        */
    }
}
