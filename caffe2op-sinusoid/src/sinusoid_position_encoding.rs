crate::ix!();

/**
  | Calculates a sinusoid position encoding
  | tensor as described in https://arxiv.org/abs/1706.03762.
  | Takes a 2-D tensor (of size M x K) of positions
  | as input, the embedding size as an argument,
  | and outputs a position encoding tensor
  | of size (M x K x embedding_size). Here
  | M is typically the max sequence length
  | and K is typically the batch size.
  | 
  | The input tensor must satisfy input[m,
  | 0] == input[m, k] for all k.
  | 
  | Encoded as amplitude * SIN(pos/alpha^(i/embedding_size))
  | if i is even, else amplitude * COS(pos/alpha^(i/embedding_size)).
  | Here, pos is the position, alpha and
  | amplitude are tuning parameters, i
  | is the current dimension for the embedding,
  | and embedding_size is the number of
  | total dimensions in the embedding.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SinusoidPositionEncodingOp<Context> {
    storage:         OperatorStorage,
    context:         Context,
    embedding_size:  i32,
    alpha:           f32,
    amplitude:       f32,
}

register_cpu_operator!{
    SinusoidPositionEncoding,
    SinusoidPositionEncodingOp<CPUContext>
}

num_inputs!{SinusoidPositionEncoding, 1}

num_outputs!{SinusoidPositionEncoding, 1}

inputs!{SinusoidPositionEncoding, 
    0 => ("positions", "2-D tensor of positions to be encoded")
}

outputs!{SinusoidPositionEncoding, 
    0 => ("output", "3-D tensor representing the positional encoding")
}

args!{SinusoidPositionEncoding, 
    0 => ("embedding_size", "Desired embedding size/number of dimensions -- defaults to 100"),
    1 => ("alpha", "Sinusoid tuning parameter -- defaults to 10000"),
    2 => ("amplitude", "Amplitude of Sin/Cos output")
}
