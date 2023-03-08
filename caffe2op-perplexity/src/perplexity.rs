crate::ix!();

/**
  | Perplexity calculates how well a probability
  | distribution predicts a sample.
  | 
  | Perplexity takes a 1-D tensor containing
  | a batch of probabilities. Each value
  | in the tensor belongs to a different
  | sample and represents the probability
  | of the model predicting the true label
  | for that sample. The operator returns
  | a single (float) perplexity value for
  | the batch.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct PerplexityOp<T,Context> {
    storage: OperatorStorage,
    context: Context,
    phantom: PhantomData<T>,
}

register_cpu_operator!{
    Perplexity, 
    PerplexityOp<float, CPUContext>
}

num_inputs!{Perplexity, 1}

num_outputs!{Perplexity, 1}

inputs!{
    Perplexity, 
    0 => ("probabilities", "The input data as Tensor. It contains a batch of true label or target probabilities")
}

outputs!{
    Perplexity, 
    0 => ("output", "The output- a single (float) perplexity value for the batch")
}

should_not_do_gradient!{Perplexity}
