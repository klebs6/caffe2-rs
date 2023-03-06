crate::ix!();

/**
  | Computes the Jensen-Shannon divergence
  | (JSD) between two Bernoulli distributions
  | where each is parametrized by a single
  | probability.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct BernoulliJSDOp<T, Context> {
    storage: OperatorStorage,
    context: Context,
    phantom: PhantomData<T>,
}

num_inputs!{BernoulliJSD, 2}

num_outputs!{BernoulliJSD, 1}

inputs!{BernoulliJSD, 
    0 => ("X", "array of probabilities for prediction"),
    1 => ("T", "array of probabilities for target")
}

outputs!{BernoulliJSD, 
    0 => ("L", "array of JSD losses")
}

register_cpu_operator!{
    BernoulliJSD, 
    BernoulliJSDOp<f32, CPUContext>
}
