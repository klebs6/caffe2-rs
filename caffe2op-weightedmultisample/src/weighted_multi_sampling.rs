crate::ix!();

/**
  | The operator performs sampling based
  | on the input sampling weights.
  | 
  | All weights are cummulative probability
  | thus sorted.
  | 
  | The output is a 1-D tensor (Tensor).
  | 
  | If two inputs are given, the second input
  | is used to provide shape of the output
  | sample tensor.
  | 
  | Otherwise, we use argument `num_samples`
  | to determine the number of samples to
  | generate.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct WeightedMultiSamplingOp<Context> {
    storage:     OperatorStorage,
    context:     Context,
    num_samples: i64,
}

num_inputs!{WeightedMultiSampling, (1,2)}

num_outputs!{WeightedMultiSampling, 1}

inputs!{WeightedMultiSampling, 
    0 => ("sampling_cdf",            "An optional 1-D Tensor.Input cumulative sampling probability (such as [0.2, 0.5, 0.8, 1.5]). All weights must be non-negative numbers. Note that the last value of CDF is not necessary 1. If the last value is not 1, all values in sampling_cdf will be scaled by this number."),
    1 => ("shape_tensor (optional)", "Tensor whose shape will be applied to output.")
}

outputs!{WeightedMultiSampling, 
    0 => ("sampled_indexes",         "The output tensor contains indices sampled from distribution given by the weight vector in the input tensor The output is a 1-D Tensor of size determined by argument `num_samples` or the second input tensor.")
}

args!{WeightedMultiSampling, 
    0 => ("num_samples",             "number of samples to sample from the input data")
}

should_not_do_gradient!{WeightedMultiSample}

register_cpu_operator!{
    WeightedMultiSampling, 
    WeightedMultiSamplingOp<CPUContext>
}
