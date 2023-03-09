crate::ix!();

/**
  | The operator performs sampling based
  | on the input sampling weights for each
  | batch.
  | 
  | All weights must be non-negative numbers.
  | 
  | The input is a 2-D tensor (Tensor) of
  | size (batch_size x weights_dim).
  | 
  | For each batch, an index is randomly
  | sampled from the distribution given
  | by the weights of the corresponding
  | batch.
  | 
  | The output is a 1-D tensor (Tensor) of
  | size (batch_size x 1) and contains the
  | index(es) of the sampled output.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct WeightedSampleOp<T, Context> {
    storage:      OperatorStorage,
    context:      Context,
    cum_mass:     Vec<f32>,
    unif_samples: Tensor,
    phantom:      PhantomData<T>,
}

num_inputs!{WeightedSample, (1,2)}

num_outputs!{WeightedSample, (1,2)}

inputs!{WeightedSample, 
    0 => ("sampling_weights", "A 2-D Tensor of size (batch_size x weights_dim). All weights must be non-negative numbers."),
    1 => ("sampling_values",  "An optional 2-D Tensor of size (batch_size x weights_dim). Its values correspond to the sampling weights.")
}

outputs!{WeightedSample, 
    0 => ("sampled_indexes", "The output tensor contains index(es) sampled from distribution given by the weight vector(s) in the input tensor The output is a 1-D Tensor of size (batch_size x 1)"),
    1 => ("sampled_values",  "The output tensor contains value(s) selected by the sampled index(es) It is a 1-D Tensor of size (batch_size x 1)")
}

tensor_inference_function!{WeightedSample, /* [](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out(2);
      int batch_size = in[0].dims(0);
      out[0] = CreateTensorShape(vector<int>{batch_size}, TensorProto::INT32);
      out[1] = CreateTensorShape(vector<int>{batch_size}, TensorProto::FLOAT);
      return out;
    } */
}

impl<T,Context> WeightedSampleOp<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
}

register_cpu_operator!{WeightedSample, WeightedSampleOp<f32, CPUContext>}

should_not_do_gradient!{WeightedSample}
