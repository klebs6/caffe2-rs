crate::ix!();

use crate::{
    CPUContext,
    OperatorStorage,
};

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
pub struct PerplexityOp<T,Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,
    phantom: PhantomData<T>,
}

register_cpu_operator!{Perplexity, PerplexityOp<float, CPUContext>}

num_inputs!{Perplexity, 1}

num_outputs!{Perplexity, 1}

inputs!{Perplexity, 
    0 => ("probabilities", "The input data as Tensor. It contains a batch of true label or target probabilities")
}

outputs!{Perplexity, 
    0 => ("output", "The output- a single (float) perplexity value for the batch")
}

should_not_do_gradient!{Perplexity}

impl<T,Context> PerplexityOp<T, Context> {
    
    #[inline] pub fn run_f32_on_cpu_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0);

      DCHECK_EQ(X.dim(), 1);
      int N = X.dim32(0);

      auto* Y = Output(0, vector<int64_t>(), at::dtype<float>());
      const auto* Xdata = X.data<float>();

      float perplexity = 1.0;
      for (int i = 0; i < N; ++i) {
        perplexity *= pow(Xdata[i], -1.0/N);
      }
      *(Y->template mutable_data<float>()) = perplexity;
      return true;
        */
    }
}

