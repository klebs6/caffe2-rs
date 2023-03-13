crate::ix!();

/**
  | This operator converts dense or sparse
  | gradients to dense ones.
  | 
  | Therefore, sparse gradient can be back
  | propagated to Operators that consume
  | dense gradients only (e.g., FCGradient).
  | 
  | The operator's behaviors:
  | 
  | - In forward, simply pass in place or
  | copy input to the output.
  | 
  | - In backward, if the gradient passed-in
  | is sparse gradient, change it to dense
  | gradient in linear time; otherwise,
  | simply pass the dense gradient.
  | 
  | -----------
  | @brief
  | 
  | Pass inputs to outputs.
  | 
  | Input:
  |     DATA - dense tensor.
  | 
  | Output:
  |     DATA - same tensor as input.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct EnsureDenseOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

num_inputs!{EnsureDense, 1}

num_outputs!{EnsureDense, 1}

inputs!{EnsureDense, 
    0 => ("input", "Input tensors.")
}

outputs!{EnsureDense, 
    0 => ("output", "Output tensor. Same dimension as inputs.")
}

identical_type_and_shape!{EnsureDense}

allow_inplace!{EnsureDense, vec![(0, 0)]}

register_gradient!{EnsureDense, GetEnsureDenseGradient}

impl<Context> EnsureDenseOp<Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& input = Input(0);
        auto* output = Output(0);
        CAFFE_ENFORCE_GT(input.dim(), 0, "Input has to be at least a vector.");
        // it is allowed to have the output inplace overwrite the input but also
        // allow the output to be copied from the input
        if (&input != output) {
          output->ResizeLike(input);
          output->CopyFrom(input, true /*async*/);
        }
        return true;
        */
    }
}
