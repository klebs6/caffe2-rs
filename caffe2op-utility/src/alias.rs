crate::ix!();

/**
  | -----------
  | @brief
  | 
  | Alias op makes the output and the input
  | share the same underlying storage.
  | 
  | WARNING: in general, in caffe2's operator
  | interface different tensors should
  | have different underlying storage,
  | which is the assumption made by components
  | such as the dependency engine and memory
  | optimization.
  | 
  | Thus, in normal situations you should
  | not use the AliasOp, especially in a
  | normal forward-backward pass.
  | 
  | The Alias op is provided so one can achieve
  | true asynchrony, such as
  | 
  | Hogwild, in a graph.
  | 
  | But make sure you understand all the
  | implications similar to multi-thread
  | computation before you use it explicitly.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct AliasOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

num_inputs!{Alias, 1}

num_outputs!{Alias, 1}

inputs!{Alias, 
    0 => ("input", "Input tensor whose storage will be shared.")
}

outputs!{Alias, 
    0 => ("output", "Tensor of same shape as input, sharing its storage.")
}

identical_type_and_shape!{Alias}

impl<Context> AliasOp<Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& input = Input(0);
        CAFFE_ENFORCE_GE(input.numel(), 0, "Tensor is not initialized");
        OutputTensorAlias(0, input);
        return true;
        */
    }
}
