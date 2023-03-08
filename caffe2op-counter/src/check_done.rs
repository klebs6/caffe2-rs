crate::ix!();

/**
  | If the internal count value <= 0, outputs
  | true, otherwise outputs false.
  | 
  | Will always use TensorCPU regardless
  | the Context
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct CheckCounterDoneOp<T, Context> {
    storage: OperatorStorage,
    context: Context,
    phantom: PhantomData<T>,
}

num_inputs!{CheckCounterDone, 1}

num_outputs!{CheckCounterDone, 1}

inputs!{CheckCounterDone, 
    0 => ("counter", "*(type: Tensor`<ptr>`)* A blob pointing to an instance of a counter.")
}

outputs!{CheckCounterDone, 
    0 => ("done", "*(type: bool)* True if the internal count is zero or negative, otherwise False.")
}

impl<T,Context> CheckCounterDoneOp<T,Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& counterPtr = this->template Input<std::unique_ptr<Counter<T>>>(0);
        auto* output = Output(0);
        output->Resize(std::vector<int>{});
        *output->template mutable_data<bool>() = counterPtr->checkIfDone();
        return true;
        */
    }
}
