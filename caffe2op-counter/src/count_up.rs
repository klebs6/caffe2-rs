crate::ix!();

/**
  | Increases count value by 1 and outputs
  | the previous value atomically.
  | 
  | Will always use TensorCPU regardless
  | the Context
  |
  */
pub struct CountUpOp<T, Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,
    phantom: PhantomData<T>,
}

num_inputs!{CountUp, 1}

num_outputs!{CountUp, 1}

inputs!{CountUp, 
    0 => ("counter", "*(type: Tensor`<ptr>`)* A blob pointing to an instance of a counter.")
}

outputs!{CountUp, 
    0 => ("previous_count", "*(type: int)* Count value BEFORE this operation.")
}

impl<T,Context> CountUpOp<T,Context> {
    
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
        *output->template mutable_data<T>() = counterPtr->countUp();
        return true;
        */
    }
}
