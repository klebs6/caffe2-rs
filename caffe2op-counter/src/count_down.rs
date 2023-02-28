crate::ix!();

/**
  | If the internal count value > 0, decreases
  | count value by 1 and outputs False, otherwise
  | outputs True.
  | 
  | Will always use TensorCPU regardless
  | the Context
  |
  */
pub struct CountDownOp<T, Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;

    storage: OperatorStorage,
    context: Context,
    phantom: PhantomData<T>,
}

num_inputs!{CountDown, 1}

num_outputs!{CountDown, 1}

inputs!{CountDown, 
    0 => ("counter", "*(type: Tensor`<ptr>`)* A blob pointing to an instance of a counter.")
}

outputs!{CountDown, 
    0 => ("done", "*(type: bool)* False unless the internal count is zero.")
}

impl<T,Context> CountDownOp<T,Context> {
    
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
        *output->template mutable_data<bool>() = counterPtr->countDown();
        return true;
        */
    }
}
