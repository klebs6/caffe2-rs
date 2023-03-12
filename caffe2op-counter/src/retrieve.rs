crate::ix!();

/**
  | Retrieve the current value from the
  | counter as an integer.
  | 
  | Will always use TensorCPU regardless
  | the Context
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct RetrieveCountOp<T, Context> {
    storage: OperatorStorage,
    context: Context,
    phantom: PhantomData<T>,
}

num_inputs!{RetrieveCount, 1}

num_outputs!{RetrieveCount, 1}

inputs!{RetrieveCount, 
    0 => ("counter", "*(type: Tensor`<ptr>`)* A blob pointing to an instance of a counter.")
}

outputs!{RetrieveCount, 
    0 => ("count", "*(type: int)* Current count value.")
}

scalar_type!{RetrieveCount, TensorProto::INT64}

impl<T,Context> RetrieveCountOp<T,Context> {
    
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
        *output->template mutable_data<T>() = counterPtr->retrieve();
        return true;
        */
    }
}
