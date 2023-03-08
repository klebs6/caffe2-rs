crate::ix!();

/**
  | Resets a count-down counter with initial
  | value specified by the `init_count`
  | argument.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct ResetCounterOp<T, Context> {
    storage: OperatorStorage,
    context: Context,

    init_count: T,
}

num_inputs!{ResetCounter, 1}

num_outputs!{ResetCounter, (0,1)}

inputs!{ResetCounter, 
    0 => ("counter", "*(type: Tensor`<ptr>`)* A blob pointing to an instance of a counter.")
}

outputs!{ResetCounter, 
    0 => ("previous_value", "*(type: int)* [OPTIONAL] count value BEFORE this operation.")
}

args!{ResetCounter, 
    0 => ("init_count", "*(type: int; default: 0)* Resets counter to this value, must be >= 0.")
}

impl<T,Context> ResetCounterOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            init_count_(this->template GetSingleArgument<T>("init_count", 0)) 

        CAFFE_ENFORCE_LE(0, init_count_, "negative init_count is not permitted.");
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& counterPtr = this->template Input<std::unique_ptr<Counter<T>>>(0);
        auto previous = counterPtr->reset(init_count_);
        if (OutputSize() == 1) {
          auto* output = Output(0);
          output->Resize();
          *output->template mutable_data<T>() = previous;
        }
        return true;
        */
    }
}
