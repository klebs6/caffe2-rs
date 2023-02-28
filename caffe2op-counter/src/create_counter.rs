/*!
  | TODO(jiayq): deprecate these ops &
  | consolidate them with IterOp/AtomicIterOp
  |
  */

crate::ix!();

/**
  | Creates a count-down counter with initial
  | value specified by the `init_count`
  | argument.
  |
  */
pub struct CreateCounterOp<T, Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,

    init_count: T,// = 0;
}

num_inputs!{CreateCounter, 0}

num_outputs!{CreateCounter, 1}

outputs!{CreateCounter, 
    0 => ("counter", "*(type: Tensor`<ptr>`)* A blob pointing to an instance of a new counter.")
}

args!{CreateCounter, 
    0 => ("init_count", "*(type: int; default: 0)* Initial count for the counter, must be >= 0.")
}

impl<T, Context> CreateCounterOp<T, Context> {
    
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
            *this->template Output<std::unique_ptr<Counter<T>>>(0) =
            std::unique_ptr<Counter<T>>(new Counter<T>(init_count_));
        return true;
        */
    }
}
