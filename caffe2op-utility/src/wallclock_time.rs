crate::ix!();

pub const kPrintFileExtension: &'static str = ".log";

/**
  | Time since epoch in nanoseconds.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct WallClockTimeOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

num_inputs!{WallClockTime, 0}

num_outputs!{WallClockTime, 1}

outputs!{WallClockTime, 
    0 => ("time", "The time in nanoseconds.")
}

should_not_do_gradient!{WallClockTime}

impl<Context> WallClockTimeOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            int64_t nanoseconds = static_cast<long int>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch())
                .count());

        TensorCPU* output = Output(0);
        output->Resize();
        *output->template mutable_data<int64_t>() = nanoseconds;

        return true;
        */
    }
}
