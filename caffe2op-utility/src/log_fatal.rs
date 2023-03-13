crate::ix!();

pub struct LogFatalOp {
    storage: OperatorStorage,
    context: CPUContext,
    message: String,
}

register_cpu_operator!{LogFatal, LogFatalOp}

num_inputs!{LogFatal, 0}

num_outputs!{LogFatal, 0}

should_not_do_gradient!{LogFatal}

impl LogFatalOp {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<CPUContext>(std::forward<Args>(args)...),
            message_(GetSingleArgument<std::string>(
                "message",
                "Logging from LogFatalOp"))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            LOG(FATAL) << message_;
        return true;
        */
    }
}
