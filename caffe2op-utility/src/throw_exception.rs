crate::ix!();

pub struct ThrowExceptionOp {
    storage: OperatorStorage,
    context: CPUContext,
    message: String,
}

register_cpu_operator!{ThrowException, ThrowExceptionOp}

num_inputs!{ThrowException, 0}

num_outputs!{ThrowException, 0}

should_not_do_gradient!{ThrowException}

impl ThrowExceptionOp {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<CPUContext>(std::forward<Args>(args)...),
            message_(GetSingleArgument<std::string>(
                "message",
                "Exception from ThrowExceptionOp"))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            CAFFE_THROW(message_);
        */
    }
}
