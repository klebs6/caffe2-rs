crate::ix!();

pub struct ThrowChildThreadExceptionOp {
    storage: OperatorStorage,
    context: CPUContext,
    message: String,
}

register_cpu_operator!{ThrowChildThreadException, ThrowChildThreadExceptionOp}

num_inputs!{ThrowChildThreadException, 0}

num_outputs!{ThrowChildThreadException, 0}

should_not_do_gradient!{ThrowChildThreadException}

impl ThrowChildThreadExceptionOp {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<CPUContext>(std::forward<Args>(args)...),
            message_(GetSingleArgument<std::string>(
                "message",
                "Exception from ThrowChildThreadExceptionOp"))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            std::thread t([this]() { CAFFE_THROW(this->message_); });

        t.join();
        return true;
        */
    }
}
