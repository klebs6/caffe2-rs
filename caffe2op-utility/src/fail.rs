crate::ix!();

pub struct FailOp {
    storage: OperatorStorage,
    context: CPUContext,
}

register_cpu_operator!{Fail, FailOp}

num_inputs!{Fail, 0}

num_outputs!{Fail, 0}

should_not_do_gradient!{Fail}

impl FailOp {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<CPUContext>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return false;
        */
    }
}
