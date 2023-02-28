crate::ix!();

/**
  | Start a wallclock timer, returning
  | a scalar tensor containing a pointer
  | to it. The timer is stopped by calling
  | **TimerEnd**.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/stats_ops.cc
  |
  */
pub struct TimerBeginOp {
    storage:     OperatorStorage,
    context:     CPUContext,
    given_name:  String,
    timer:       TimerInstance,
}

register_cpu_operator!{TimerBegin, TimerBeginOp}

num_inputs!{TimerBegin, 0}

num_outputs!{TimerBegin, 1}

outputs!{TimerBegin, 
    0 => ("timer", "(*Tensor`<ptr>`*): pointer to a timer object")
}

args!{TimerBegin, 
    0 => ("counter_name", "(*str*): name of the timer object; if not set use output name")
}

impl TimerBeginOp {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : Operator(operator_def, ws),
            given_name_(GetSingleArgument<std::string>( "counter_name", operator_def.output().Get(0))),
            timer_([this]() { return given_name_; }())
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            *OperatorStorage::Output<TimerInstance*>(0) = &timer_;
        timer_.begin();
        return true;
        */
    }
}

