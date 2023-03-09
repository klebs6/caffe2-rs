crate::ix!();

/**
  | Stop a timer started with **TimerBegin**.
  | Publishes a CAFFE_EVENT.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/stats_ops.cc
  |
  */
pub struct TimerEndOp {
    storage: OperatorStorage,
    context: CPUContext,
}

register_cpu_operator!{TimerEnd, TimerEndOp}

num_inputs!{TimerEnd, 1}

num_outputs!{TimerEnd, 0}

inputs!{
    TimerEnd, 
    0 => ("timer", "(*Tensor`<ptr>`*): pointer to a timer object; obtained from **TimerBegin** op")
}

impl TimerEndOp {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            OperatorStorage::Input<TimerInstance*>(0)->end();
        return true;
        */
    }
}
