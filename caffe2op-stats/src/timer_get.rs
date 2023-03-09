crate::ix!();

/**
  | Queries the current time of a timer object
  | in nanoseconds.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/stats_ops.cc
  |
  */
pub struct TimerGetOp {
    storage: OperatorStorage,
    context: CPUContext,
}

register_cpu_operator!{TimerGet, TimerGetOp}

num_inputs!{TimerGet, 1}

num_outputs!{TimerGet, 1}

inputs!{
    TimerGet, 
    0 => ("timer", "(*Tensor`<ptr>`*): pointer to a timer object; obtained from **TimerBegin** op")
}

outputs!{
    TimerGet, 
    0 => ("nanos", "(*Tensor`<int64>`*): scalar containing time in nanoseconds")
}

impl TimerGetOp {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            int64_t nanos = OperatorStorage::Input<TimerInstance*>(0)->get_ns();
        auto* res = Output(0);
        res->Resize();
        res->template mutable_data<int64_t>()[0] = nanos;
        return true;
        */
    }
}
