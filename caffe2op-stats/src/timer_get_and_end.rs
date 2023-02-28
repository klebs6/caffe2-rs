crate::ix!();

/**
  | Queries the current time of a timer in
  | nanos, stops the timer publishing a
  | CAFFE_EVENT.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/stats_ops.cc
  |
  */
pub struct TimerGetAndEndOp {
    storage: OperatorStorage,
    context: CPUContext,
}

register_cpu_operator!{TimerGetAndEnd, TimerGetAndEndOp}

#[test] fn timer_get_and_end_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    timerbegin_op = core.CreateOperator(
        "TimerBegin",
        [],
        ["timer"]
    )

    timerget_op = core.CreateOperator(
        "TimerGet",
        ["timer"],
        ["nanos"]
    )

    timerend_op = core.CreateOperator(
        "TimerEnd",
        ["timer"],
        []
    )

    timergetandend_op = core.CreateOperator(
        "TimerGetAndEnd",
        ["timer"],
        ["nanos"]
    )

    // Test TimerBegin/TimerGet/TimerEnd
    workspace.RunOperatorOnce(timerbegin_op)
    print("timer:", workspace.FetchBlob("timer"))
    workspace.RunOperatorOnce(timerget_op)
    print("nanos:", workspace.FetchBlob("nanos"))
    workspace.RunOperatorOnce(timerend_op)


    // Test TimerBegin/TimerGetAndEnd
    workspace.RunOperatorOnce(timerbegin_op)
    print("timer:", workspace.FetchBlob("timer"))
    workspace.RunOperatorOnce(timergetandend_op)
    print("nanos:", workspace.FetchBlob("nanos"))

    timer: b'timer, a C++ native class of type caffe2::TimerInstance*.'
    nanos: 361140
    timer: b'timer, a C++ native class of type caffe2::TimerInstance*.'
    nanos: [252250]

    */
}

num_inputs!{TimerGetAndEnd, 1}

num_outputs!{TimerGetAndEnd, 1}

inputs!{TimerGetAndEnd, 
    0 => ("timer", "(*Tensor`<ptr>`*): pointer to a timer object; obtained from **TimerBegin** op")
}

outputs!{TimerGetAndEnd, 
    0 => ("nanos", "(*Tensor`<int64>`*): scalar tensor containing time in nanoseconds")
}

impl TimerGetAndEndOp {

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
        OperatorStorage::Input<TimerInstance*>(0)->end();
        auto* res = Output(0);
        res->Resize(1);
        res->template mutable_data<int64_t>()[0] = nanos;
        return true;
        */
    }
}
