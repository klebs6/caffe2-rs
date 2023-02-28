crate::ix!();

/**
  | TODO(jiayq): deprecate these ops &
  | consolidate them with
  | 
  | IterOp/AtomicIterOp
  |
  */
register_cpu_operator!{CreateCounter,      CreateCounterOp<i64, CPUContext>}
register_cpu_operator!{ResetCounter,       ResetCounterOp<i64, CPUContext>}
register_cpu_operator!{CountDown,          CountDownOp<i64, CPUContext>}
register_cpu_operator!{CheckCounterDone,   CheckCounterDoneOp<i64, CPUContext>}
register_cpu_operator!{CountUp,            CountUpOp<i64, CPUContext>}
register_cpu_operator!{RetrieveCount,      RetrieveCountOp<i64, CPUContext>}

should_not_do_gradient!{CreateCounter}
should_not_do_gradient!{ResetCounter}
should_not_do_gradient!{CountDown}
should_not_do_gradient!{CountUp}
should_not_do_gradient!{RetrieveCount}

caffe_known_type!{std::unique_ptr<Counter<i64>>}

register_blob_serializer!{
    /*
    TypeMeta::Id<Box<Counter<i64>>>(),
    CounterSerializer
    */
}

register_blob_deserializer!{
    /*
    Box<Counter<i64>>,
    CounterDeserializer
    */
}

register_cuda_operator!{CreateCounter,        CreateCounterOp<i64, CUDAContext>}
register_cuda_operator!{ResetCounter,         ResetCounterOp<i64, CUDAContext>}
register_cuda_operator!{CountDown,            CountDownOp<i64, CUDAContext>}
register_cuda_operator!{CheckCounterDone,     CheckCounterDoneOp<i64, CUDAContext>}
register_cuda_operator!{CountUp,              CountUpOp<i64, CUDAContext>}
register_cuda_operator!{RetrieveCount,        RetrieveCountOp<i64, CUDAContext>}
