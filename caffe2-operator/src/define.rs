crate::ix!();

/**
  | experimental support for multiple
  | streams per worker per GPU
  |
  */
define_int!{caffe2_streams_per_gpu,
    1,
    "Number of streams per worker per GPU to use in GPU thread pool (experimental)"
}

define_bool!{caffe2_net_async_inference_mode,
    false,
    "If set, use one single chain containing all ops"}

define_bool!{caffe2_net_async_profile_operators,
    false,
    "If set, profile operators of the net regardless of net being prof_dag."}

define_int!{caffe2_net_async_max_gpus,
    16,
    "Max number of GPUs allowed in net async executor"}

define_int!{caffe2_net_async_max_numa_nodes,
    8,
    "Max number of NUMA nodes allowed in net async executor"}

define_int!{caffe2_net_async_thread_pool_size,
    0,
    "Number of threads in device thread pool by default"}

define_bool!{caffe2_net_async_check_stream_status,
    false,
    "Select next non-busy stream"}

define_bool!{caffe2_net_async_use_single_pool,
    false,
    "Use single thread pool for all devices"}

define_bool!{caffe2_net_async_use_per_net_pools,
    false,
    "Use per net thread pools"}

define_bool!{caffe2_net_async_run_root_tasks_inline,
    false,
    "Run root tasks in current thread instread of scheduling to threadpool"
}

define_string!{caffe2_net_async_tracing_filepath,
    "/tmp",
    "Path to save tracing information"}

define_string!{caffe2_net_async_names_to_trace,
    "",
    "Comma-separated list of net names to trace"}

define_int!{caffe2_net_async_tracing_nth, 100, "Trace every Nth batch"}

/**
  | For every Nth iterations, we will dump the
  | tracing results to a json file
  |
  | The file is appended with the iteration number.
  */
define_int!{caffe2_net_async_tracing_dumping_nth,
    10000,
    "Dump profiling result file every Nth batch"}


define_string!{
    caffe2_task_graph_engine,
    "futures",
    "Task graph engine type used by net executor"}

define_shared_registry!{
    /*
    TaskGraphRegistry,
    AsyncTaskGraphBase,
    ExecutorHelper*,
    const ExecutionOptions&
    */
}

#[cfg(c10_mobile)]
define_typed_registry!{
    ExternalTensorFunctionsBaseRegistry,
    TypeIdentifier,
    ExternalTensorFunctionsBase,
    Box
}
