crate::ix!();

pub struct ExecutionOptions {

    /**
      | number of gpu streams per gpu per cpu
      | thread
      |
      */
    streams_per_gpu:           i32,

    /// ops synchronization options
    finish_chain:              bool,

    always_schedule_child:     bool,

    /// try to pick gpu stream that is not busy
    check_stream_status:       bool,

    /// use single thread pool for all devices
    use_single_pool:           bool,

    /**
      | use per net instances thread pools instead
      | of global ones
      |
      */
    use_per_net_pools:         bool,

    /// whether RunAsync is blocking
    is_blocking:               bool,

    /// prof_dag counters reporting
    report_stats:              bool,

    /**
      | immediately run children tasks inline
      | whenever possible
      |
      */
    use_dfs_scheduling:        bool,

    /**
      | run net's root tasks in RunAsync thread
      | instead of in thread pool
      |
      */
    run_root_tasks_inline:     bool,
}

impl Default for ExecutionOptions {

    fn default() -> Self {
        Self {
            streams_per_gpu:           1,
            finish_chain:              false,
            always_schedule_child:     false,
            check_stream_status:       false,
            use_single_pool:           false,
            use_per_net_pools:         false,
            is_blocking:               false,
            report_stats:              false,
            use_dfs_scheduling:        false,
            run_root_tasks_inline:     false,
        }
    }
}

impl From<&Arc<NetDef>> for ExecutionOptions {

    fn from(net_def: &Arc<NetDef>) -> Self {
    
        todo!();
        /*
            static const std::string kDag = "dag";
      static const std::string kProfDag = "prof_dag";
      static const std::string kAsyncDag = "async_dag";
      static const std::string kSimpleNet = "simple";

      std::string net_type;
      if (net_def->has_type() && !net_def->type().empty()) {
        net_type = net_def->type();
      } else {
        net_type = kSimpleNet;
      }
      if (net_type == kDag || net_type == kProfDag) {
        streams_per_gpu_ = 1;
        finish_chain_ = true;
        always_schedule_child_ = true;
        check_stream_status_ = false;
        use_single_pool_ = true;
        use_per_net_pools_ = true;
        is_blocking_ = true;
        report_stats_ = (net_type == kProfDag);
      } else if (net_type == kAsyncDag) {
        streams_per_gpu_ = 1;
        finish_chain_ = false;
        always_schedule_child_ = true;
        check_stream_status_ = false;
        use_single_pool_ = true;
        use_per_net_pools_ = true;
        is_blocking_ = true;
        report_stats_ = false;
      } else {
        streams_per_gpu_ = FLAGS_caffe2_streams_per_gpu;
        finish_chain_ = false;
        always_schedule_child_ = false;
        check_stream_status_ = FLAGS_caffe2_net_async_check_stream_status;
        use_single_pool_ = FLAGS_caffe2_net_async_use_single_pool;
        use_per_net_pools_ = FLAGS_caffe2_net_async_use_per_net_pools;
        is_blocking_ = false;
        report_stats_ = false;
      }

      use_dfs_scheduling_ = false;

      for (int arg_idx = 0; arg_idx < net_def->arg_size(); ++arg_idx) {
        auto& arg = net_def->arg(arg_idx);
        if (arg.has_name() && arg.name() == "enable_profiling") {
          CAFFE_ENFORCE(arg.has_i(), "enable_profiling should be an int");
          report_stats_ = arg.i() == 1;
        }
        if (arg.has_name() && arg.name() == "deferrable_mode") {
          CAFFE_ENFORCE(arg.has_i(), "deferrable_mode should be an int");
          use_dfs_scheduling_ = arg.i() == 1; // corr. to DFS scheduling
        }
      }

      if (FLAGS_caffe2_net_async_profile_operators) {
        report_stats_ = true;
      }
      run_root_tasks_inline_ = FLAGS_caffe2_net_async_run_root_tasks_inline;
        */
    }
}
