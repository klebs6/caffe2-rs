crate::ix!();

/**
  | Check if the net name is white-listed
  | for tracing (specified via a command
  | line flag)
  |
  */
#[inline] pub fn is_traceable_net_name(net_name: &String) -> bool {
    
    todo!();
    /*
        auto tracing_nets = caffe2::split(',', FLAGS_caffe2_net_async_names_to_trace);
      return !net_name.empty() &&
          std::find(tracing_nets.begin(), tracing_nets.end(), net_name) !=
          tracing_nets.end();
    */
}

#[inline] pub fn has_enable_tracing_flag(net: *const NetBase) -> bool {
    
    todo!();
    /*
        if (!net->has_debug_def()) {
        return false;
      }
      return GetFlagArgument(net->debug_def(), "enable_tracing", false);
    */
}

#[inline] pub fn get_tracing_config_from_net(net: *const NetBase) -> TracingConfig {
    
    todo!();
    /*
        ArgumentHelper arg_helper(net->debug_def());
      TracingConfig cfg;

      cfg.mode = (arg_helper.GetSingleArgument<std::string>("tracing_mode", "") ==
                  "GLOBAL_TIMESLICE")
          ? TracingMode::GLOBAL_TIMESLICE
          : TracingMode::EVERY_K_ITERATIONS;

      cfg.filepath = arg_helper.GetSingleArgument<std::string>(
          "tracing_filepath", FLAGS_caffe2_net_async_tracing_filepath);

      cfg.trace_every_nth_batch = arg_helper.GetSingleArgument<int>(
          "trace_every_nth_batch", FLAGS_caffe2_net_async_tracing_nth);
      cfg.dump_every_nth_batch = arg_helper.GetSingleArgument<int>(
          "dump_every_nth_batch", FLAGS_caffe2_net_async_tracing_dumping_nth);

      cfg.trace_for_n_ms =
          arg_helper.GetSingleArgument<int>("trace_for_n_ms", cfg.trace_for_n_ms);
      cfg.trace_every_n_ms = arg_helper.GetSingleArgument<int>(
          "trace_every_n_ms", cfg.trace_every_n_ms);

      return cfg;
    */
}
