crate::ix!();

#[inline] pub fn create_rnnexecutor<CUDAContext>(
    step_net_def:        &NetDef,
    recurrent_input_map: &mut HashMap<String,String>,
    timestep_blob:       String,
    arg_helper:          ArgumentHelper) -> Box<RecurrentNetworkExecutorBase> {
    
    todo!();
    /*
        auto* exec = new CUDARecurrentNetworkExecutor(
          step_net_def, recurrent_input_map, timestep_blob);
      int max_streams = arg_helper.GetSingleArgument<int>("rnn_executor.max_cuda_streams", 0);
      if (max_streams > 0) {
        exec->setMaxStreams(max_streams);
        LOG(INFO) << "Set max streams:" << max_streams;
      }
      std::unique_ptr<RecurrentNetworkExecutorBase> ptr(exec);
      return ptr;
    */
}

/*
/**
  | Implementation of RecurrentNetworkExecutor
  | that uses thread pool for multithreaded
  | execution of RNNs. Used with CPU.
  |
  */
#[inline] pub fn create_rnnexecutor<CPUContext>(
    step_net_def:        &NetDef,
    recurrent_input_map: &mut HashMap<String,String>,
    timestep_blob:       String,
    rnn_args:            ArgumentHelper) -> Box<RecurrentNetworkExecutorBase> {
    
    todo!();
    /*
        auto* exec = new ThreadedRecurrentNetworkExecutor(
          step_net_def, recurrent_input_map, timestep_blob);
      int num_threads =
          rnn_args.GetSingleArgument<int>("rnn_executor.num_threads", 0);
      if (num_threads > 0) {
        exec->setNumThreads(num_threads);
        LOG(INFO) << "Set num threads: " << num_threads;
      }
      exec->debug_ = rnn_args.GetSingleArgument<int>("rnn_executor_debug", 0);
      return std::unique_ptr<RecurrentNetworkExecutorBase>(exec);
    */
}
*/
