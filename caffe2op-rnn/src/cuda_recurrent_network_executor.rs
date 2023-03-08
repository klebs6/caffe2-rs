crate::ix!();

pub struct CUDARecurrentNetworkExecutor {
    base:                      RecurrentNetworkExecutorBase,
    events:                    Vec<CudaEvent>,
    has_timestep_parallelism:  bool, // default = false
    max_cuda_streams:          i32, // default = 2
}

impl Drop for CUDARecurrentNetworkExecutor {

    fn drop(&mut self) {
        todo!();
        /* 
          for (cudaEvent_t ev : events_) {
            if (ev != nullptr) {
              CUDA_CHECK(cudaEventDestroy(ev));
            }
          }
       */
    }
}

impl CUDARecurrentNetworkExecutor {

    pub fn new(
        step_net_def:        &NetDef,
        recurrent_input_map: &mut HashMap<String,String>,
        timestep_blob:       String) -> Self {
    
        todo!();
        /*
            : RecurrentNetworkExecutorBase(step_net_def, recurrent_input_map, timestep_blob)
        */
    }
    
    #[inline] pub fn ignore_link_dependencies(&mut self) -> bool {
        
        todo!();
        /*
            return true;
        */
    }
    
    #[inline] pub fn analyze_ops(&mut self)  {
        
        todo!();
        /*
            /**
          * Check if there is an op that only depends on ops from previous
          * timestep, and that ops is not the last op. Then we can start computation
          * in subsequent timesteps before the whole previous timestep has finished.
          * If there is no parallelism, we can avoid overhead of event-based
          * dependency management.
          */
        has_timestep_parallelism_ = false;
        for (auto& rnn_op : timestep_ops_template_) {
          int i = rnn_op.order;
          if (rnn_op.parents.size() >= 1 && i < timestep_ops_template_.size() - 1) {
            bool only_recurrent_deps = std::all_of(
                      rnn_op.parents.begin(),
                      rnn_op.parents.end(), [&](const int &parent) {
                        return parent > i;
                      }
            );
            if (only_recurrent_deps) {
              VLOG(1) << "Timestep parallel op: " << ProtoDebugString(step_net_def_.op(i));
              has_timestep_parallelism_ = true;

              for (int dep : rnn_op.parents) {
                if (dep == timestep_ops_template_.size() - 1) {
                  // This op depends on the last op of the previous iteration,
                  // so it will block any parallelism
                  has_timestep_parallelism_ = false;
                  break;
                }
              }
              break;
            }
          }
        }
        LOG(INFO) << "Analyzed ops for timestep parallelism: " << has_timestep_parallelism_;
        */
    }
    
    #[inline] pub fn set_max_streams(&mut self, n: i32)  {
        
        todo!();
        /*
            max_cuda_streams_ = n;
        */
    }
    
    /**
      | Special execution for CUDA.
      | 
      | It tries to run ops with as little overhead
      | as possible, but to identify opportunities
      | to run ops with "frontier execution"
      | parallelism, i.e by starting kernel
      | from next timestep in parallel with
      | the current timestep.
      | 
      | This is done by assigning streams.
      |
      */
    #[inline] pub fn exec_range(&mut self, from: i32, to: i32)  {
        
        todo!();
        /*
            int direction = to > from ? 1 : -1;

      int max_streams = max_parallel_timesteps_ > 0 ?
                        std::min(max_parallel_timesteps_, max_cuda_streams_)
                        : max_cuda_streams_;
      int stream_seq = 0;
      int num_ops = timestep_ops_[0].size();

      events_.resize(num_ops * timestep_ops_.size(), nullptr);

      int gpu_id = -1;

      // Loop over timesteps
      for (int t = from; t != to; t += direction) {
        bool first_timestep = t == from;
        bool last_timestep =
            (direction == -1 && t == 0) || (direction == 1 && t == to - 1);
        auto& ops = timestep_ops_[t];
        int stream_id = stream_seq % max_streams;

        for (int i = 0; i < ops.size(); i++) {
          auto& rnn_op = ops[i];

          // Special handling for link ops -- we just run them directly
          // they do not execute any kernels.
          if (rnn_op.link_op) {
            rnn_op.op->RunAsync(stream_id);
            CAFFE_ENFORCE(
                rnn_op.dependencies.empty(),
                "GPU executor ignores link dependencies");
            continue;
          }

          if (gpu_id == -1 &&
              rnn_op.op->device_option().device_type() ==
                  DeviceTypeProto::PROTO_CUDA) {
            gpu_id = rnn_op.op->device_option().device_id();
          } else {
            CAFFE_ENFORCE(
                rnn_op.op->device_option().device_type() == 0 ||
                    rnn_op.op->device_option().device_id() == gpu_id,
                "RNN Executor only supports ops on one GPU");
          }

          // If have recurrent parents, add for event waits so that those
          // parents complete their work.
          if (has_timestep_parallelism_ && !first_timestep) {
            for (int parent : rnn_op.parents) {
              if (parent > i) {
                int parent_ev_idx = (t - direction) * num_ops + parent;
                CHECK(events_.size() > parent_ev_idx);
                CAFFE_ENFORCE(events_[parent_ev_idx] != nullptr);
                CUDA_CHECK(cudaStreamWaitEvent(
                    CUDAContext::cuda_stream(gpu_id, stream_id),
                    events_[parent_ev_idx],
                    0));
            }
            }
          }

          // Run the op in the given stream
          rnn_op.op->RunAsync(stream_id);

          // Create and record event for this op, if it has at least one
          // recurrent dependency.
          if (has_timestep_parallelism_ && !last_timestep) {
            for (int dep : rnn_op.dependencies) {
              if (dep < i) {
                int event_idx = t * num_ops + i;
                // Create event for recurrent connections
                if (events_[event_idx] == nullptr) {
                  CUDA_CHECK(cudaEventCreate(&events_[event_idx]));
                }
                CUDA_CHECK(cudaEventRecord(
                    events_[event_idx],
                    CUDAContext::cuda_stream(gpu_id, stream_id)));
                break;
              }
            }
          }
        } // for over ops

        // Next timestep will run on different stream
        if (has_timestep_parallelism_) {
          stream_seq++;
        }
      } // for over timesteps

      /**
       * Wait for all the started streams to complete.
       */
      for (int stream_id = 0; stream_id <= std::min(stream_seq, max_streams - 1);
           stream_id++) {
        VLOG(1) << "Wait for stream:" << stream_id;
        CUDA_CHECK(
            cudaStreamSynchronize(CUDAContext::cuda_stream(gpu_id, stream_id)));
      }
        */
    }
    
    #[inline] pub fn run(&mut self, t: i32) -> bool {
        
        todo!();
        /*
            CAFFE_ENFORCE_GE(T, 0, "Negative number of steps");
      if (T == 0) {
        return true;
      }
      _ExecRange(0, T);
      return true;
        */
    }
    
    #[inline] pub fn run_backwards(&mut self, t: i32) -> bool {
        
        todo!();
        /*
            CAFFE_ENFORCE_GE(T, 0, "Negative number of steps");
      if (T == 0) {
        return true;
      }
      _ExecRange(T - 1, -1);
      return true;
        */
    }
}
