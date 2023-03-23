crate::ix!();

pub struct AsyncNetBase {

    base:                NetBase,

    /// Operator/task graph
    operators:           Vec<*mut OperatorStorage>,

    operator_nodes:      Vec<OperatorNode>,
    chains:              Vec<Vec<i32>>,

    /// chains' parents/children
    chain_nodes:         Vec<OpGraphNode>,

    /// for testing
    execution_chains:    ExecutionChains,

    /// Pools and streams
    pools_mutex:         parking_lot::RawMutex,

    cpu_pools:           PoolsMap,
    gpu_pools:           PoolsMap,

    num_workers:         i32,
    success:             AtomicBool,

    /// Tracing
    tracer:              Arc<Tracer>,

    /// execution mode flags
    options:             ExecutionOptions,

    counters:            ProfDAGCounters,
    helper:              Box<AsyncNetExecutorHelper>,
}

impl Drop for AsyncNetBase {

    fn drop(&mut self) {
        todo!();
        /* 
      if (options_.report_stats_) {
        counters_.GetReport().PrintStats();
      }
 */
    }
}

impl AsyncNetBase {

    pub fn new_from_net_def_and_workspace(net_def: &Arc<NetDef>, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
        
        */
    }
    
    #[inline] pub fn supports_async(&mut self) -> bool {
        
        todo!();
        /*
            return true;
        */
    }
    
    #[inline] pub fn get_operators(&self) -> Vec<*mut OperatorStorage> {
        
        todo!();
        /*
            return operators_;
        */
    }
    
    #[inline] pub fn tEST_execution_chains(&self) -> &ExecutionChains {
        
        todo!();
        /*
            return execution_chains_;
        */
    }
    
    #[inline] pub fn get_stream_counters(&mut self) -> &mut Vec<i32> {
        
        todo!();
        /*
            static thread_local std::vector<int> stream_counters_;
      return stream_counters_;
        */
    }
    
    pub fn new(net_def: &Arc<NetDef>, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : NetBase(net_def, ws), options_(net_def), counters_(net_def) 

      operator_nodes_ = dag_utils::prepareOperatorNodes(net_def, ws);
      helper_ = std::make_unique<AsyncNetExecutorHelper>(this);
      operators_.reserve(operator_nodes_.size());
      for (const auto& node : operator_nodes_) {
        auto op_ptr = node.operator_.get();
        op_ptr->SetExecutorHelper(helper_.get());
        operators_.push_back(op_ptr);
      }

      if (FLAGS_caffe2_net_async_inference_mode) {
        execution_chains_ = dag_utils::computeGroups(operator_nodes_);
      } else {
        execution_chains_ = dag_utils::computeChains(operator_nodes_);
      }
      chains_.reserve(execution_chains_.size());
      for (const auto& kv : execution_chains_) {
        chains_.push_back(kv.second);
      }
      chain_nodes_ = dag_utils::prepareChainGraphNodes(operator_nodes_, chains_);

      events_.reserve(chains_.size());
      for (const auto& chain : chains_) {
        const auto& last_op = operators_[chain.back()];
        events_.push_back(&last_op->event());
        // keep events for inner chain ops in case of profiling
        if (!options_.report_stats_) {
          for (const auto& op_id : chain) {
            if (op_id == chain.back() || op_id == chain.front()) {
              continue;
            }
            const auto& op = operators_[op_id];
            op->DisableEvent();
          }
        }
      }

      num_workers_ = net_def->has_num_workers() ? net_def->num_workers() : -1;

      tracer_ = tracing::create(this, net_def->name());
      if (tracer_) {
        LOG(INFO) << "Tracing net: " << net_def->name();
      }
        */
    }
    
    #[inline] pub fn handle_run_error(&mut self) -> bool {
        
        todo!();
        /*
            #ifdef CAFFE2_USE_EXCEPTION_PTR
      // Check net's events for exceptions and rethrow chronologically the first one
      int first_exc_task_id = -1;
      int64_t first_exc_ts = 0;
      for (int task_id = 0; task_id < tasksNum(); ++task_id) {
        if (event(task_id).HasException()) {
          if (first_exc_task_id >= 0) {
            auto exc_ts = event(task_id).ErrorTimestamp();
            if (exc_ts < first_exc_ts) {
              first_exc_task_id = task_id;
              first_exc_ts = exc_ts;
            }
          } else {
            first_exc_task_id = task_id;
            first_exc_ts = event(task_id).ErrorTimestamp();
          }
        }
      }
      if (first_exc_task_id >= 0) {
        LOG(ERROR) << "Rethrowing exception from the run of '" << Name() << "'";
        event(first_exc_task_id).RethrowException();
      }
    #endif // CAFFE2_USE_EXCEPTION_PTR

      if (!success_) {
        LOG(ERROR) << "Error encountered in the run of '" << Name() << "'";
      }
      return success_;
        */
    }
    
    #[inline] pub fn run_async(&mut self) -> bool {
        
        todo!();
        /*
            tracing::startIter(tracer_);
      reset();
      return DoRunAsync();
        */
    }
    
    #[inline] pub fn pool_getter(&mut self, 
        pools:       &mut PoolsMap,
        device_type: i32,
        device_id:   i32,
        pool_size:   i32) -> Arc<dyn TaskThreadPoolBaseInterface> {
        
        todo!();
        /*
            std::unique_lock<std::mutex> pools_lock(pools_mutex_);
      auto pool = pools[device_id][pool_size];
      if (!pool) {
        pool = c10::ThreadPoolRegistry()->Create(
            DeviceTypeName(device_type),
            device_id,
            pool_size,
            options_.use_per_net_pools_);
        pools[device_id][pool_size] = pool;
      }
      return pool.get();
        */
    }
    
    #[inline] pub fn pool(&mut self) -> Arc<dyn TaskThreadPoolBaseInterface> {
        
        todo!();
        /*
            // By default using a non-pinned CPU option
      DeviceOption dev;
      dev.set_device_type(PROTO_CPU);
      return pool(dev);
        */
    }
    
    #[inline] pub fn pool_with_device_option(&mut self, device_option: &DeviceOption) -> Arc<dyn TaskThreadPoolBaseInterface> {
        
        todo!();
        /*
            if (options_.use_single_pool_) {
        return poolGetter(cpu_pools_, PROTO_CPU, -1, num_workers_);
      }
      const auto device_type = device_option.device_type();
      if (IsCPUDeviceType(device_type)) {
        auto numa_node_id = -1;
        if (device_option.has_numa_node_id()) {
          numa_node_id = device_option.numa_node_id();
          CAFFE_ENFORCE_GE(numa_node_id, 0, "Invalid NUMA node id: ", numa_node_id);
        }
        CAFFE_ENFORCE_LT(
            numa_node_id,
            FLAGS_caffe2_net_async_max_numa_nodes,
            "Invalid NUMA node id: ",
            numa_node_id);
        return poolGetter(cpu_pools_, device_type, numa_node_id, num_workers_);
      } else if (IsGPUDeviceType(device_type)) {
        auto gpu_id = device_option.device_id();
        CAFFE_ENFORCE(
            gpu_id >= 0 && gpu_id < FLAGS_caffe2_net_async_max_gpus,
            "Invalid GPU id: " + c10::to_string(gpu_id));
        return poolGetter(gpu_pools_, device_type, gpu_id, num_workers_);
      } else {
        CAFFE_THROW("Unsupported device type " + c10::to_string(device_type));
      }
        */
    }
    
    #[inline] pub fn stream(&mut self, task_id: i32) -> i32 {
        
        todo!();
        /*
            const auto& device_option = event(task_id).GetDeviceOption();
      int stream_id = 0;
      if (IsGPUDeviceType(device_option.device_type())) {
        int gpu_id = device_option.device_id();
        CAFFE_ENFORCE_GE(gpu_id, 0, "Invalid gpu id: " + c10::to_string(gpu_id));
        if ((unsigned)gpu_id >= getStreamCounters().size()) {
          getStreamCounters().resize(gpu_id + 1, 0);
        }
        do {
          stream_id = getStreamCounters().at(gpu_id)++;
          getStreamCounters().at(gpu_id) %= options_.streams_per_gpu_;
        } while (options_.check_stream_status_ &&
                 !isStreamFree(task_id, stream_id));
      }
      return stream_id;
        */
    }
    
    #[inline] pub fn is_stream_free(&self, task_id: i32, stream_id: i32) -> bool {
        
        todo!();
        /*
            auto& task = chains_[task_id];
      auto& last_task_op = operators_[task.back()];
      return last_task_op->IsStreamFree(stream_id);
        */
    }
    
    #[inline] pub fn can_schedule(&mut self, 
        task_id:       i32,
        status:        *const Vec<EventStatus>,
        parent_failed: *mut bool) -> bool {
        
        todo!();
        /*
            auto first_child_op_id = chains_[task_id].front();
      for (auto parent_id : parents(task_id)) {
        auto last_parent_op_id = chains_[parent_id].back();
        EventStatus parent_status;
        if (status) {
          parent_status = status->at(parent_id);
        } else {
          parent_status = operators_[last_parent_op_id]->event().Query();
        }

        if (parent_status == EventStatus::EVENT_FAILED) {
          if (parent_failed) {
            *parent_failed = true;
          }
          return false;
        }

        bool can_schedule = Event::CanSchedule(
            operators_[last_parent_op_id]->event().GetType(),
            parent_status,
            operators_[first_child_op_id]->event().GetType(),
            operators_[first_child_op_id]->SupportsAsyncScheduling());
        if (!can_schedule) {
          return false;
        }
      }

      return true;
        */
    }
    
    #[inline] pub fn can_schedule_with_parent_id_and_child_id(
        &mut self, 
        parent_id: i32, 
        child_id:  i32) -> bool {
        
        todo!();
        /*
            auto& parent_event = event(parent_id);
      auto first_child_op_id = chains_[child_id].front();
      auto* first_child_op = operators_[first_child_op_id];
      return Event::CanSchedule(
          parent_event.GetType(),
          parent_event.Query(),
          first_child_op->event().GetType(),
          first_child_op->SupportsAsyncScheduling());
        */
    }
    
    #[inline] pub fn tasks_num(&self) -> i32 {
        
        todo!();
        /*
            return chains_.size();
        */
    }
    
    #[inline] pub fn event(&self, task_id: i32) -> &mut Event {
        
        todo!();
        /*
            auto& task = chains_[task_id];
      auto& last_task_op = operators_[task.back()];
      return last_task_op->event();
        */
    }
    
    #[inline] pub fn query(&self, task_id: i32) -> EventStatus {
        
        todo!();
        /*
            return event(task_id).Query();
        */
    }
    
    #[inline] pub fn children(&self, task_id: i32) -> &Vec<i32> {
        
        todo!();
        /*
            const auto& task_node = chain_nodes_[task_id];
      return task_node.children_;
        */
    }
    
    #[inline] pub fn parents(&self, task_id: i32) -> &Vec<i32> {
        
        todo!();
        /*
            const auto& task_node = chain_nodes_[task_id];
      return task_node.parents_;
        */
    }
    
    #[inline] pub fn get_parent_count(&mut self, child_id: i32) -> i32 {
        
        todo!();
        /*
            auto& child_ops = chains_[child_id];
      auto& child_node = operator_nodes_[child_ops.front()];
      return child_node.runtime_parent_count_.load();
        */
    }
    
    #[inline] pub fn update_parent_count(&mut self, child_id: i32) -> i32 {
        
        todo!();
        /*
            auto& child_ops = chains_[child_id];
      auto& child_node = operator_nodes_[child_ops.front()];
      int parent_count = --child_node.runtime_parent_count_;
      CAFFE_ENFORCE_GE(parent_count, 0);
      return parent_count;
        */
    }
    
    #[inline] pub fn test_and_set_scheduled(&mut self, task_id: i32) -> bool {
        
        todo!();
        /*
            auto& task_ops = chains_[task_id];
      auto& task_op_node = operator_nodes_[task_ops.front()];
      return !task_op_node.scheduled_.test_and_set();
        */
    }
    
    #[inline] pub fn num_ops(&self, task_id: i32) -> i32 {
        
        todo!();
        /*
            return chains_[task_id].size();
        */
    }
    
    #[inline] pub fn first_task_op_id(&self, task_id: i32) -> i32 {
        
        todo!();
        /*
            return chains_[task_id].front();
        */
    }
    
    #[inline] pub fn last_task_op_id(&self, task_id: i32) -> i32 {
        
        todo!();
        /*
            return chains_[task_id].back();
        */
    }
    
    #[inline] pub fn first_task_op(&self, task_id: i32) -> *const OperatorStorage {
        
        todo!();
        /*
            return operator_nodes_[firstTaskOpId(task_id)].operator_.get();
        */
    }
    
    #[inline] pub fn last_task_op(&self, task_id: i32) -> *const OperatorStorage {
        
        todo!();
        /*
            return operator_nodes_[lastTaskOpId(task_id)].operator_.get();
        */
    }
    
    #[inline] pub fn first_task_op_mut(&mut self, task_id: i32) -> *mut OperatorStorage {
        
        todo!();
        /*
            return operator_nodes_[firstTaskOpId(task_id)].operator_.get();
        */
    }
    
    #[inline] pub fn last_task_op_mut(&mut self, task_id: i32) -> *mut OperatorStorage {
        
        todo!();
        /*
            return operator_nodes_[lastTaskOpId(task_id)].operator_.get();
        */
    }

    #[inline] pub fn async_wait(&self, 
        task_id:       i32,
        stream_id:     i32,
        wait_task_ids: &Vec<i32>)  {
        
        todo!();
        /*
            auto first_op_id = chains_[task_id].front();
      auto& first_op = operators_[first_op_id];
      std::vector<const Event*> events;
      events.reserve(wait_task_ids.size());
      for (auto wait_task_id : wait_task_ids) {
        events.push_back(&event(wait_task_id));
      }
      first_op->WaitEvents(events, stream_id);
        */
    }
    
    #[inline] pub fn reset(&mut self)  {
        
        todo!();
        /*
            for (auto& op : GetOperators()) {
        op->ResetEvent();
      }
      for (auto task_id = 0; task_id < tasksNum(); ++task_id) {
        auto& task_ops = chains_[task_id];
        auto& task_op_node = operator_nodes_[task_ops.front()];
        task_op_node.runtime_parent_count_ = parents(task_id).size();
        task_op_node.scheduled_.clear();
      }

      success_ = true;
        */
    }
    
    /// Exception/error handling
    #[inline] pub fn handle_chain_error(&mut self, 
        task_id:        i32,
        op:             *mut OperatorStorage,
        err_str:        *const u8,
        save_exception: Option<bool>)  
    {
        let save_exception = save_exception.unwrap_or(false);
        
        todo!();
        /*
            std::string err_msg = err_str;
      if (op) {
        err_msg += ",  op " + (op->has_debug_def() ? op->type() : " unknown");
      }
      LOG(ERROR) << err_msg;
      // mark end of chain with an error
      if (query(task_id) == EventStatus::EVENT_INITIALIZED) {
        if (save_exception) {
          event(task_id).SetFinishedWithException(err_msg.c_str());
        } else {
          event(task_id).SetFinished(err_msg.c_str());
        }
      }
        */
    }
    
    #[inline] pub fn run(&mut self, task_id: i32, stream_id: i32) -> bool {
        
        todo!();
        /*
            OperatorStorage* op = nullptr;
      try {
        // Optionally insert async wait ops,
        // skip when finish_chain_ is set -
        // all parents are guaranteed to be finished
        if (!options_.finish_chain_) {
          asyncWait(task_id, stream_id, parents(task_id));
        }
        int iter_id = -1;
        if (tracer_) {
          iter_id = tracer_->getIter();
        }
        for (auto& op_id : chains_[task_id]) {
          op = operators_[op_id];
          bool success = false;
          if (!options_.report_stats_) {
            TRACE_EVENT(
                tracing::TRACE_OP,
                op_id,
                tracing::TRACE_TASK,
                task_id,
                tracing::TRACE_STREAM,
                stream_id,
                tracing::TRACE_ITER,
                iter_id);
            success = op->RunAsync(stream_id);
          } else {
            counters_.AddPerOpStartTime(op_id);
            success = op->RunAsync(stream_id);
            if (success && op->device_option().device_type() != PROTO_CPU) {
              op->Finish();
            }
            counters_.AddPerOpEndTime(op_id);
          }

          if (!success) {
            handleChainError(task_id, op, "Failed to execute an op");
            return false;
          }
        }

        op = nullptr;
        if (options_.finish_chain_) {
          operators_[chains_[task_id].back()]->event().Finish();
        }
      } catch (const std::exception& e) {
        handleChainError(task_id, op, e.what(), /* save_exception */ true);
        return false;
      } catch (...) {
        handleChainError(
            task_id,
            op,
            "Failed to execute task: unknown error",
            /* save_exception */ true);
        return false;
      }

      return true;
        */
    }
    
    #[inline] pub fn finish_tasks(&mut self, task_ids: &HashSet<i32>)  {
        
        todo!();
        /*
            for (const auto& task_id : task_ids) {
        event(task_id).Finish();
      }
        */
    }
    
    #[inline] pub fn finalize_events(&mut self)  {
        
        todo!();
        /*
            std::vector<OperatorStorage*> pending_ops;
      for (auto task_id = 0; task_id < tasksNum(); ++task_id) {
        auto status = query(task_id);
        if (status == EventStatus::EVENT_SCHEDULED) {
          // async cpu ops need to be handled separately,
          // as they may potentially never finish
          auto* op = lastTaskOp(task_id);
          if (op->HasAsyncPart() &&
              op->device_option().device_type() == PROTO_CPU) {
            pending_ops.push_back(op);
          } else {
            event(task_id).Finish();
          }
        } else if (status == EventStatus::EVENT_INITIALIZED) {
          event(task_id).SetFinished();
        }
      }

      // avoid events cancelling each other and causing
      // a deadlock
      std::atomic_flag error_happened = ATOMIC_FLAG_INIT;
      for (auto* pending_op : pending_ops) {
        pending_op->event().SetCallback(
            [pending_op, &pending_ops, &error_happened]() {
              // if one of the async cpu ops failed,
              // we have to terminate other pending async cpu ops
              auto status = pending_op->event().Query();
              TORCH_CHECK(
                  status == EventStatus::EVENT_SUCCESS ||
                  status == EventStatus::EVENT_FAILED);
              if (status == EventStatus::EVENT_FAILED) {
                // go through all the ops and terminate them,
                // we may get an exception in case of multiple
                // SetFinished() calls
                if (!error_happened.test_and_set()) {
                  for (auto* op : pending_ops) {
                    if (op != pending_op) {
                      try {
                        op->CancelAsyncCallback();

                        // throw and catch exception to preserve stack trace
                        try {
                          throw AsyncNetCancelled();
                        } catch (const AsyncNetCancelled& e) {
                          op->event().SetFinishedWithException(e.what());
                        }
                      } catch (const EnforceNotMet&) {
                        // ignore
                      }
                    }
                  }
                }
              }
            });
      }

      // wait for all pending ops to be finished or be terminated
      for (auto* pending_op : pending_ops) {
        pending_op->event().Finish();
      }

      for (auto task_id = 0; task_id < tasksNum(); ++task_id) {
        if (event(task_id).Query() != EventStatus::EVENT_SUCCESS) {
          success_ = false;
          break;
        }
      }
        */
    }
    
    #[inline] pub fn get_operator_stats(&self) -> ProfDAGProtos {
        
        todo!();
        /*
            return counters_.GetReport().GetOperatorStats();
        */
    }
    
    #[inline] pub fn get_per_operator_cost(&self) -> ProfDAGProtos {
        
        todo!();
        /*
            return counters_.GetReport().GetPerOperatorCost();
        */
    }
    
    #[inline] pub fn get_prof_report(&self) -> ProfDAGReport {
        
        todo!();
        /*
            return counters_.GetReport();
        */
    }
}
