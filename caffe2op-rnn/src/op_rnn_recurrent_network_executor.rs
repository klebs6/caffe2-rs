crate::ix!();

/**
  | RecurrentNetworkExecutor is a specialized
  | runtime for recurrent neural networks
  | (RNNs). It is invoked from the RecurrentNetworkOp
  | and RecurrentNetworkGradientOp.
  | 
  | Its main benefit over running each RNN
  | timestep as a separate net is that it
  | can run ops in subsequent timesteps
  | in parallel when possible.
  | 
  | For example, multi-layer LSTMs allow
  | for timestep parallelism because next
  | timestep's lower layer can start executing
  | at the same time as the same timestep's
  | upper layer.
  | 
  | There are two implementations of the
  | RNN executor: one for CPUs (ThreadedRecurrentNetworkExecutor)
  | and another for GPUs (CUDARecurrentNetworkExecutor).
  |
  */
pub trait RecurrentNetworkExecutorBaseTrait {

    fn run(&mut self, t: i32) -> bool;

    fn run_backwards(&mut self, t: i32) -> bool;

    fn ignore_link_dependencies(&mut self) -> bool;
}

pub struct RecurrentNetworkExecutorBase {
    timestep_ops:            Vec<Vec<RNNNetOperator>>,
    op_ptrs:                 Vec<*mut OperatorStorage>,
    timestep_ops_template:   Vec<RNNNetOperator>,
    step_net_def:            NetDef,
    op_deps:                 Vec<Vec<String>>,
    workspaces:              Vec<*mut Workspace>,
    recurrent_input_map:     HashMap<String,String>,
    timestep_blob:           String,
    max_parallel_timesteps:  i32,  // default = -1
    debug:                   bool, // default = false
}

impl Drop for RecurrentNetworkExecutorBase {
    fn drop(&mut self) {
        todo!();
        /* 
        if (debug_) {
          if (timestep_ops_.size() > 0) {
            PrintInfo(0);
          }
        }
       */
    }
}

impl RecurrentNetworkExecutorBase {
    
    pub fn new(
        step_net_def:        &NetDef,
        recurrent_input_map: &mut HashMap<String,String>,
        timestep_blob:       String) -> Self {
    
        todo!();
        /*
            : step_net_def_(step_net_def),
            recurrent_input_map_(recurrent_input_map),
            timestep_blob_(timestep_blob) 

        const bool net_def_has_device_option = step_net_def_.has_device_option();
        for (int i = 0; i < step_net_def_.op_size(); i++) {
          if (net_def_has_device_option) {
            // In the case when net def specifies device option, final device option
            // will be equal to merge of operator and net def device options, with
            // preference to settings from the operator.
            DeviceOption option;
            option.CopyFrom(step_net_def_.device_option());
            option.MergeFrom(step_net_def_.op(i).device_option());
            step_net_def_.mutable_op(i)->mutable_device_option()->CopyFrom(option);
          }
          op_deps_.push_back(op_deps(i));
        }
        */
    }
    
    /**
      | Callers must call EnsureTimestepInitialized
      | before starting execution for each
      | of the relevant timesteps.
      | 
      | If timestep was initialized before,
      | this is a no-op. First time this is called
      | the dependencies of the operators in
      | timestep are analyzed, and that incurs
      | higher overhead than subsequent calls.
      |
      */
    #[inline] pub fn ensure_timestep_initialized(&mut self, 
        t:              i32,
        ws:             *mut Workspace,
        observers_list: &Vec<Box<ObserverBase<OperatorStorage>>>)  {

        todo!();
        /*
            if (timestep_ops_template_.size() == 0) {
          // Firsrt invocation -- compute dependencies
          CalculateInternalDependencies();

          // Label ops based on whether they contain reference to the timestep
          // blob. This is an optimization to avoid string comparisons later.
          for (auto& rnn_op : timestep_ops_template_) {
            rnn_op.has_timestep_blob = false;
            const OperatorDef& op = step_net_def_.op(rnn_op.order);
            for (int i = 0; i < op.input_size(); i++) {
              if (op.input(i) == timestep_blob_) {
                rnn_op.has_timestep_blob = true;
                break;
              }
            }
            CAFFE_ENFORCE(
                !HasOutput(op, timestep_blob_),
                "Timestep cannot be output of an op: ",
                timestep_blob_,
                " op=" + ProtoDebugString(op));
          }
        }

        // Initialize timestep if it is not initialized
        if (timestep_ops_.size() <= t ||
            (timestep_ops_.size() > t && timestep_ops_[t].size() == 0)) {
          // Initialize empty timestep ops vectors for each timestep preceding
          // this.
          for (int j = timestep_ops_.size(); j < t + 1; j++) {
            timestep_ops_.push_back(std::vector<RNNNetOperator>());
            timestep_ops_.back().reserve(timestep_ops_template_.size());
          }

          // Keep track of workspaces for optimization in forward-only case
          if (workspaces_.size() < t + 1) {
            workspaces_.resize(t + 1);
          }
          workspaces_[t] = ws;

          // Create a specific timestep blob for this timestep. This is to
          // avoid conflicting timestep blobs when reusing workspaces, as with
          // the forward-only mode.
          std::string this_timestep_blob =
              timestep_blob_ + "_rnnexec_t" + c10::to_string(t);
          BlobGetMutableTensor(ws->CreateBlob(this_timestep_blob), CPU)->Resize(1);
          auto b = ws->GetBlob(this_timestep_blob);
          CAFFE_ENFORCE(b);
          BlobGetMutableTensor(b, CPU)->template mutable_data<int32_t>()[0] = t;

          // Copy the operators from template
          for (auto& template_rnn_op : timestep_ops_template_) {
            auto& rnn_op = template_rnn_op;

            // For ops that have the timestep blob as an input we need to
            // create a new operator definition with the timestep-specific
            // timestep blob. This is required to avoid race conditions when
            // multiple timesteps execute in paralle.
            if (rnn_op.has_timestep_blob) {
              OperatorDef op_copy = step_net_def_.op(rnn_op.order);

              for (int i = 0; i < op_copy.input_size(); i++) {
                if (op_copy.input(i) == timestep_blob_) {
                  op_copy.set_input(i, this_timestep_blob);
                }
              }

              rnn_op.op = CreateOperator(op_copy, ws);
              for (const auto& observer : observers_list) {
                std::unique_ptr<ObserverBase<OperatorStorage>> rnn_observer_copy =
                    observer.get()->rnnCopy(rnn_op.op.get(), rnn_op.order);
                if (rnn_observer_copy) {
                  rnn_op.op->AttachObserver(std::move(rnn_observer_copy));
                }
              }
            } else {
              // Optimization for forward-only models when we can share workspaces
              // with timesteps: then we can just copy the op reference.
              if (t > max_parallel_timesteps_ && max_parallel_timesteps_ > 0 &&
                  workspaces_[t - max_parallel_timesteps_] == ws) {
                rnn_op.op =
                    timestep_ops_[t - max_parallel_timesteps_][rnn_op.order].op;
              } else {
                // Otherwise, we need to create a brand new op with the workspace
                // owned by this timestep.
                rnn_op.op = CreateOperator(step_net_def_.op(rnn_op.order), ws);
                for (const auto& observer : observers_list) {
                  std::unique_ptr<ObserverBase<OperatorStorage>> rnn_observer_copy =
                      observer.get()->rnnCopy(rnn_op.op.get(), rnn_op.order);
                  if (rnn_observer_copy) {
                    rnn_op.op->AttachObserver(std::move(rnn_observer_copy));
                  }
                }
              }
            }
            rnn_op.op->DisableEvent();

            timestep_ops_[t].emplace_back(rnn_op);
          }
        }
        */
    }
    
    /**
      | Set limit for the number of timesteps
      | that run in parallel.
      | 
      | Useful for forward-only execution
      | when we rotate workspaces over timesteps,
      | i.e when timestep[t] and timestep[t
      | + p] have same workspace.
      |
      */
    #[inline] pub fn set_max_parallel_timesteps(&mut self, p: i32)  {
        
        todo!();
        /*
            max_parallel_timesteps_ = p;
        */
    }
    
    #[inline] pub fn num_observers_step_net(&mut self) -> usize {
        
        todo!();
        /*
            size_t num = 0;
        for (auto& ops_at_timestep_t : timestep_ops_) {
          for (auto& rnn_op : ops_at_timestep_t) {
            num += rnn_op.op->NumObservers();
          }
        }
        return num;
        */
    }

    /**
      | Utility method to check if any of the
      | op inputs or control inputs contain
      | given blob 'input'
      |
      */
    #[inline] pub fn has_input(&mut self, x: String, opidx: i32) -> bool {
        
        todo!();
        /*
            for (auto& inp : step_net_def_.op(opidx).input()) {
          if (inp == x) {
            return true;
          }
        }
        for (auto& inp : step_net_def_.op(opidx).control_input()) {
          if (inp == x) {
            return true;
          }
        }
        return false;
        */
    }

    /**
      | Return all outbound dependencies of
      | an op. Special case for rnn dependencies,
      | that are set in recurent_network_op.
      |
      */
    #[inline] pub fn op_deps(&mut self, i: i32) -> Vec<String> {
        
        todo!();
        /*
            std::vector<string> outs;
        auto& opdef = step_net_def_.op(i);
        for (string o : opdef.output()) {
          outs.push_back(o);
        };
        for (auto& arg : opdef.arg()) {
          if (arg.name().find("rnn_dependency") == 0) {
            outs.push_back(arg.s());
          }
        }
        return outs;
        */
    }
    
    /**
      | Calculate dependencies of this op,
      | for the ops following it in this timestep
      | and also for the next timestep. Removes
      | redundant dependencies.
      |
      */
    #[inline] pub fn infer_dependencies(&mut self, 
        start_i: i32,
        outputs: HashSet<String>,
        rnn_ops: &mut Vec<RNNNetOperator>,
        dep_ops: *mut HashSet<i32>)  {
        
        todo!();
        /*
            std::unordered_set<int> already_accounted_deps;
        int num_ops = step_net_def_.op_size();
        bool ignore_links = this->ignoreLinkDependencies();
        for (int j = 0; j < num_ops - 1 && !outputs.empty(); j++) {
          int i = (start_i + j) % num_ops;
          if (ignore_links && rnn_ops[i].link_op) {
            continue;
          }
          for (auto& outp : outputs) {
            if (has_input(outp, i)) {
              if (already_accounted_deps.find(i) == already_accounted_deps.end()) {
                dep_ops->insert(i);
              }

              // Now we can take the deps of this ops and not
              // add them anymore
              for (int odep : rnn_ops[i].dependencies) {
                already_accounted_deps.insert(odep);
              }
              for (string& dep_out : op_deps_[i]) {
                auto oit = outputs.find(dep_out);
                if (oit != outputs.end()) {
                  // This op produces output of the original op, so the dependency
                  // passed through that op
                  outputs.erase(oit);
                }
              }
              break;
            }
          }
        }
        */
    }
    
    /**
      | Add dependencies to ops in the next timestep
      | that would write an op that this op has
      | as an input or output. This is special
      | for RNNs, since we can have ops running
      | in different timesteps concurrently.
      | 
      | Also, we need to check ops that output
      | a blob that is input of of the op in question.
      |
      */
    #[inline] pub fn add_race_conflict_dependencies(&mut self, 
        opidx:   i32,
        rnn_ops: &mut Vec<RNNNetOperator>,
        dep_ops: *mut HashSet<i32>)  {

        todo!();
        /*
            for (int i = 0; i < rnn_ops.size(); i++) {
          if (i == opidx) {
            continue;
          }
          if (rnn_ops[i].link_op && this->ignoreLinkDependencies()) {
            continue;
          }
          for (auto& dep_blob : op_deps_[i]) {
            for (auto& inp : step_net_def_.op(opidx).input()) {
              if (inp == dep_blob) {
                dep_ops->insert(i);
                break;
              }
            }
            if (i < opidx) {
              for (auto& outp : step_net_def_.op(opidx).output()) {
                if (outp == dep_blob) {
                  dep_ops->insert(i);
                  break;
                }
              }
            }
          }
        }
        */
    }

    /**
      | Calculate the dependencies between
      | ops inside timestep and across timestep.
      | 
      | These are store in timestep_ops_ vector
      | that is copied for each timestep.
      |
      */
    #[inline] pub fn calculate_internal_dependencies(&mut self)  {
        
        todo!();
        /*
            for (int i = 0; i < step_net_def_.op_size(); i++) {
          timestep_ops_template_.push_back(RNNNetOperator(step_net_def_.op(i), i));
        }
        // Then see which outputs appear as inputs, and those are
        // the internal blobs.
        for (auto& rnn_op : timestep_ops_template_) {
          std::unordered_set<string> dep_outputs;
          for (auto& outp : op_deps_[rnn_op.order]) {
            dep_outputs.insert(outp);
          }

          // Add recurrent dependencies as 'outputs' for this op
          for (auto& outp : dep_outputs) {
            auto rit = recurrent_input_map_.find(outp);
            if (rit != recurrent_input_map_.end()) {
              dep_outputs.insert(rit->second);
            } else {
              dep_outputs.insert(outp);
            }
          }

          // Compute dependencies of this op.
          if (!rnn_op.link_op || !this->ignoreLinkDependencies()) {
            std::unordered_set<int> dependent_ops;
            infer_dependencies(
                rnn_op.order + 1,
                dep_outputs,
                timestep_ops_template_,
                &dependent_ops);

            // Race conditions arise when operator writes a blob that is
            // being read by another.
            if (!this->ignoreLinkDependencies()) {
              add_race_conflict_dependencies(
                rnn_op.order, timestep_ops_template_, &dependent_ops);
            }

            for (int i : dependent_ops) {
              rnn_op.dependencies.push_back(i);
            }

            // Sort in ascending order of dependency distance. If op
            // j > i, then distance is j - i. But if j < i, then distance
            // from i to j passes the timestep boundary and is j + num ops - i.
            std::sort(
                rnn_op.dependencies.begin(),
                rnn_op.dependencies.end(),
                [&](const int& a, const int& b) {
                  if (a < rnn_op.order && b < rnn_op.order) {
                    return a < b;
                  }
                  if (a >= rnn_op.order && b >= rnn_op.order) {
                    return a < b;
                  }
                  if (a >= rnn_op.order && b < rnn_op.order) {
                    return true;
                  }
                  return false;
                });
          }
        }

        // Update dependency counts
        for (auto& rnn_op : timestep_ops_template_) {
          for (int i : rnn_op.dependencies) {
            timestep_ops_template_[i].num_dynamic_inputs++;

            if (i > rnn_op.order) {
              timestep_ops_template_[i].frontier = false;
            } else {
              timestep_ops_template_[i].num_recurrent_inputs++;
            }
          }
        }
        // Find ops that have no recurrent inputs, and bind them
        // to the last op of the timestep. If there is only one op
        // in the step net, then it will depend on itself. Note that
        // we do not increase the dynamic input counter.
        for (auto& rnn_op : timestep_ops_template_) {
          if (rnn_op.num_dynamic_inputs == 0 && rnn_op.num_recurrent_inputs == 0) {
            if (rnn_op.link_op && this->ignoreLinkDependencies()) {
              continue;
            }
            timestep_ops_template_.back().dependencies.push_back(rnn_op.order);
          }
        }

        // compute parents
        for (auto& rnn_op : timestep_ops_template_) {
          for (int dep : rnn_op.dependencies) {
            timestep_ops_template_[dep].parents.push_back(rnn_op.order);
          }
        }
        AnalyzeOps();
        */
    }
    
    /**
      | For debug purposes, print the dependency
      | structure.
      | 
      | Set rnn_executor_debug=1 in the RecurrentNetworkOp
      | to enable.
      |
      */
    #[inline] pub fn print_info(&mut self, t: i32)  {
        
        todo!();
        /*
            auto& rnn_ops = timestep_ops_[t];

        LOG(INFO) << "Timestep: " << t;
        for (auto& rnn_op : rnn_ops) {
          auto& op = rnn_op.op;
          LOG(INFO) << "Operator " << rnn_op.order << ": " << op->type()
                    << " dep inputs:" << rnn_op.num_dynamic_inputs
                    << " rec inputs:" << rnn_op.num_recurrent_inputs
                    << " frontier: " << rnn_op.frontier;
          for (auto& inp : rnn_op.op->debug_def().input()) {
            LOG(INFO) << " ---- input: " << inp;
          }
          for (auto& outp : rnn_op.op->debug_def().output()) {
            LOG(INFO) << " ---- output: " << outp;
          }
          for (auto j : rnn_op.dependencies) {
            LOG(INFO) << " dep: " << j << ": " << rnn_ops[j].op->type();
          }
          for (auto j : rnn_op.parents) {
            LOG(INFO) << " parent: " << j << ": " << rnn_ops[j].op->type();
          }
        }

        LOG(INFO) << "recurrent_inputs:" << recurrent_input_map_;

        for (auto& rnn_op : rnn_ops) {
          LOG(INFO) << "Operator " << rnn_op.order;
          LOG(INFO) << ProtoDebugString(rnn_op.op->debug_def());
        }
        */
    }
    
    #[inline] pub fn analyze_ops(&mut self)  {
        
        todo!();
        /*
        
        */
    }
}

///---------------------------------
pub struct ThreadedRecurrentNetworkExecutor {
    base:                RecurrentNetworkExecutorBase,
    task_queue:          SimpleQueue<OpTask>,
    countdown:           Atomic<i32>,
    failed:              AtomicBool,
    finished_timesteps:  Atomic<i32>,
    num_ops:             i32,
    countdown_mtx:       RawMutex,
    cv:                  Condvar,
    workers:             Vec<Thread>,
    num_threads:         i32, // default = 4
}

impl ThreadedRecurrentNetworkExecutor {

    pub fn new(
        step_net_def:        &NetDef,
        recurrent_input_map: &mut HashMap<String,String>,
        timestep_blob:       String) -> Self {
    
        todo!();
        /*
            : RecurrentNetworkExecutorBase(step_net_def, recurrent_input_map, timestep_blob),
            failed_(false)
        */
    }
    
    #[inline] pub fn ignore_link_dependencies(&mut self) -> bool {
        
        todo!();
        /*
            return false;
        */
    }
    
    #[inline] pub fn set_num_threads(&mut self, n: i32)  {
        
        todo!();
        /*
            num_threads_ = n;
        */
    }
    
    #[inline] pub fn exec_range(&mut self, from: i32, to: i32)  {
        
        todo!();
        /*
        
        */
    }
}

impl Drop for ThreadedRecurrentNetworkExecutor {

    fn drop(&mut self) {
        todo!();
        /* 
        task_queue_.NoMoreJobs();
        VLOG(1) << "Joining workers.";
        for (auto& worker : workers_) {
          worker.join();
        }
       */
    }
}

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


impl ThreadedRecurrentNetworkExecutor {
    
    /**
      | Run forwardpass with T timesteps.
      |
      */
    #[inline] pub fn run(&mut self, t: i32) -> bool {
        
        todo!();
        /*
            CAFFE_ENFORCE_GE(T, 0, "Negative number of steps");
      if (T == 0) {
        return true;
      }

      CAFFE_ENFORCE(timestep_ops_.size() >= T);
      countdown_ = T * timestep_ops_[0].size();
      finished_timesteps_ = 0;

      CHECK(task_queue_.size() == 0);

      for (auto& rnn_op : timestep_ops_[0]) {
        // Launch "frontier"-ops first.
        if (rnn_op.frontier) {
          task_queue_.Push(OpTask(0, rnn_op.order, T, 1));
        }
      }

      _Exec();
      return true;
        */
    }
    
    /**
      | Run backward pass with T timesteps.
      |
      */
    #[inline] pub fn run_backwards(&mut self, t: i32) -> bool {
        
        todo!();
        /*
            CAFFE_ENFORCE_GE(T, 0, "Negative number of steps");
      if (T == 0) {
        return true;
      }

      CAFFE_ENFORCE(timestep_ops_.size() >= T);
      countdown_ = T * timestep_ops_[0].size();
      finished_timesteps_ = 0;

      // Frontier
      CHECK(task_queue_.size() == 0);

      for (auto& rnn_op : timestep_ops_[T - 1]) {
        if (rnn_op.frontier) {
          task_queue_.Push(OpTask(T - 1, rnn_op.order, T, -1));
        }
      }

      _Exec();
      return true;
        */
    }
    
    /**
      | Runs a single op and updates its dependencies
      | when finished.
      | 
      | If dependent ops are ready to run, adds
      | them to the task_queue.
      |
      */
    #[inline] pub fn run_op(&mut self, job: OpTask, thread_id: i32)  {
        
        todo!();
        /*
            bool first_timestep =
          ((job.forward() && job.timestep == 0) ||
           (job.backward() && job.timestep == job.T - 1));
      bool last_timestep =
          ((job.backward() && job.timestep == 0) ||
           (job.forward() && job.timestep == job.T - 1));
      auto& rnn_op = timestep_ops_[job.timestep][job.op_idx];
      if (rnn_op.num_dynamic_inputs > 0 && !rnn_op.frontier) {
        CAFFE_ENFORCE_EQ(
            rnn_op.proc_inputs,
            rnn_op.num_dynamic_inputs -
                first_timestep * rnn_op.num_recurrent_inputs,
            "Error at operator ",
            job.op_idx,
            " on timestep ",
            job.timestep,
            " T=",
            job.T,
            " first =",
            first_timestep);
      }

      // Reset input dependency counter
      rnn_op.proc_inputs = 0;

      // Run the operator
      rnn_op.op->Run();

      // Knock down dependencies and start next ops, if this
      // was last dependency fulfilled.
      for (int depidx : rnn_op.dependencies) {
        int t = job.timestep;
        bool for_next_timestep = depidx <= rnn_op.order;
        if (!last_timestep && for_next_timestep) {
          t += job.direction;
        } else if (for_next_timestep) {
          continue;
        }

        auto& dep_op = timestep_ops_[t][depidx];
        int proc_inputs = dep_op.proc_inputs.fetch_add(1) + 1;

        // Schedule next op, if this was the last dependency. Note that on
        // first timestep we don't have recurrent inputs.
        int num_req_inputs = dep_op.num_dynamic_inputs;
        if (first_timestep && !for_next_timestep) {
          num_req_inputs -= dep_op.num_recurrent_inputs;
        }

        if (proc_inputs == num_req_inputs || num_req_inputs == 0) {
          task_queue_.Push(OpTask(t, depidx, job.T, job.direction));
        }
      }

      // Decrement countdown: when at zero, we have run all ops and can
      // notify the caller thread.
      if (countdown_.fetch_sub(1) == 1) {
        CAFFE_ENFORCE_EQ(0, task_queue_.size());
        std::unique_lock<std::mutex> lk(countdown_mtx_);
        cv_.notify_one();
      }
        */
    }
    
    /**
      | Run-loop for executor threads: pop
      | tasks from task_queue and execute them
      | with RunOp().
      |
      */
    #[inline] pub fn worker_function(&mut self)  {
        
        todo!();
        /*
            size_t num_jobs = 0;
      static std::atomic<int> seq(0);
      int id = seq.fetch_add(1);

      while (!failed_) {
        OpTask job;
        if (!task_queue_.Pop(&job)) {
          break;
        }

        // Check for limited timestep parallelism, and if too many timesteps would
        // be started concurrently, return the task to task queue.
        if (max_parallel_timesteps_ > 0) {
          int t = (job.direction == 1 ? job.timestep : job.T - job.timestep + 1);
          if (t - finished_timesteps_ >= max_parallel_timesteps_) {
            // Return to queue
            task_queue_.Push(job);
            continue;
          }
        }

        try {
          RunOp(job, id);
          if (job.op_idx == timestep_ops_template_.size() - 1) {
            finished_timesteps_.fetch_add(1);
          }
          num_jobs++;
        } catch (::caffe2::EnforceNotMet& enf) {
          std::unique_lock<std::mutex> lk(countdown_mtx_);
          LOG(ERROR) << "Crash at thread " << id << " timestep " << job.timestep
                     << " op:" << ProtoDebugString(step_net_def_.op(job.op_idx))
                     << enf.what();
          task_queue_.NoMoreJobs();
          failed_ = true;
          cv_.notify_one();
          return;
        }
      }
      VLOG(1) << "Worker exiting, did run: " << num_jobs << " jobs";
        */
    }
    
    /**
      | Start worker threads if not started
      | yet, wait until all tasks finished,
      | or a failure. Called by Run() and RunBackwards().
      |
      */
    #[inline] pub fn exec(&mut self)  {
        
        todo!();
        /*
            CAFFE_ENFORCE_EQ(
          false, failed_, "Tried to execute a previously failed RNN executor");

      // Start threads if not started
      std::unique_lock<std::mutex> lk(countdown_mtx_);
      while (workers_.size() < num_threads_) {
        VLOG(1) << "Start RNN worker " << workers_.size() << " / " << num_threads_;
        workers_.push_back(
            std::thread(&ThreadedRecurrentNetworkExecutor::WorkerFunction, this));
      }

      // Wait until threads finish.
      Timer t;
      while (!failed_ && countdown_ > 0) {
        cv_.wait_for(lk, std::chrono::seconds(30), [&] {
          // Log if we are still running, so that we catch deadlocks.. there
          // should not be any deadlocks, but...
          if (t.Seconds() > 10) {
            LOG(INFO) << "RNN Executor still running, remaining ops: "
                      << countdown_;
          }
          return failed_ || countdown_ == 0;
        });
      }

      CAFFE_ENFORCE_EQ(
          false,
          failed_,
          "RNN executor encountered failure. See prior error logs for details.");
        */
    }
}
