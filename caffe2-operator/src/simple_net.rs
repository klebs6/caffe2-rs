crate::ix!();

/**
  | This is the very basic structure you
  | need to run a network - all it does is simply
  | to run everything in sequence.
  | 
  | If you want more fancy control such as
  | a DAG-like execution, check out other
  | better net implementations.
  |
  */
pub struct SimpleNet {
    base:      NetBase,
    operators: Vec<Box<OperatorStorage>>,
}

define_bool!{
    caffe2_simple_net_benchmark_run_whole_net,
    true,
    "If false, whole net passes won't be performed"}

impl SimpleNet {
    
    pub fn new(net_def: &Arc<NetDef>, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : NetBase(net_def, ws) 

      VLOG(1) << "Constructing SimpleNet " << net_def->name();
      const bool net_def_has_device_option = net_def->has_device_option();
      // Initialize the operators
      for (int idx = 0; idx < net_def->op_size(); ++idx) {
        const auto& operator_def = net_def->op(idx);
        VLOG(1) << "Creating operator " << operator_def.name() << ": "
                << operator_def.type();
        std::unique_ptr<OperatorStorage> op{nullptr};
        if (net_def_has_device_option) {
          // In the case when net def specifies device option, final device option
          // will be equal to merge of operator and net def device options, with
          // preference to settings from the operator.
          OperatorDef temp_def(operator_def);

          DeviceOption temp_dev(net_def->device_option());
          temp_dev.MergeFrom(operator_def.device_option());

          temp_def.mutable_device_option()->CopyFrom(temp_dev);
          op = CreateOperator(temp_def, ws, idx);
        } else {
          op = CreateOperator(operator_def, ws, idx);
          op->set_debug_def(
              std::shared_ptr<const OperatorDef>{net_def, &(net_def->op(idx))});
        }
        operators_.emplace_back(std::move(op));
      }
        */
    }

    #[inline] pub fn supports_async(&mut self) -> bool {
        
        todo!();
        /*
            return false;
        */
    }
    
    /**
      | This returns a list of pointers to objects
      | stored in unique_ptrs.
      | 
      | Used by Observers.
      | 
      | Think carefully before using.
      |
      */
    #[inline] pub fn get_operators(&self) -> Vec<OperatorStorage> {
        
        todo!();
        /*
            vector<OperatorStorage*> op_list;
        for (auto& op : operators_) {
          op_list.push_back(op.get());
        }
        return op_list;
        */
    }
    
    #[inline] pub fn run(&mut self) -> bool {
        
        todo!();
        /*
            StartAllObservers();
      VLOG(1) << "Running net " << name_;
      for (auto& op : operators_) {
        VLOG(1) << "Running operator " << op->debug_def().name() << "("
                << op->debug_def().type() << ").";
    #ifdef CAFFE2_ENABLE_SDT
        const auto& op_name = op->debug_def().name().c_str();
        const auto& op_type = op->debug_def().type().c_str();
        auto* op_ptr = op.get();
        const auto& net_name = name_.c_str();
        CAFFE_SDT(operator_start, net_name, op_name, op_type, op_ptr);
    #endif
        bool res = op->Run();
    #ifdef CAFFE2_ENABLE_SDT
        CAFFE_SDT(operator_done, net_name, op_name, op_type, op_ptr);
    #endif
        // workaround for async cpu ops, we need to explicitly wait for them
        if (res && op->HasAsyncPart() &&
            op->device_option().device_type() == PROTO_CPU) {
          op->Finish();
          res = op->event().Query() == EventStatus::EVENT_SUCCESS;
        }
        if (!res) {
          LOG(ERROR) << "Operator failed: " << ProtoDebugString(op->debug_def());
          return false;
        }
      }
      StopAllObservers();
      return true;
        */
    }
    
    #[inline] pub fn run_async(&mut self) -> bool {
        
        todo!();
        /*
            return Run();
        */
    }
    
    #[inline] pub fn test_benchmark(&mut self, 
        warmup_runs:    i32,
        main_runs:      i32,
        run_individual: bool) -> Vec<f32> {

        todo!();
        /*
            /* Use std::cout because logging may be disabled */
      std::cout << "Starting benchmark." << std::endl;
      std::cout << "Running warmup runs." << std::endl;
      CAFFE_ENFORCE(
          warmup_runs >= 0,
          "Number of warm up runs should be non negative, provided ",
          warmup_runs,
          ".");
      for (int i = 0; i < warmup_runs; ++i) {
        CAFFE_ENFORCE(Run(), "Warmup run ", i, " has failed.");
      }

      std::cout << "Main runs." << std::endl;
      CAFFE_ENFORCE(
          main_runs >= 0,
          "Number of main runs should be non negative, provided ",
          main_runs,
          ".");
      Timer timer;
      auto millis = timer.MilliSeconds();
      if (FLAGS_caffe2_simple_net_benchmark_run_whole_net) {
        for (int i = 0; i < main_runs; ++i) {
          CAFFE_ENFORCE(Run(), "Main run ", i, " has failed.");
        }
        millis = timer.MilliSeconds();
        std::cout << "Main run finished. Milliseconds per iter: "
                  << millis / main_runs
                  << ". Iters per second: " << 1000.0 * main_runs / millis
                  << std::endl;
      }

      auto operators = GetOperators();
      auto results = IndividualMetrics(operators);
      if (run_individual) {
        for (int i = 0; i < main_runs; ++i) {
          results.RunOpsWithProfiling();
        }
        results.PrintOperatorProfilingResults();
      }
      // We will reuse time_per_op to return the result of BenchmarkNet.
      std::vector<float> time_per_op(results.GetTimePerOp());
      for (size_t i = 0; i < time_per_op.size(); ++i) {
        time_per_op[i] /= main_runs;
      }
      if (FLAGS_caffe2_simple_net_benchmark_run_whole_net) {
        time_per_op.insert(time_per_op.begin(), millis / main_runs);
      }
      return time_per_op;
        */
    }
}

