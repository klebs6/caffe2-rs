crate::ix!();

pub type NetObserver        = ObserverBase<NetBase>;
pub type NetObserverCreator = fn(n: *mut NetBase) -> Box<NetObserver>;

pub struct ExecutorHelper { }

declare_registry!{
    NetRegistry,
    NetBase,
    Arc<NetDef>,
    Workspace
}

/**
  | Net is a thin struct that owns all the
  | operators together with the operator
  | contexts.
  |
  */
pub struct NetBase {
    base: Observable<NetBase>,

    external_input:  Vec<String>,
    external_output: Vec<String>,
    name:            String,
    events:          Vec<*const Event>,
    net_def:         Arc<NetDef>,
}

pub trait NetBaseTrait {

    fn supports_async() -> bool;

    /**
      | Used to attach Observers to operators
      | of a Net
      | 
      | Returns pointers to objects owned with
      | unique_ptrs.
      | 
      | Use with caution.
      |
      */
    fn get_operators() -> Vec<*mut OperatorStorage>;
}

impl NetBase {
    
    #[inline] pub fn events(&self) -> &Vec<*const Event> {
        
        todo!();
        /*
            return events_;
        */
    }
    
    #[inline] pub fn external_output(&self) -> &Vec<String> {
        
        todo!();
        /*
            return external_output_;
        */
    }
    
    #[inline] pub fn external_input(&self) -> &Vec<String> {
        
        todo!();
        /*
            return external_input_;
        */
    }
    
    #[inline] pub fn name(&self) -> &String {
        
        todo!();
        /*
            return name_;
        */
    }
    
    #[inline] pub fn debug_def(&self) -> &NetDef {
        
        todo!();
        /*
            CAFFE_ENFORCE(has_debug_def(), "net_def was null!");
        return *net_def_;
        */
    }
    
    #[inline] pub fn has_debug_def(&self) -> bool {
        
        todo!();
        /*
            return net_def_ != nullptr;
        */
    }
    
    #[inline] pub fn do_run_async(&mut self) -> bool {
        
        todo!();
        /*
            CAFFE_THROW("Not implemented");
        */
    }
    
    #[inline] pub fn handle_run_error(&mut self) -> bool {
        
        todo!();
        /*
            for (const Event* event : events_) {
          if (event->Query() != EventStatus::EVENT_SUCCESS) {
            CAFFE_THROW(event->ErrorMessage());
          }
        }
        return true;
        */
    }
    
    #[inline] pub fn wait(&mut self)  {
        
        todo!();
        /*
            // by default just wait till all events are finished
        for (const auto& event : events_) {
          event->Finish();
        }
        */
    }
    
    #[inline] pub fn run(&mut self) -> bool {
        
        todo!();
        /*
            if (!RunAsync()) {
          LOG(ERROR) << "Failed to execute async run";
          return false;
        }
        Wait();
        return handleRunError();
        */
    }
    
    pub fn new(
        def:    &Arc<NetDef>, 
        unused: *mut Workspace) -> Self 
    {
        todo!();
        /*
            : external_input_(
              def->external_input().begin(),
              def->external_input().end()),
          external_output_(
              def->external_output().begin(),
              def->external_output().end()),
          name_(def->name()),
          net_def_(def) 

      static GlobalInitIsCalledGuard guard;
      C10_LOG_API_USAGE_ONCE("caffe2.net.create");
      // Check that node_name is empty for all ops
      for (const OperatorDef& op : def->op()) {
        if (op.has_device_option()) {
          CAFFE_ENFORCE(
              !op.device_option().has_node_name(),
              "node_name must be empty for all operators at execution time.");
        }
      }

      // Go through the operators and make sure that blobs are correctly made.
      std::set<string> known_blobs(
          external_input_.begin(), external_input_.end());
      std::set<string> remaining_output(
          external_output_.begin(), external_output_.end());
      for (const auto& blob : known_blobs) {
        remaining_output.erase(blob);
      }
      for (const OperatorDef& op : def->op()) {
        for (const string& in : op.input()) {
          if (!known_blobs.count(in)) {
            if (external_input_.size()) {
              CAFFE_THROW(
                  "op ",
                  op.type(),
                  ": Source for input ",
                  in,
                  " is unknown for net ",
                  def->name(),
                  ", operator ",
                  ProtoDebugString(op));
            } else {
              // If we are not declaring input and output, we will simply VLOG it
              // for debugging purposes.
              VLOG(1) << "op " << op.type() << ": input " << in << " is unknown.";
            }
          }
        }
        for (const string& out : op.output()) {
          known_blobs.insert(out);
          remaining_output.erase(out);
        }
      }
      // Finally, check if all declared outputs are being created.
      CAFFE_ENFORCE(
          remaining_output.size() == 0,
          "Some of the blobs are declared as output but never produced by the "
          "net ",
          def->name(),
          ", the first one is ",
          *remaining_output.begin());
        */
    }
    
    #[inline] pub fn run_async(&mut self) -> bool {
        
        todo!();
        /*
            for (auto& op : GetOperators()) {
        op->ResetEvent();
      }
      return DoRunAsync();
        */
    }
    
    #[inline] pub fn cancel(&mut self)  {
        
        todo!();
        /*
            for (auto& op : GetOperators()) {
        op->Cancel();
      }
        */
    }

    /**
      | Benchmarks a network for one individual
      | run so that we can feed new inputs on additional
      | calls.
      | 
      | This function returns the number of
      | microseconds spent during the benchmark
      | 
      | benchmark an individual run so that
      | we can FeedBlobs with new inputs no warmup
      | return time taken in microseconds
      |
      */
    #[inline] pub fn test_benchmark_one_run(&mut self) -> f32 {
        
        todo!();
        /*
            Timer timer;
      CAFFE_ENFORCE(Run(), "Run has failed.");
      return timer.MicroSeconds();
        */
    }
    
    /**
      | Benchmarks a network.
      | 
      | This function returns a vector of float
      | recording the number of milli- seconds
      | spent during the benchmark.
      | 
      | The 0-th item is the time spent per each
      | network run, and if a net instantiation
      | supports run_individual, the remainder
      | of the vector returns the number of milliseconds
      | spent per operator.
      |
      */
    #[inline] pub fn test_benchmark(
        &mut self, 
        warmup_runs:    i32,
        main_runs:      i32,
        run_individual: bool) -> Vec<f32> 
    {
        todo!();
        /*
            LOG(INFO) << "Starting benchmark, running warmup runs";
      CAFFE_ENFORCE(
          warmup_runs >= 0,
          "Number of warm up runs should be non negative, provided ",
          warmup_runs);
      for (int run_idx = 0; run_idx < warmup_runs; ++run_idx) {
        CAFFE_ENFORCE(Run(), "Warmup run ", run_idx, " has failed");
      }

      LOG(INFO) << "Running main runs";
      CAFFE_ENFORCE(
          main_runs >= 0,
          "Number of main runs should be non negative, provided ",
          main_runs);

      Timer timer;
      for (int run_idx = 0; run_idx < main_runs; ++run_idx) {
        CAFFE_ENFORCE(Run(), "Main run ", run_idx, " has failed");
      }
      auto millis = timer.MilliSeconds();
      LOG(INFO) << "Main runs finished. Milliseconds per iter: "
                << millis / main_runs
                << ". Iters per second: " << 1000.0 * main_runs / millis;

      if (run_individual) {
        LOG(INFO) << "Net does not support per-op benchmark; "
                     "to run it, switch to a simple net type";
      }
      return std::vector<float>{millis / main_runs};
        */
    }
}


