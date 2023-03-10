crate::ix!();


declare_string!{caffe2_net_async_tracing_filepath}
declare_string!{caffe2_net_async_names_to_trace}
declare_int!{caffe2_net_async_tracing_nth}

pub struct TracerEvent {
    op_id:         i32,                   // default = -1
    task_id:       i32,                   // default = -1
    stream_id:     i32,                   // default = -1
    name:          *const u8,             // default = nullptr
    category:      *const u8,             // default = nullptr
    timestamp:     i64,                   // default = -1.0
    is_beginning:  bool,                  // default = false
    thread_label:  i64,                   // default = -1
    tid:           std::thread::ThreadId,
    iter:          i32,                   // default = -1
}

pub enum TracingField {
    TRACE_OP,
    TRACE_TASK,
    TRACE_STREAM,
    TRACE_THREAD,
    TRACE_NAME,
    TRACE_CATEGORY,
    TRACE_ITER,
}

pub enum TracingMode {
  EVERY_K_ITERATIONS,
  GLOBAL_TIMESLICE,
}

///---------------------------------
pub struct TracingConfig {
    mode:                      TracingMode, // {TracingMode::EVERY_K_ITERATIONS};
    filepath:                  String,      // {"/tmp"};

    /// for TracingMode::EVERY_K_ITERATIONS
    trace_every_nth_batch:     i64, // default = 100
    dump_every_nth_batch:      i64, // default = 10000

    // for TracingMode::GLOBAL_TIMESLICE
    trace_every_n_ms:          i64, // = 2 * 60 * 1000; // 2min
    trace_for_n_ms:            i64, // default = 1000 // 1sec
}

///---------------------------------
pub struct Tracer {
    net:           *const NetBase, // default = nullptr
    filename:      String,
    events:        Vec<TracerEvent>,
    tracer_mutex:  parking_lot::RawMutex,
    enabled:       bool, // default = false
    timer:         Timer,
    iter:          i32,
    dumping_iter:  i32,
    config:        TracingConfig,
}

///---------------------------------
pub struct TracerGuard {
    enabled:  bool, // default = false
    event:    TracerEvent,
    tracer:   *mut Tracer,
}

impl TracerGuard {

    #[inline] pub fn init_from_tracer(&mut self, tracer: *mut Tracer)  {
        
        todo!();
        /*
        
        */
    }
    
    #[inline] pub fn add_argument_with_args<T, Args>(&mut self, 
        field: TracingField,
        value: &T,
        args:  &Args)  {

        todo!();
        /*
            addArgument(field, value);
        addArgument(args...);
        */
    }

}

macro_rules! trace_name_concatenate {
    ($s1:ident, $s2:ident) => {
        todo!();
        /*
                s1##s2
        */
    }
}

macro_rules! trace_anonymous_name {
    ($str:ident) => {
        todo!();
        /*
                TRACE_NAME_CONCATENATE(str, __LINE__)
        */
    }
}

macro_rules! trace_event_init {
    ($($arg:ident),*) => {
        todo!();
        /*
        
          TRACE_ANONYMOUS_NAME(trace_guard).init(tracer_.get());      
          TRACE_ANONYMOUS_NAME(trace_guard).addArgument(__VA_ARGS__); 
          TRACE_ANONYMOUS_NAME(trace_guard).recordEventStart();
        */
    }
}

/**
  | Supposed to be used only once per scope
  | in AsyncNetBase-derived nets
  |
  */
macro_rules! trace_event {
    (, $($arg:ident),*) => {
        todo!();
        /*
        
          tracing::TracerGuard TRACE_ANONYMOUS_NAME(trace_guard); 
          if (tracer_ && tracer_->isEnabled()) {                  
            TRACE_EVENT_INIT(__VA_ARGS__)                         
          }
        */
    }
}

macro_rules! trace_event_if {
    ($cond:ident, $($arg:ident),*) => {
        todo!();
        /*
        
          tracing::TracerGuard TRACE_ANONYMOUS_NAME(trace_guard); 
          if (tracer_ && tracer_->isEnabled() && (cond)) {        
            TRACE_EVENT_INIT(__VA_ARGS__)                         
          }
        */
    }
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

#[inline] pub fn get_counter_for_net_name(net_name: &String) -> i32 {
    
    todo!();
    /*
        // Append a unique number suffix because there could be multiple instances
      // of the same net and we want to uniquely associate each instance with
      // a profiling trace.
      static std::unordered_map<std::string, int> net_name_to_counter;
      static std::mutex map_mutex;
      std::unique_lock<std::mutex> map_lock(map_mutex);
      int counter = net_name_to_counter[net_name] + 1;
      net_name_to_counter[net_name] = counter;
      return counter;
    */
}

impl Tracer {
    
    pub fn new(
        net:      *const NetBase,
        net_name: &String,
        config:   TracingConfig) -> Self {
    
        todo!();
        /*
            : net_(net),
          filename_(net_name),
          iter_(0),
          dumping_iter_(0),
          config_(config) 

      std::replace(filename_.begin(), filename_.end(), '/', '_');
      filename_ = this->config().filepath + "/" + filename_ + "_id_" +
          c10::to_string(getCounterForNetName(net_name));
      timer_.Start();
        */
    }

    #[inline] pub fn config(&mut self) -> &TracingConfig {
        
        todo!();
        /*
            return config_;
        */
    }
    
    #[inline] pub fn record_event(&mut self, event: &TracerEvent)  {
        
        todo!();
        /*
            std::lock_guard<std::mutex> lock(tracer_mutex_);
      events_.push_back(event);
        */
    }

    /// Special handling of shard blob annotations
    #[inline] pub fn op_trace_name(&mut self, op: *const OperatorStorage) -> String {
        
        todo!();
        /*
            int unique_shard_id =
          op->has_debug_def() ? getUniqueShardId(op->debug_def()) : -1;
      if (unique_shard_id != -1) {
        return op->type() + ":" + c10::to_string(unique_shard_id);
      } else {
        return op->type();
      }
        */
    }
    
    #[inline] pub fn op_blobs_info(&mut self, op: &OperatorStorage) -> String {
        
        todo!();
        /*
            std::string blobs_info;
      if (op.has_debug_def()) {
        blobs_info += "I: ";
        const auto& op_def = op.debug_def();
        for (const auto& input : op_def.input()) {
          blobs_info += input + "; ";
        }
        blobs_info += "O: ";
        for (const auto& output : op_def.output()) {
          blobs_info += output + "; ";
        }
      }
      return blobs_info;
        */
    }
    
    #[inline] pub fn serialize_event(&mut self, event: &TracerEvent) -> String {
        
        todo!();
        /*
            std::stringstream serialized_event;
      serialized_event << std::fixed;
      serialized_event << "{\n";
      serialized_event << " \"ts\": " << event.timestamp_ << ",\n";
      serialized_event << " \"pid\": 0,\n"; // not using pid field
      if (event.thread_label_ >= 0) {
        serialized_event << " \"tid\": " << event.thread_label_ << ",\n";
      } else {
        serialized_event << " \"tid\": " << event.tid_ << ",\n";
      }

      if (event.is_beginning_) {
        std::unordered_map<std::string, int> int_args;
        std::unordered_map<std::string, std::string> string_args;
        if (event.name_) {
          serialized_event << " \"name\": \"" << event.name_ << "\",\n";
        } else if (event.op_id_ >= 0) {
          auto* op = net_->GetOperators().at(event.op_id_);
          serialized_event << " \"name\": \"" << opTraceName(op) << "\",\n";
        } else {
          serialized_event << " \"name\": \"n/a\",\n";
        }

        if (event.category_) {
          serialized_event << " \"cat\": \"" << event.category_ << "\",\n";
        } else {
          serialized_event << " \"cat\": \"net\",\n";
        }

        if (event.op_id_ >= 0) {
          auto* op = net_->GetOperators().at(event.op_id_);
          int_args["op_id"] = event.op_id_;
          int_args["device_type"] = op->device_option().device_type();
          int_args["device_id"] = DeviceId(op->device_option());
          string_args["blobs"] = opBlobsInfo(*op);
        }

        if (event.task_id_ >= 0) {
          int_args["task_id"] = event.task_id_;
        }

        if (event.iter_ >= 0) {
          int_args["iter_id"] = event.iter_;
        }

        if (event.stream_id_ >= 0) {
          int_args["stream_id"] = event.stream_id_;
        }

        serialized_event << " \"ph\": \"B\"";
        if (!int_args.empty() || !string_args.empty()) {
          serialized_event << ",\n \"args\": {\n";
          auto left_to_output = int_args.size() + string_args.size();
          for (const auto& kv : int_args) {
            serialized_event << "  \"" << kv.first << "\": " << kv.second;
            --left_to_output;
            if (left_to_output > 0) {
              serialized_event << ",\n";
            }
          }
          for (const auto& kv : string_args) {
            serialized_event << "  \"" << kv.first << "\": \"" << kv.second << "\"";
            --left_to_output;
            if (left_to_output > 0) {
              serialized_event << ",\n";
            }
          }
          serialized_event << "\n }";
        }
      } else {
        serialized_event << " \"ph\": \"E\"\n";
      }
      serialized_event << "\n}";

      return serialized_event.str();
        */
    }

    /// fix occasional cases with zero duration events
    #[inline] pub fn linearize_events(&mut self)  {
        
        todo!();
        /*
            std::unordered_map<long, long> time_offsets;
      std::unordered_map<long, long> last_times;
      std::hash<std::thread::id> hasher;
      const long time_eps = 1; // us
      for (auto& event : events_) {
        long tid =
            (event.thread_label_ >= 0) ? event.thread_label_ : hasher(event.tid_);
        auto event_ts = event.timestamp_;
        if (last_times.count(tid)) {
          event_ts += time_offsets[tid];
          CAFFE_ENFORCE(event_ts >= last_times[tid]);
          if (event_ts <= last_times[tid] + time_eps) {
            event_ts += time_eps;
            time_offsets[tid] += time_eps;
          } else if (event_ts > last_times[tid] + 2 * time_eps) {
            long eps_len = (event_ts - last_times[tid]) / time_eps;
            if (time_offsets[tid] >= time_eps * (eps_len - 1)) {
              time_offsets[tid] -= time_eps * (eps_len - 1);
              event_ts -= time_eps * (eps_len - 1);
            } else {
              event_ts -= time_offsets[tid];
              time_offsets[tid] = 0;
            }
          }
          event.timestamp_ = event_ts;
          last_times[tid] = event_ts;
        } else {
          last_times[tid] = event_ts;
          time_offsets[tid] = 0;
        }
      }
        */
    }
    
    #[inline] pub fn rename_threads(&mut self)  {
        
        todo!();
        /*
            std::unordered_map<long, int> tids;
      std::unordered_map<int, int> numa_counters;
      std::unordered_map<long, int> tid_to_numa;
      std::hash<std::thread::id> hasher;
      const long numa_multiplier = 1000000000;
      for (auto& event : events_) {
        if (event.thread_label_ >= 0 || event.op_id_ < 0) {
          continue;
        }
        auto* op = net_->GetOperators().at(event.op_id_);
        if (!op->device_option().has_numa_node_id()) {
          continue;
        }
        int numa_node_id = op->device_option().numa_node_id();
        CAFFE_ENFORCE_GE(numa_node_id, 0, "Invalid NUMA node id: ", numa_node_id);
        long tid = hasher(event.tid_);

        if (!tid_to_numa.count(tid)) {
          tid_to_numa[tid] = numa_node_id;
        } else {
          CAFFE_ENFORCE_EQ(tid_to_numa[tid], numa_node_id);
        }

        if (!numa_counters.count(numa_node_id)) {
          numa_counters[numa_node_id] = 1;
        }
        if (!tids.count(tid)) {
          tids[tid] = numa_counters[numa_node_id]++;
        }
        event.thread_label_ = numa_multiplier * (numa_node_id + 1) + tids[tid];
      }
        */
    }
    
    #[inline] pub fn set_enabled(&mut self, enabled: bool)  {
        
        todo!();
        /*
            enabled_ = enabled;
        */
    }
    
    #[inline] pub fn is_enabled(&self) -> bool {
        
        todo!();
        /*
            return enabled_;
        */
    }
    
    #[inline] pub fn bump_iter(&mut self) -> i32 {
        
        todo!();
        /*
            return iter_++;
        */
    }
    
    #[inline] pub fn get_iter(&mut self) -> i32 {
        
        todo!();
        /*
            return iter_;
        */
    }
    
    #[inline] pub fn bump_dumping_iter(&mut self) -> i32 {
        
        todo!();
        /*
            return dumping_iter_++;
        */
    }
    
    /**
      | Dump the tracing result to file with
      | given suffix, and then clear current
      | events.
      |
      */
    #[inline] pub fn dump_tracing_result_and_clear_events(&mut self, file_suffix: &String)  {
        
        todo!();
        /*
            if (events_.empty() || filename_.empty()) {
        return;
      }
      linearizeEvents();
      renameThreads();
      std::stringstream serialized;
      serialized << "[\n";
      for (size_t idx = 0; idx < events_.size(); ++idx) {
        serialized << serializeEvent(events_[idx]);
        if (idx != events_.size() - 1) {
          serialized << ",\n";
        }
      }
      serialized << "\n]\n";

      auto output_file_name = filename_ + "_iter_" + file_suffix + ".json";
      LOG(INFO) << "Dumping profiling result file to " << output_file_name;
      WriteStringToFile(serialized.str(), output_file_name.c_str());
      events_.clear();
        */
    }
}

impl Drop for Tracer {
    fn drop(&mut self) {
        todo!();
        /* 
      dumpTracingResultAndClearEvents("final_batch");
 */
    }
}

thread_local!{
    pub static current_tracer_guard: *mut TracerGuard = todo!();
}

impl TracerGuard {
    
    #[inline] pub fn init(&mut self, tracer: *mut Tracer)  {
        
        todo!();
        /*
            enabled_ = tracer && tracer->isEnabled();
      if (enabled_) {
        current_tracer_guard = this;
      }
      tracer_ = tracer;
        */
    }
    
    #[inline] pub fn add_argument_with_value(
        &mut self, 
        field: TracingField, 
        value: *const u8)  
    {
        todo!();
        /*
            switch (field) {
        case TRACE_NAME: {
          event_.name_ = value;
          break;
        }
        case TRACE_CATEGORY: {
          event_.category_ = value;
          break;
        }
        default: {
          CAFFE_THROW("Unexpected tracing string field ", field);
        }
      }
        */
    }
    
    #[inline] pub fn add_argument(&mut self, field: TracingField, value: i32)  {
        
        todo!();
        /*
            switch (field) {
        case TRACE_OP: {
          event_.op_id_ = value;
          break;
        }
        case TRACE_TASK: {
          event_.task_id_ = value;
          break;
        }
        case TRACE_STREAM: {
          event_.stream_id_ = value;
          break;
        }
        case TRACE_THREAD: {
          event_.thread_label_ = value;
          break;
        }
        case TRACE_ITER: {
          event_.iter_ = value;
          break;
        }
        default: {
          CAFFE_THROW("Unexpected tracing int field ", field);
        }
      }
        */
    }
    
    #[inline] pub fn record_event_start(&mut self)  {
        
        todo!();
        /*
            if (enabled_) {
        if (event_.thread_label_ < 0) {
          event_.tid_ = std::this_thread::get_id();
        }
        event_.is_beginning_ = true;
        event_.timestamp_ = (long)caffe2::round(tracer_->timer_.MicroSeconds());
        tracer_->recordEvent(event_);
      }
        */
    }
}

impl Drop for TracerGuard {
    fn drop(&mut self) {
        todo!();
        /* 
      if (enabled_) {
        event_.is_beginning_ = false;
        event_.timestamp_ = (long)caffe2::round(tracer_->timer_.MicroSeconds());
        tracer_->recordEvent(event_);
        if (current_tracer_guard == this) {
          current_tracer_guard = nullptr;
        }
      }
 */
    }
}

impl TracerGuard {
    
    #[inline] pub fn disable(&mut self)  {
        
        todo!();
        /*
            enabled_ = false;
        */
    }
    
    #[inline] pub fn get_current_tracer_guard(&mut self) -> *mut TracerGuard {
        
        todo!();
        /*
            return current_tracer_guard;
        */
    }
}

/**
  | Extract the shard id from name of the form
  | "...shard:123..."
  |
  | Return -1 if there is no shard found
  */
#[inline] pub fn extract_shard_id(name: &String) -> i32 {
    
    todo!();
    /*
        const std::string kShard = "shard:";
      // We sometimes have multiple shards, but actually need the last one, hence
      // using rfind here. Hacky but it works till we pass shard id in graph
      // metadata.
      auto pos = name.rfind(kShard);
      if (pos != std::string::npos) {
        int left_pos = pos + kShard.length();
        int right_pos = left_pos;
        while (right_pos < name.length() && isdigit(name[right_pos])) {
          right_pos++;
        }
        return c10::stoi(name.substr(left_pos, right_pos - left_pos));
      } else {
        return -1;
      }
    */
}

/**
  | Return unique shard id, or -1 if it is
  | not unique.
  |
  */
#[inline] pub fn get_unique_shard_id(op_def: &OperatorDef) -> i32 {
    
    todo!();
    /*
        int unique_shard_id = -1;
      for (const auto& names : {op_def.input(), op_def.output()}) {
        for (const auto& name : names) {
          int shard_id = extractShardId(name);
          if (shard_id != -1) {
            if (unique_shard_id != -1) {
              return -1;
            }
            unique_shard_id = shard_id;
          }
        }
      }
      return unique_shard_id;
    */
}

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

#[inline] pub fn create(net: *const NetBase, net_name: &String) -> Arc<Tracer> {
    
    todo!();
    /*
        // Enable the tracer if the net has the "enable_tracing" argument set OR
      // if the command line option includes the net name option in the list of
      // traceable nets.
      bool trace_net = hasEnableTracingFlag(net) || isTraceableNetName(net_name);
      return trace_net
          ? std::make_shared<Tracer>(net, net_name, getTracingConfigFromNet(net))
          : nullptr;
    */
}

#[inline] pub fn start_iter(tracer: &Arc<Tracer>) -> bool {
    
    todo!();
    /*
        if (!tracer) {
        return false;
      }
      auto iter = tracer->bumpIter();
      bool is_enabled;
      bool should_dump;
      if (tracer->config().mode == TracingMode::EVERY_K_ITERATIONS) {
        is_enabled = iter % tracer->config().trace_every_nth_batch == 0;
        should_dump = iter % tracer->config().dump_every_nth_batch == 0;
      } else {
        using namespace std::chrono;
        auto ms =
            duration_cast<milliseconds>(system_clock::now().time_since_epoch())
                .count();
        is_enabled = (ms % tracer->config().trace_every_n_ms) <
            tracer->config().trace_for_n_ms;
        // dump just after disabled tracing
        should_dump = tracer->isEnabled() && !is_enabled;
      }
      tracer->setEnabled(is_enabled);
      if (should_dump) {
        int dumping_iter = tracer->bumpDumpingIter();
        tracer->dumpTracingResultAndClearEvents(c10::to_string(dumping_iter));
      }
      return is_enabled;
    */
}
