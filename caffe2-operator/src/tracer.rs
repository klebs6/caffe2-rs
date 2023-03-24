crate::ix!();

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

impl Drop for Tracer {

    fn drop(&mut self) {
        todo!();
        /* 
          dumpTracingResultAndClearEvents("final_batch");
         */
    }
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
