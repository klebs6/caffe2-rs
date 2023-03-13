crate::ix!();


pub struct ProfDAGStats {
    sum:     f32,
    sqrsum:  f32,
    cnt:     usize,
}

impl Default for ProfDAGStats {
    
    fn default() -> Self {
        todo!();
        /*
            : sum_(0.0), sqrsum_(0.0), cnt_(0
        */
    }
}

impl ProfDAGStats {
    
    pub fn new(time_ms: f32) -> Self {
    
        todo!();
        /*
            : sum_(time_ms), sqrsum_(time_ms * time_ms), cnt_(1)
        */
    }
    
    #[inline] pub fn compute_moments(&self) -> (f32,f32) {
        
        todo!();
        /*
            CAFFE_ENFORCE_GT(cnt_, 0U);
        float mean = sum_ / cnt_;
        float stddev = std::sqrt(std::abs(sqrsum_ / cnt_ - mean * mean));
        return {mean, stddev};
        */
    }
    
    #[inline] pub fn sum(&self) -> f32 {
        
        todo!();
        /*
            return sum_;
        */
    }
    
    #[inline] pub fn sqrsum(&self) -> f32 {
        
        todo!();
        /*
            return sqrsum_;
        */
    }
    
    #[inline] pub fn cnt(&self) -> usize {
        
        todo!();
        /*
            return cnt_;
        */
    }
}

impl AddAssign<&ProfDAGStats> for ProfDAGStats {
    
    fn add_assign(&mut self, other: &ProfDAGStats) {
        todo!();
        /*
            sum_ += rhs.sum_;
        sqrsum_ += rhs.sqrsum_;
        cnt_ += rhs.cnt_;
        return *this;
        */
    }
}

///---------------------------

pub struct ProfDAGReport {
    
    op_types:                      Vec<String>,
    op_extra_info:                 Vec<Vec<String>>,
    net_name:                      String,
    num_runs:                      i32,

    /**
      | Cumulative stats per operator instance
      | of the net
      |
      */
    time_per_op_total:             Vec<ProfDAGStats>,

    /// Cumulative stats per unique operator type
    time_per_op_type_total:        HashMap<String,ProfDAGStats>,

    times_per_run_per_type_total:  HashMap<String,ProfDAGStats>,
    runtime_stats:                 ProfDAGStats,
}

/**
  | A simple wrapper around prof_dag's
  | counters
  |
  */
pub struct ProfDAGCounters {
    timer:                   Timer,
    op_start_times_run:      Vec<f32>,
    op_end_times_run:        Vec<f32>,
    op_async_end_times_run:  Vec<f32>,
    report:                  ProfDAGReport,
}

impl ProfDAGCounters {
    
    pub fn new(net_def: &Arc<NetDef>) -> Self {
    
        todo!();
        /*
            report_.net_name_ = net_def->name();
      report_.num_runs_ = 0;
      auto num_ops = net_def->op_size();
      report_.op_types_.reserve(num_ops);
      report_.op_extra_info_.reserve(num_ops);

      for (auto op_id = 0; op_id < num_ops; ++op_id) {
        const auto& op = net_def->op(op_id);
        if (op.engine() == "") {
          report_.op_types_.push_back(op.type());
        } else {
          report_.op_types_.push_back(op.type() + "(" + op.engine() + ")");
        }
        vector<std::string> op_extra_info;
        if (op.has_device_option() && op.device_option().extra_info_size() > 0) {
          for (auto i = 0; i < op.device_option().extra_info_size(); ++i) {
            std::string extra_info_str = op.device_option().extra_info(i);
            op_extra_info.push_back(extra_info_str);
          }
        }
        report_.op_extra_info_.push_back(op_extra_info);
      }
      report_.time_per_op_total_.resize(num_ops);
        */
    }
    
    /**
      | ReportRunStart/End are called at the
      | beginning and at the end of each net's
      | run
      |
      */
    #[inline] pub fn report_run_start(&mut self)  {
        
        todo!();
        /*
            report_.num_runs_ += 1;
      timer_.Start();
      auto num_ops = report_.op_types_.size();
      op_start_times_run_.clear();
      op_start_times_run_.resize(num_ops, -1.0);
      op_end_times_run_.clear();
      op_end_times_run_.resize(num_ops, -1.0);
      op_async_end_times_run_.clear();
      op_async_end_times_run_.resize(num_ops, -1.0);
        */
    }
    
    #[inline] pub fn add_per_op_start_time(&mut self, op_id: usize)  {
        
        todo!();
        /*
            if (report_.num_runs_ <= 1) {
        return;
      }

      CAFFE_ENFORCE(op_id >= 0 && op_id < op_start_times_run_.size());
      op_start_times_run_[op_id] = timer_.MilliSeconds();
        */
    }
    
    #[inline] pub fn add_per_op_end_time(&mut self, op_id: usize)  {
        
        todo!();
        /*
            if (report_.num_runs_ <= 1) {
        return;
      }

      CAFFE_ENFORCE(op_id >= 0 && op_id < op_end_times_run_.size());
      op_end_times_run_[op_id] = timer_.MilliSeconds();
        */
    }
    
    #[inline] pub fn add_per_op_async_end_time(&mut self, op_id: usize)  {
        
        todo!();
        /*
            if (report_.num_runs_ <= 1) {
        return;
      }

      CAFFE_ENFORCE(op_id >= 0 && op_id < op_async_end_times_run_.size());
      op_async_end_times_run_[op_id] = timer_.MilliSeconds();
        */
    }
    
    /**
      | ReportRunStart/End are called at the
      | beginning and at the end of each net's
      | run
      |
      */
    #[inline] pub fn report_run_end(&mut self)  {
        
        todo!();
        /*
            if (report_.num_runs_ <= 1) {
        return;
      }

      auto runtime = timer_.MilliSeconds();

      CaffeMap<std::string, float> cum_per_type_time_run;
      CaffeMap<std::string, float> cum_per_type_invocations_run;
      std::vector<float> per_op_time_run(report_.op_types_.size(), 0.0);
      for (const auto op_id : c10::irange(report_.op_types_.size())) {
        // check that we have valid times, otherwise return;
        // times might not be valid if network execution ended prematurely
        // because of operator errors
        if (op_start_times_run_[op_id] < 0.0) {
          return;
        }

        float op_time = 0.0;
        if (op_async_end_times_run_[op_id] > 0.0) {
          op_time = op_async_end_times_run_[op_id] - op_start_times_run_[op_id];
        } else {
          if (op_end_times_run_[op_id] < 0.0) {
            return;
          }
          op_time = op_end_times_run_[op_id] - op_start_times_run_[op_id];
        }

        per_op_time_run[op_id] = op_time;

        const string& op_type = report_.op_types_[op_id];
        cum_per_type_time_run[op_type] += op_time;
        cum_per_type_invocations_run[op_type] += 1;
      }

      // all operator times are valid, update report stats
      report_.runtime_stats_ += ProfDAGStats(runtime);

      for (const auto op_id : c10::irange(report_.op_types_.size())) {
        report_.time_per_op_total_[op_id] += ProfDAGStats(per_op_time_run[op_id]);
      }

      for (const auto& kv : cum_per_type_time_run) {
        report_.time_per_op_type_total_[kv.first] += ProfDAGStats(kv.second);
        report_.times_per_run_per_type_total_[kv.first] +=
            ProfDAGStats(cum_per_type_invocations_run[kv.first]);
      }
        */
    }
    
    #[inline] pub fn get_report(&self) -> ProfDAGReport {
        
        todo!();
        /*
            return report_;
        */
    }
}

impl ProfDAGReport {
    
    #[inline] pub fn has_stats(&self) -> bool {
        
        todo!();
        /*
            return runtime_stats_.cnt() > 0;
        */
    }
    
    #[inline] pub fn stats_proto(&self, 
        name:          &String,
        stats:         &ProfDAGStats,
        op_extra_info: &Vec<String>) -> ProfDAGProto {

        todo!();
        /*
            ProfDAGProto stats_proto;
      const auto& moments = stats.computeMoments();
      stats_proto.set_mean(moments.first);
      stats_proto.set_stddev(moments.second);
      stats_proto.set_name(name);
      for (auto& extra_info : op_extra_info) {
        stats_proto.add_extra_info(extra_info);
      }
      return stats_proto;
        */
    }
    
    /**
      | Collects the execution time per each
      | operator type
      |
      */
    #[inline] pub fn get_operator_stats(&self) -> ProfDAGProtos {
        
        todo!();
        /*
            ProfDAGProtos prof_dag_protos;
      prof_dag_protos.set_net_name(net_name_);
      if (hasStats()) {
        for (auto& item : time_per_op_type_total_) {
          auto buf = prof_dag_protos.add_stats();
          buf->CopyFrom(statsProto(item.first, item.second, vector<std::string>()));
        }
      }
      return prof_dag_protos;
        */
    }
    
    /**
      | Collects the execution time of each
      | operator, the output is formatted as
      | a map: (netName__opIndex__opType,
      | cost)
      |
      */
    #[inline] pub fn get_per_operator_cost(&self) -> ProfDAGProtos {
        
        todo!();
        /*
            ProfDAGProtos prof_dag_protos;
      prof_dag_protos.set_net_name(net_name_);
      if (hasStats()) {
        for (const auto op_id : c10::irange(op_types_.size())) {
          const string& op_type = op_types_[op_id];
          auto buf = prof_dag_protos.add_stats();
          std::string op_output_name =
              net_name_ + "___" + to_string(op_id) + "___" + op_type;
          buf->CopyFrom(statsProto(
              op_output_name, time_per_op_total_[op_id], op_extra_info_[op_id]));
        }
      }
      return prof_dag_protos;
        */
    }
    
    #[inline] pub fn print_stats(&mut self)  {
        
        todo!();
        /*
            if (!hasStats()) {
        LOG(INFO) << "Insufficient number of runs";
        return;
      }

      std::ostringstream debug_out;
      debug_out << "Measured operators over " << runtime_stats_.cnt()
                << " net runs (" << net_name_ << "), #ops: " << op_types_.size()
                << std::endl;

      debug_out << "Mean time in operator type per run (stddev):" << std::endl;
      for (const auto& item : time_per_op_type_total_) {
        const auto& moments = item.second.computeMoments();
        const auto& times_moments =
            times_per_run_per_type_total_[item.first].computeMoments();
        debug_out << std::setw(10) << std::setfill(' ') << moments.first
                  << " ms/run (" << std::setw(10) << std::setfill(' ')
                  << moments.second << " ms/run) "
                  << " Op count per run: " << times_moments.first << "  "
                  << item.first << std::endl;
      }
      const auto& runtime_moments = runtime_stats_.computeMoments();
      debug_out << net_name_ << " runtime: " << runtime_moments.first << " ms ("
                << runtime_moments.second << " ms)" << std::endl;

      LOG(INFO) << debug_out.str();
        */
    }
}


impl AddAssign<&ProfDAGReport> for ProfDAGReport {
    
    fn add_assign(&mut self, other: &ProfDAGReport) {
        todo!();
        /*
            // Verify nets are compatible for addition
      CAFFE_ENFORCE_EQ(
          net_name_, rhs.net_name_, "Incompatible nets to add counters");
      CAFFE_ENFORCE_EQ(
          op_types_.size(),
          rhs.op_types_.size(),
          "Incompatible nets to add counters");
      for (const auto idx : c10::irange(op_types_.size())) {
        CAFFE_ENFORCE_EQ(
            op_types_[idx],
            rhs.op_types_[idx],
            "Incompatible nets to add counters");
      }

      if (!rhs.hasStats()) {
        // rhs does not have valid profiling results, do nothing
        return *this;
      } else if (!hasStats()) {
        // "this" does not have valid profiling results, but rhs does. copy rhs
        time_per_op_total_ = rhs.time_per_op_total_;
        time_per_op_type_total_ = rhs.time_per_op_type_total_;
        times_per_run_per_type_total_ = rhs.times_per_run_per_type_total_;
        runtime_stats_ = rhs.runtime_stats_;
        num_runs_ = rhs.num_runs_;
        return *this;
      }

      // Do the addition
      for (const auto idx : c10::irange(time_per_op_total_.size())) {
        time_per_op_total_[idx] += rhs.time_per_op_total_.at(idx);
      }
      for (auto& item : time_per_op_type_total_) {
        item.second += rhs.time_per_op_type_total_.at(item.first);
      }
      for (auto& item : times_per_run_per_type_total_) {
        item.second += rhs.times_per_run_per_type_total_.at(item.first);
      }
      runtime_stats_ += rhs.runtime_stats_;
      num_runs_ += rhs.num_runs_;

      return *this;
        */
    }
}
