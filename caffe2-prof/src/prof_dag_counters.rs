crate::ix!();

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

