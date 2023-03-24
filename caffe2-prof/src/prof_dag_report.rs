crate::ix!();

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
