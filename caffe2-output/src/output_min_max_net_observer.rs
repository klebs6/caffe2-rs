crate::ix!();

pub struct OutputMinMaxNetObserver {
    base:           NetObserver,
    dump_freq:      i32,
    cnt:            i32,
    out_file_name:  String,
    delimiter:      String,
    min_max_infos:  Vec<Arc<OperatorInfo>>,
}

impl Drop for OutputMinMaxNetObserver {

    fn drop(&mut self) {
        todo!();
        /* 
      DumpAndReset_(out_file_name_, true);

    #ifdef _OPENMP
    #pragma omp critical
    #endif
      {
        ofstream f;
        time_t rawtime;
        time(&rawtime);
        struct tm timeinfo;
        localtime_r(&rawtime, &timeinfo);
        char buffer[128] = {};
        strftime(buffer, sizeof(buffer), "%Y-%m-%d-%H-%M-%S", &timeinfo);
        char buffer2[256] = {};
        snprintf(buffer2, sizeof(buffer2), "global_%s.minmax", buffer);

        f.open(buffer2);
        int op_index = 0;
        for (auto key_value : min_max_map_) {
          ostringstream ost;
          assert(key_value.second.first <= key_value.second.second);
          assert(key_value.second.first < 1e38);
          ost << op_index << " 0 " << key_value.first << " "
              << key_value.second.first << " " << key_value.second.second;
          f << ost.str() << endl;

          ++op_index;
        }
        f.close();
      }
     */
    }
}

impl OutputMinMaxNetObserver {
    
    #[inline] pub fn stop(&mut self)  {
        
        todo!();
        /*
            ++cnt_;
      if (dump_freq_ == -1 || (cnt_ % dump_freq_) != 0) {
        return;
      }

      ostringstream ost;
      size_t last_dot = out_file_name_.rfind('.');
      size_t last_slash = out_file_name_.rfind('/');
      if (last_dot != string::npos &&
          (last_slash == string::npos || last_slash < last_dot)) {
        ost << out_file_name_.substr(0, last_dot) << "_" << cnt_ / dump_freq_
            << out_file_name_.substr(last_dot);
      } else {
        ost << out_file_name_ << "_" << cnt_ / dump_freq_;
      }

      DumpAndReset_(ost.str());
      return;
        */
    }
    
    /**
      | @params dump_freq Print out only once in
      | destructor if -1.
      |
      | Otherwise, print out every dum_freq
      | invocations
      */
    pub fn new(
        subject:       *mut NetBase,
        out_file_name: &String,
        dump_freq:     Option<i32>,
        delimiter:     Option<String>) -> Self {

        let delimiter = delimiter.unwrap_or(" ".to_string());

        let dump_freq: i32 = dump_freq.unwrap_or(-1);

        todo!();
        /*
            : NetObserver(subject),
          dump_freq_(dump_freq),
          cnt_(0),
          out_file_name_(out_file_name),
          delimiter_(delimiter) 

          VLOG(2) << out_file_name;
          min_max_infos_.resize(subject->GetOperators().size());
          int i = 0;
          for (auto* op : subject->GetOperators()) {
            OutputMinMaxObserver* observer = new OutputMinMaxObserver(op);
            op->AttachObserver(std::unique_ptr<OutputMinMaxObserver>(observer));
            min_max_infos_[i] = observer->GetInfo();
            ++i;
          }
        */
    }
    
    #[inline] pub fn dump_and_reset(
        &mut self, 
        out_file_name: &String, 
        print_total_min_max: Option<bool>)
    {
        let print_total_min_max: bool = print_total_min_max.unwrap_or(false);
        
        todo!();
        /*
            ofstream f(out_file_name);
      if (!f) {
        LOG(WARNING) << this << ": can't open " << out_file_name;
      }

      for (int op_index = 0; op_index < min_max_infos_.size(); ++op_index) {
        OutputMinMaxObserver::OperatorInfo* op_info =
            min_max_infos_[op_index].get();
        if (op_info) {
          for (int i = 0; i < op_info->tensor_infos.size(); ++i) {
            const OutputMinMaxObserver::TensorInfo& tensor_info =
                op_info->tensor_infos[i];

            ostringstream ost;
            ost << op_index << delimiter_ << op_info->type << delimiter_ << i
                << delimiter_ << tensor_info.name << delimiter_;
            if (print_total_min_max) {
              ost << tensor_info.total_min << delimiter_ << tensor_info.total_max;
            } else {
              ost << tensor_info.min << delimiter_ << tensor_info.max;
            }

            LOG(INFO) << this << delimiter_ << ost.str();
            f << ost.str() << endl;

            op_info->tensor_infos[i].min = float::max;
            op_info->tensor_infos[i].max = numeric_limits<float>::lowest();
          }
        }
      }
      f.close();
        */
    }
}
