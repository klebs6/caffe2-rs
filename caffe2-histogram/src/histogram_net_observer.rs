crate::ix!();



///-----------------------
pub struct HistogramNetObserver {
    base: NetObserver,

    dump_freq:      i32,
    cnt:            i32,

    /** 
     | If multiple nets exist and are attached
     | with the observers, the histogram files for
     | the nets will be appended with netbase
     | addresses.
     */
    mul_nets:       bool,
    net_name:       String,
    op_filter:      String,
    delimiter:      String,
    out_file_name:  String,
    hist_infos:     Vec<Arc<HistogramObserverInfo>>,
}

impl HistogramNetObserver {
    
    #[inline] pub fn dump_histogram_file(&mut self)  {
        
        todo!();
        /*
            DumpAndReset_(out_file_name_, false);
        */
    }
    
    /**
      | @param mul_nets
      | 
      | true if we expect multiple nets with
      | the same name so we include extra information
      | in the file name to distinghuish them
      | ----------
      | @param dump_freq
      | 
      | if not -1 we dump histogram every dump_freq
      | invocation of the net
      |
      */
    pub fn new(
        subject:       *mut NetBase,
        out_file_name: &String,
        nbins:         i32,
        dump_freq:     Option<i32>,
        mul_nets:      Option<bool>,
        op_filter:     Option<String>,
        delimiter:     Option<String>) -> Self {

        let op_filter = op_filter.unwrap_or("".to_string());
        let delimiter = delimiter.unwrap_or(" ".to_string());

        let dump_freq: i32 = dump_freq.unwrap_or(-1);
        let mul_nets: bool = mul_nets.unwrap_or(false);
    
        todo!();
        /*
            : NetObserver(subject),
          dump_freq_(dump_freq),
          cnt_(0),
          mul_nets_(mul_nets),
          op_filter_(op_filter),
          delimiter_(delimiter),
          out_file_name_(out_file_name) 

      net_name_ = subject->Name();
      if (op_filter != "") {
        bool has_op = false;
        for (auto* op : subject->GetOperators()) {
          if (op->debug_def().type() == op_filter) {
            has_op = true;
            break;
          }
        }
        if (!has_op) {
          LOG(INFO) << "Net " << net_name_ << " doesn't include operator "
                    << op_filter;
          return;
        }
      }

      hist_infos_.resize(subject->GetOperators().size());

      int i = 0;
      for (auto* op : subject->GetOperators()) {
        shared_ptr<HistogramObserver::Info> info(new HistogramObserver::Info);
        info->min_max_info.type = op->debug_def().type();

        for (int j = 0; j < op->OutputSize(); ++j) {
          info->histograms.emplace_back(nbins);
          info->total_histograms.emplace_back(nbins);
          info->min_max_info.tensor_infos.emplace_back(op->debug_def().output(j));
        }

        HistogramObserver* observer = new HistogramObserver(op, info);
        op->AttachObserver(unique_ptr<HistogramObserver>(observer));
        hist_infos_[i] = info;
        ++i;
      }
        */
    }
    
    #[inline] pub fn dump_and_reset(
        &mut self, 
        out_file_name: &String, 
        print_total_min_max: Option<bool>) {

        let print_total_min_max: bool = print_total_min_max.unwrap_or(false);
        
        todo!();
        /*
            if (hist_infos_.size() == 0) {
        return;
      }
      stringstream file_name;
      file_name << out_file_name;
      LOG(INFO) << "Dumping histograms of net " << net_name_ << " in " << this;
      if (mul_nets_) {
        file_name << ".";
        file_name << this;
      }
      ofstream f(file_name.str());
      if (!f) {
        LOG(WARNING) << this << ": can't open " << file_name.str();
      }

      for (int op_index = 0; op_index < hist_infos_.size(); ++op_index) {
        HistogramObserver::Info* info = hist_infos_[op_index].get();
        if (!info) {
          continue;
        }

        for (int i = 0; i < info->histograms.size(); ++i) {
          const Histogram* hist =
              (print_total_min_max ? info->total_histograms : info->histograms)[i]
                  .Finalize();
          if (hist->Min() >= hist->Max()) {
            LOG(WARNING) << "Histogram of "
                         << info->min_max_info.tensor_infos[i].name
                         << " has an empty range: min " << hist->Min()
                         << " and max " << hist->Max();
          }
          if (hist->GetHistogram()->empty()) {
            LOG(WARNING) << "Histogram of "
                         << info->min_max_info.tensor_infos[i].name << " is empty";
          }

          ostringstream ost;
          ost << op_index << delimiter_ << info->min_max_info.type << delimiter_
              << i << delimiter_ << info->min_max_info.tensor_infos[i].name
              << delimiter_ << hist->Min() << delimiter_ << hist->Max()
              << delimiter_ << hist->GetHistogram()->size();

          for (uint64_t c : *hist->GetHistogram()) {
            ost << delimiter_ << c;
          }

          if (print_total_min_max) {
            LOG(INFO) << this << delimiter_ << ost.str();
          }

          f << ost.str() << endl;

          if (!print_total_min_max) {
            info->histograms[i] = DynamicHistogram(hist->GetHistogram()->size());
          }
        }
      }
      f.flush();
      f.close();
        */
    }
    
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
}

impl Drop for HistogramNetObserver {
    fn drop(&mut self) {
        todo!();
        /* 
      DumpAndReset_(out_file_name_, false);
 */
    }
}
