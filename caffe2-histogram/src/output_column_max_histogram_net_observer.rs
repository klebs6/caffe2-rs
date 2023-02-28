crate::ix!();

use crate::{
    HistogramObserverInfo,
    NetObserver,
    NetBase
};


///-------------------------------
pub struct OutputColumnMaxHistogramNetObserver {
    base: NetObserver,

    dump_freq:           i32,
    cnt:                 i32,
    mul_nets:            bool,
    out_file_name:       String,
    delimiter:           String,
    col_max_blob_names:  HashSet<String>,

    /// {op_idx: {output_index: col_hists}}
    hist_infos:  HashMap<i32,HashMap<i32,Arc<HistogramObserverInfo>>>,
}

impl OutputColumnMaxHistogramNetObserver {
    
    #[inline] pub fn dump_output_column_max_histogram_file(&mut self)  {
        
        todo!();
        /*
            DumpAndReset_(out_file_name_, false);
        */
    }
    
    pub fn new(
        subject:                      *mut NetBase,
        out_file_name:                &String,
        observe_column_max_for_blobs: &Vec<String>,
        nbins:                        i32,
        dump_freq:                    Option<i32>,
        mul_nets:                     Option<bool>,
        delimiter:                    Option<String>) -> Self {

        let delimiter      = delimiter.unwrap_or(" ".to_string());
        let dump_freq: i32 = dump_freq.unwrap_or(-1);
        let mul_nets: bool = mul_nets.unwrap_or(false);

        todo!();
        /*
            : NetObserver(subject),
          dump_freq_(dump_freq),
          cnt_(0),
          mul_nets_(mul_nets),
          out_file_name_(out_file_name),
          delimiter_(delimiter) 

      if (observe_column_max_for_blobs.size() == 0) {
        return;
      }
      col_max_blob_names_.insert(
          observe_column_max_for_blobs.begin(), observe_column_max_for_blobs.end());
      int op_idx = 0;
      for (auto* op : subject->GetOperators()) {
        const auto& op_output_names = op->debug_def().output();
        int output_idx = 0;
        std::unordered_map<int, std::shared_ptr<HistogramObserver::Info>>
            output_col_hists_map;
        for (const auto& output_blob : op_output_names) {
          if (col_max_blob_names_.find(output_blob) == col_max_blob_names_.end()) {
            ++output_idx;
            continue;
          }
          /// create col max hist observer for blob
          auto info = std::make_shared<HistogramObserver::Info>();
          info->min_max_info.type = op->debug_def().type();
          // number of histograms in info will be determined at runtime by the
          // number of columns in the tensor.
          OutputColumnMaxHistogramObserver* observer =
              new OutputColumnMaxHistogramObserver(op, output_blob, nbins, info);
          op->AttachObserver(
              unique_ptr<OutputColumnMaxHistogramObserver>(observer));
          output_col_hists_map[output_idx] = info;
          ++output_idx;
        }
        if (output_col_hists_map.size() > 0) {
          hist_infos_[op_idx] = output_col_hists_map;
        }
        ++op_idx;
      }
        */
    }
    
    #[inline] pub fn dump_and_reset(
        &mut self, 
        out_file_name:       &String, 
        print_total_min_max: Option<bool>)  
    {
        let print_total_min_max: bool = print_total_min_max.unwrap_or(false);
        
        todo!();
        /*
            stringstream file_name;
      file_name << out_file_name;
      if (mul_nets_) {
        file_name << ".";
        file_name << this;
      }
      ofstream f(file_name.str());
      if (!f) {
        LOG(WARNING) << this << ": can't open " << file_name.str();
      }
      for (const auto& it : hist_infos_) {
        auto output_idx_hists_map = it.second;
        for (const auto& output_idx_hist : output_idx_hists_map) {
          int output_idx = output_idx_hist.first;
          HistogramObserver::Info* info = output_idx_hist.second.get();
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
                           << info->min_max_info.tensor_infos[i].name
                           << " is empty";
            }
            ostringstream ost;
            // op_idx, output_idx, blob_name, col, min, max, nbins
            ost << it.first << delimiter_ << output_idx << delimiter_
                << info->min_max_info.tensor_infos[i].name << delimiter_ << i
                << delimiter_ << hist->Min() << delimiter_ << hist->Max()
                << delimiter_ << hist->GetHistogram()->size();

            // bins
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
      }
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

impl Drop for OutputColumnMaxHistogramNetObserver {
    fn drop(&mut self) {
        todo!();
        /* 
      DumpAndReset_(out_file_name_, true);
 */
    }
}

