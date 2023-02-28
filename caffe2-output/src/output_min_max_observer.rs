crate::ix!();

use crate::{
    OperatorStorage,
    OperatorInfo,
    ObserverBase
};

pub struct OutputMinMaxObserver {
    base:             ObserverBase<OperatorStorage>,
    info:             Arc<OperatorInfo>,
    warning_printed:  bool, // default = false
}

impl OutputMinMaxObserver {

    /**
      | OutputMinMaxObserver is assumed to
      | be used together with OutputMinMaxNetObserver
      | and the information shared via shared_ptr
      | to be prepared for the case when
      | OutputMinMaxObserver is destroyed
      | before OutputMinMaxNetObserver
      |
      */
    #[inline] pub fn get_info(&mut self) -> Arc<OperatorInfo> {

        todo!();
        /*
            return info_;
        */
    }
}

impl OutputMinMaxObserver {
    
    #[inline] pub fn stop(&mut self)  {
        
        todo!();
        /*
            for (int i = 0; i < subject_->OutputSize(); ++i) {
        if (!subject_->OutputIsTensorType(i, CPU)) {
          continue;
        }
        Tensor* tensor = subject_->template Output<Tensor>(i, CPU);
        if (!tensor || tensor->numel() == 0 || tensor->numel() == -1)
          continue;
        string out_name(subject_->debug_def().output(i));

        float min = numeric_limits<float>::lowest(),
              max = float::max;

        if (tensor->IsType<float>()) {
          if (!tensor->data<float>()) {
            continue;
          }
          FindMinMax(tensor->data<float>(), &min, &max, tensor->numel());
        } else if (tensor->IsType<int>()) {
          if (!tensor->data<int>()) {
            continue;
          }
          FindMinMax(tensor->data<int>(), &min, &max, tensor->numel());
        } else if (tensor->IsType<long>()) {
          if (!tensor->data<long>()) {
            continue;
          }
          FindMinMax(tensor->data<long>(), &min, &max, tensor->numel());
        } else {
          if (!warning_printed_) {
            LOG(INFO) << "Tensor " << out_name << " has unsupported type "
                      << tensor->meta().name() << " with size " << tensor->numel();
            warning_printed_ = true;
          }
          continue;
        }

    #ifdef _OPENMP
    #pragma omp critical
    #endif
        {
          if (min_max_map_.find(out_name) == min_max_map_.end()) {
            min_max_map_[out_name] = make_pair(
                float::max, numeric_limits<float>::lowest());
          }

          info_->tensor_infos[i].Update(min, max);

          min_max_map_[out_name].first =
              std::min(min_max_map_[out_name].first, min);
          min_max_map_[out_name].second =
              std::max(min_max_map_[out_name].second, max);
          assert(min_max_map_[out_name].second >= min_max_map_[out_name].first);
          assert(min_max_map_[out_name].first < 1e38);

          VLOG(2) << this << " " << info_->type << " " << i << " " << out_name
                  << " " << info_->tensor_infos[i].min << " "
                  << info_->tensor_infos[i].max << " "
                  << min_max_map_[out_name].first << " "
                  << min_max_map_[out_name].second;
        }
      }

      return;
        */
    }
}

impl Drop for OutputMinMaxObserver {
    fn drop(&mut self) {
        todo!();
        /* 
      /*#pragma omp critical
        {
          for (int i = 0; i < info_->tensor_infos.size(); ++i) {
            LOG(INFO) <<
              this << " " << info_->type << " " << i << " " <<
              info_->tensor_infos[i].name << " " <<
              info_->tensor_infos[i].min << " " <<
              info_->tensor_infos[i].max << " " <<
              min_max_map_[info_->tensor_infos[i].name].first << " " <<
              min_max_map_[info_->tensor_infos[i].name].second << " ";
          }
        }*/
 */
    }
}
