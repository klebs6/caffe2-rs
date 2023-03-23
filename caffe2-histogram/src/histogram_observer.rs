crate::ix!();

pub struct HistogramObserverInfo {
    histograms:        Vec<DynamicHistogram>,
    total_histograms:  Vec<DynamicHistogram>,
    min_max_info:      OperatorInfo,
}

/**
  | Given min/max, collect histogram
  |
  */
pub struct HistogramObserver {
    base:             ObserverBase<OperatorStorage>,
    info:             Arc<HistogramObserverInfo>,
    warning_printed:  bool, // default = false
}

impl HistogramObserver {
    
    pub fn new(op: *mut OperatorStorage, info: Arc<HistogramObserverInfo>) -> Self {
    
        todo!();
        /*
            : ObserverBase<OperatorStorage>(op), info_(info)
        */
    }
    
    #[inline] pub fn stop(&mut self)  {
        
        todo!();
        /*
            for (int i = 0; i < subject_->OutputSize(); ++i) {
        if (!subject_->OutputIsTensorType(i, CPU)) {
          continue;
        }
        Tensor* tensor = subject_->template Output<Tensor>(i, CPU);
        if (!tensor || tensor->numel() == 0 || tensor->numel() == -1) {
          continue;
        }

        string out_name(subject_->debug_def().output(i));

        const float* data = nullptr;
        vector<float> data_temp;

        if (tensor->IsType<float>()) {
          if (!tensor->data<float>()) {
            continue;
          }
          data = tensor->template data<float>();
        } else if (tensor->IsType<int>()) {
          if (!tensor->data<int>()) {
            continue;
          }
          const int* data_orig = tensor->data<int>();
          data_temp.resize(tensor->numel());
          for (int j = 0; j < tensor->numel(); ++j) {
            data_temp[j] = data_orig[j];
          }
          data = data_temp.data();
        } else if (tensor->IsType<long>()) {
          if (!tensor->data<long>()) {
            continue;
          }
          const long* data_orig = tensor->data<long>();
          data_temp.resize(tensor->numel());
          for (int j = 0; j < tensor->numel(); ++j) {
            data_temp[j] = data_orig[j];
          }
          data = data_temp.data();
        } else {
          if (!warning_printed_) {
            LOG(INFO) << "Tensor " << out_name << " has unsupported type "
                      << tensor->meta().name() << " with size " << tensor->numel();
            warning_printed_ = true;
          }
          continue;
        }

        info_->histograms[i].Add(data, tensor->numel());
        info_->total_histograms[i].Add(data, tensor->numel());
      }
      return;
        */
    }
}
