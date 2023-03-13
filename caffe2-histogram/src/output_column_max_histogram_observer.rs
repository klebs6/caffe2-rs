crate::ix!();


/**
  | Given min/max, collect histogram of
  | the max value of each column of tensor
  |
  */
pub struct OutputColumnMaxHistogramObserver {
    base:               ObserverBase<OperatorStorage>,
    col_max_blob_name:  String,
    nbins:              i32,
    info:               Arc<HistogramObserverInfo>,
    warning_printed:    bool, // default = false
    col_max_blob_idx:   i32,  // default = -1
    num_columns:        i32,  // default = -1
}

impl OutputColumnMaxHistogramObserver {

    pub fn new(
        op:                *mut OperatorStorage,
        col_max_blob_name: &String,
        nbins:             i32,
        info:              Arc<HistogramObserverInfo>) -> Self {

        todo!();
        /*
            : ObserverBase<OperatorStorage>(op),
          col_max_blob_name_(col_max_blob_name),
          nbins_(nbins),
          info_(info) 

      const auto& output_names = op->debug_def().output();
      auto it =
          std::find(output_names.begin(), output_names.end(), col_max_blob_name);
      CAFFE_ENFORCE(
          it != output_names.end(), "Cannot find blob in operator output.");
      col_max_blob_idx_ = std::distance(output_names.begin(), it);
        */
    }
    
    #[inline] pub fn stop(&mut self)  {
        
        todo!();
        /*
            if (!subject_->OutputIsTensorType(col_max_blob_idx_, CPU)) {
        return;
      }
      Tensor* tensor = subject_->template Output<Tensor>(col_max_blob_idx_, CPU);
      if (!tensor || tensor->numel() == 0 || tensor->numel() == -1) {
        return;
      }

      float* data = GetFloatTensorData(tensor);
      if (data == nullptr && !warning_printed_) {
        LOG(INFO) << "Tensor " << col_max_blob_name_
                  << " has mismatching type, or unsupported type "
                  << tensor->meta().name() << " with size " << tensor->numel();
        warning_printed_ = true;
        return;
      }

      // determine number of columns
      CAFFE_ENFORCE(
          tensor->dim() == 2,
          "Tensor " + col_max_blob_name_ +
              " is not two-dimensional. Tensor.dim() = " +
              caffe2::to_string(tensor->dim()));
      int num_columns = tensor->size_from_dim(1);
      if (num_columns_ == -1) {
        num_columns_ = num_columns;
      }
      CAFFE_ENFORCE(
          num_columns_ == num_columns, "Observed inconsistent number of columns.");
      int num_rows = tensor->size_to_dim(1);
      for (int col = 0; col < num_columns; col++) {
        // find col max of the ith column
        auto col_max = std::abs(data[col]);
        for (int r = 0; r < num_rows; r++) {
          int idx = r * num_columns + col;
          col_max = max(col_max, std::abs(data[idx]));
        }
        if (info_->histograms.size() <= col) {
          info_->histograms.emplace_back(nbins_);
          info_->total_histograms.emplace_back(nbins_);
          info_->min_max_info.tensor_infos.emplace_back(col_max_blob_name_);
        }
        info_->histograms[col].Add(col_max);
        info_->total_histograms[col].Add(col_max);
      }
        */
    }
}


