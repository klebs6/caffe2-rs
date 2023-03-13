crate::ix!();

impl<T,Context> AccumulateHistogramOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            lower_bound_(
                this->template GetSingleArgument<float>("lower_bound", 0.0)),
            upper_bound_(
                this->template GetSingleArgument<float>("upper_bound", 1.0)),
            num_buckets_(this->template GetSingleArgument<int>("num_buckets", 1)) 

        CAFFE_ENFORCE_GT(num_buckets_, 0);
        // 2 more for histograms < lower_bound, >= upper_bound respectively
        num_output_buckets_ = num_buckets_ + 2;
        accumulate_hist_ = std::vector<int64_t>(num_output_buckets_, 0);
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(X_IN);
        auto* X_data = X.template data<T>();
        int N = X.numel();
        auto* cur_hist = Output(CUR_HIST);
        auto* acc_hist = Output(ACC_HIST);
        cur_hist->Resize(num_output_buckets_);
        acc_hist->Resize(num_output_buckets_);
        auto* cur_hist_data = cur_hist->template mutable_data<int64_t>();
        auto* acc_hist_data = acc_hist->template mutable_data<int64_t>();
        auto segment = (upper_bound_ - lower_bound_) / num_buckets_;
        math::Set<int64_t, Context>(
            num_output_buckets_, 0, cur_hist_data, &context_);

        for (int i = 0; i < N; i++) {
          int bucket_index = -1;
          if (X_data[i] < lower_bound_) {
            bucket_index = 0;
          } else if (X_data[i] >= upper_bound_) {
            bucket_index = num_buckets_ + 1;
          } else {
            bucket_index = (int)((X_data[i] - lower_bound_) / segment) + 1;
          }
          cur_hist_data[bucket_index] += 1;
          accumulate_hist_[bucket_index] += 1;
        }

        for (int i = 0; i < num_output_buckets_; i++) {
          acc_hist_data[i] = accumulate_hist_[i];
        }

        return true;
        */
    }
}
