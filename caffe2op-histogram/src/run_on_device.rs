crate::ix!();

impl<Context> HistogramOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            bin_edges_(this->template GetRepeatedArgument<float>("bin_edges")) 

        CAFFE_ENFORCE_GE(
            bin_edges_.size(),
            2,
            "Number of bin edges must be greater than or equal to 2.");
        for (int i = 1; i < bin_edges_.size(); i++) {
          CAFFE_ENFORCE_GT(
              bin_edges_[i],
              bin_edges_[i - 1],
              "bin_edges must be a strictly increasing sequence of values.");
        }
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<float, double>>::call(this, Input(0));
        */
    }
    
    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        
        todo!();
        /*
            CheckInputs();

        const auto* histogram = Output(HISTOGRAM);
        histogram->Resize(bin_edges_.size() - 1);
        auto* histogram_data = histogram->template mutable_data<int64_t>();
        math::Set<int64_t, Context>(
            bin_edges_.size() - 1, 0, histogram_data, &context_);

        for (int input_idx = 0; input_idx < InputSize(); input_idx++) {
          const auto& x = Input(input_idx);
          const int64_t N = x.numel();
          const auto* x_data = x.template data<T>();
          for (int64_t data_idx = 0; data_idx < N; data_idx++) {
            const auto bisection_it = std::upper_bound(
                bin_edges_.begin(), bin_edges_.end(), x_data[data_idx]);
            const int bisection_idx = bisection_it - bin_edges_.begin();
            if (bisection_idx > 0 && bisection_idx < bin_edges_.size()) {
              histogram_data[bisection_idx - 1]++;
            }
          }
        }

        return true;
        */
    }
    
    #[inline] pub fn check_inputs(&mut self)  {
        
        todo!();
        /*
            const auto& input_zero = Input(0);
        for (int i = 1; i < InputSize(); i++) {
          CAFFE_ENFORCE_EQ(
              Input(i).dtype(),
              input_zero.dtype(),
              "All inputs must have the same type; expected ",
              input_zero.dtype().name(),
              " but got ",
              Input(i).dtype().name(),
              " for input ",
              i);
        }
        */
    }
}
