crate::ix!();

/**
  | Computes a histogram for values in the
  | given list of tensors.
  | 
  | For logging activation histograms
  | for post-hoc analyses, consider using
  | the HistogramObserver observer.
  | 
  | For iteratively computing a histogram
  | for all input tensors encountered through
  | history, consider using the AccumulateHistogram
  | operator.
  |
  */
pub struct HistogramOp<Context> {
    storage: OperatorStorage,
    context: Context,
    bin_edges: Vec<f32>,
}

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

pub enum HistogramOpOutputs {
    Histogram,
}

register_cpu_operator!{Histogram, HistogramOp<CPUContext>}

num_inputs!{Histogram, (1,INT_MAX)}

num_outputs!{Histogram, 1}

inputs!{Histogram, 
    0 => ("X1, X2, ...", "*(type: Tensor`<float>`)* List of input tensors.")
}

outputs!{Histogram, 
    0 => ("histogram", 
        "1D tensor of length k, wherein the i-th element expresses the 
        count of tensor values that fall within range [bin_edges[i], bin_edges[i + 1])")
}

args!{Histogram, 
    0 => ("bin_edges", 
        "length-(k + 1) sequence of float values wherein the i-th element 
        represents the inclusive left boundary of the i-th bin for i in [0, k - 1] and 
        the exclusive right boundary of the (i-1)-th bin for i in [1, k].")
}

should_not_do_gradient!{Histogram}
