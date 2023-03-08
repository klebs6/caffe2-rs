crate::ix!();

/**
  | Operator computes the pair wise loss
  | between all pairs within a batch using
  | the logit loss function on the difference
  | in scores between pairs
  | 
  | support multiple batches of sessions
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct PairWiseLossOp<T,Context> {
    storage: OperatorStorage,
    context: Context,
    phantom: PhantomData<T>,
}

impl<T,Context> PairWiseLossOp<T,Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(XVALUE);
      auto& label = Input(LABEL);

      int N = X.dim() > 0 ? X.dim32(0) : 0;
      if (N == 0) {
        // Set correct data type for output
        Output(YVALUE, {0}, at::dtype<T>());
        return true;
      }

      const int32_t* lengths_vec;
      int len_size = 1;
      if (InputSize() > LENGTHS) {
        auto& lengths = Input(LENGTHS);
        CAFFE_ENFORCE_EQ(lengths.dim(), 1);
        len_size = lengths.numel();
        lengths_vec = lengths.template data<int32_t>();
        int len_sum = 0;
        if (len_size > 0) {
          math::Sum<int, Context>(len_size, lengths_vec, &len_sum, &context_);
        }
        CAFFE_ENFORCE_EQ(len_sum, N);
      } else {
        lengths_vec = &N;
      }

      // a total of len_size sessions
      auto* Y = Output(YVALUE, {len_size}, at::dtype<T>());
      auto* Ydata = Y->template mutable_data<T>();

      int D = X.numel() / N;
      CAFFE_ENFORCE(
          (label.dim() == 1) || (label.dim() == 2 && label.dim32(1) == 1));
      CAFFE_ENFORCE_EQ(label.dim32(0), N);
      CAFFE_ENFORCE_EQ(1, D); // only support one class at the moment

      const auto* Xdata = X.template data<T>();
      const auto* labelData = label.template data<T>();
      int offset = 0;
      for (int idx = 0; idx < len_size; ++idx) {
        Ydata[idx] = 0;
        int numPairs = 0;
        for (int i = offset; i < offset + lengths_vec[idx]; ++i) {
          for (int j = offset; j < i; ++j) {
            if (std::abs(labelData[i] - labelData[j]) <
                std::numeric_limits<T>::epsilon()) {
              continue;
            }
            ++numPairs;
            // only use sigmoid loss function at the moment
            auto sign = labelData[i] > labelData[j] ? 1 : -1;
            Ydata[idx] += logLogit(sign * (Xdata[j] - Xdata[i]));
          }
        }
        if (numPairs > 0) {
          Ydata[idx] /= numPairs;
        }
        offset += lengths_vec[idx];
      }
      return true;
        */
    }
}
