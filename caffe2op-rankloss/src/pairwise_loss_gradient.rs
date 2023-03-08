crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct PairWiseLossGradientOp<T,Context> {
    storage: OperatorStorage,
    context: Context,
    phantom: PhantomData<T>,
}

impl<T,Context> PairWiseLossGradientOp<T,Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(XVALUE);
      auto& label = Input(LABEL);
      auto& dY = Input(DYVALUE);

      int N = X.dim() > 0 ? X.dim32(0) : 0;
      CAFFE_ENFORCE_EQ(N, X.numel());
      CAFFE_ENFORCE(
          (label.dim() == 1) || (label.dim() == 2 && label.dim32(1) == 1));
      CAFFE_ENFORCE_EQ(label.dim32(0), N);
      auto* dX = Output(DXVALUE, X.sizes(), at::dtype<T>());
      math::Set<T, CPUContext>(
          dX->numel(), 0.f, dX->template mutable_data<T>(), &context_);

      if (N == 0) {
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

      CAFFE_ENFORCE_EQ(dY.dim(), 1);
      CAFFE_ENFORCE_EQ(dY.dim32(0), len_size);

      const T* Xdata = X.template data<T>();
      const T* dYdata = dY.template data<T>();
      const T* labelData = label.template data<T>();
      T* dXdata = dX->template mutable_data<T>();
      int offset = 0;
      for (int idx = 0; idx < len_size; ++idx) {
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
            auto grad =
                sign * dYdata[idx] / (1 + exp(-sign * (Xdata[j] - Xdata[i])));
            dXdata[i] -= grad;
            dXdata[j] += grad;
          }
        }
        if (numPairs > 0) {
          for (int i = offset; i < offset + lengths_vec[idx]; ++i) {
            dXdata[i] /= numPairs;
          }
        }
        offset += lengths_vec[idx];
      }
      return true;
        */
    }
}
