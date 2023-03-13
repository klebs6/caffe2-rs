crate::ix!();

impl<CPUContext,W: Write> NanCheckOp<CPUContext,W> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0);
      auto* Y = Output(0);
      const int D = X.numel();
      const float* data = X.data<float>();
      ConstEigenVectorMap<float> input_data(data, D);

      bool all_finite = input_data.allFinite();

      if (!all_finite) {
        std::cerr << "Tensor contained NaN or inf: [" << this->debug_def().input(0)
                  << "]" << std::endl;

        for (int j = 0; j < InputSize(); j++) {
          std::cerr << "Tensor name: " << this->debug_def().input(j) << std::endl;
          std::cerr << "Input tensor:" << std::endl;
          tensorPrinter_.Print<float>(Input(j));
          std::cerr << "NaN idxs:" << std::endl;
          const float* x = Input(j).data<float>();
          for (size_t i = 0; i < Input(j).numel(); ++i) {
            if (std::isnan(x[i]) || std::isinf(x[i])) {
              std::cerr << i << " ";
            }
          }
          std::cerr << std::endl;
        }
        return false;
      }

      if (&X != Y) {
        Y->CopyFrom(X);
      }
      return true;
        */
    }
}
