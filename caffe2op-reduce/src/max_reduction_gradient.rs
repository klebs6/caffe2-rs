crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct MaxReductionGradientOp<T,Context,const ROWWISE: bool> {
    storage: OperatorStorage,
    context: Context,
    phantom: PhantomData<T>,
}

impl<T, Context, const ROWWISE: bool> 
MaxReductionGradientOp<T, Context, ROWWISE> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0);
      auto& Y = Input(1);
      auto& dY = Input(2);

      auto* dX = Output(0, X.sizes(), at::dtype<T>());

      CAFFE_ENFORCE_EQ(X.dim(), 3);

      const int batch_size = X.dim32(0);
      const int M = X.dim32(1);
      const int N = X.dim32(2);

      const T* Xdata = X.template data<T>();
      const T* Ydata = Y.template data<T>();
      const T* dYdata = dY.template data<T>();
      T* dXdata = dX->template mutable_data<T>();

      const int input_size = M * N;
      for (int i = 0; i < batch_size; ++i) {
        const T* Xdata_i = Xdata + i * input_size;
        T* dXdata_i = dXdata + i * input_size;
        if (ROWWISE) {
          const T* Ydata_i = Ydata + i * M;
          const T* dYdata_i = dYdata + i * M;
          for (int m = 0; m < M; ++m) {
            const T* Xdata_m = Xdata_i + m * N;
            T* dXdata_m = dXdata_i + m * N;
            for (int n = 0; n < N; ++n) {
              if (Xdata_m[n] == Ydata_i[m]) {
                dXdata_m[n] = dYdata_i[m];
              } else {
                dXdata_m[n] = static_cast<T>(0);
              }
            }
          }
        } else {
          const T* Ydata_i = Ydata + i * N;
          const T* dYdata_i = dYdata + i * N;
          for (int n = 0; n < N; ++n) {
            for (int m = 0; m < M; ++m) {
              const T* Xdata_m = Xdata_i + m * N;
              T* dXdata_m = dXdata_i + m * N;
              if (Xdata_m[n] == Ydata_i[n]) {
                dXdata_m[n] = dYdata_i[n];
              } else {
                dXdata_m[n] = static_cast<T>(0);
              }
            }
          }
        }
      }

      return true;
        */
    }
}
