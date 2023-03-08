crate::ix!(); 

impl<T, Context> PiecewiseLinearTransformOp<T, Context> {
    
    #[inline] pub fn transform_binary(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(PREDICTIONS);

        CAFFE_ENFORCE(X.dim() == 1 || X.dim() == 2);
        int64_t N = X.dim32(0);
        int64_t M = X.dim() == 2 ? X.dim32(1) : 1;
        CAFFE_ENFORCE(
            M == 1 || M == 2,
            "If binary is set to true, the input must be Nx2 or Nx1 tensor");
        auto* Y = Output(0, X.sizes(), at::dtype<T>());
        const auto* Xdata = X.template data<T>();
        T* Ydata = Y->template mutable_data<T>();

        const T* bounds;
        const T* slopes;
        const T* intercepts;
        int64_t num_func_per_group;
        int64_t num_group;
        GetTransParamData(
            &bounds, &slopes, &intercepts, &num_func_per_group, &num_group);
        CAFFE_ENFORCE_EQ(num_group, 1);

        if (M == 1) {
          for (int64_t i = 0; i < N; ++i) {
            Ydata[i] = PiecewiseLinearTransform(
                Xdata[i], bounds, slopes, intercepts, num_func_per_group);
          }
        } else {
          for (int64_t i = 0; i < N; ++i) {
            Ydata[i * M + 1] = PiecewiseLinearTransform(
                Xdata[i * M + 1], bounds, slopes, intercepts, num_func_per_group);
            Ydata[i * M] = 1.0f - Ydata[i * M + 1];
          }
        }

        return true;
        */
    }
}
