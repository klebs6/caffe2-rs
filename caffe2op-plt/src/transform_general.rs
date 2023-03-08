crate::ix!(); 

impl<T, Context> PiecewiseLinearTransformOp<T, Context> {

    #[inline] pub fn transform_general(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0);

        CAFFE_ENFORCE_EQ(X.dim(), 2);
        int64_t N = X.dim32(0);
        int64_t M = X.dim32(1);
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
        CAFFE_ENFORCE_EQ(num_group, M);

        for (int64_t j = 0; j < M; ++j) {
          const T* bounds_group = bounds + j * (num_func_per_group + 1);
          const T* slopes_group = slopes + j * num_func_per_group;
          const T* intercepts_group = intercepts + j * num_func_per_group;
          for (int64_t i = 0; i < N; ++i) {
            Ydata[i * M + j] = PiecewiseLinearTransform(
                Xdata[i * M + j],
                bounds_group,
                slopes_group,
                intercepts_group,
                num_func_per_group);
          }
        }
        return true;
        */
    }
}
