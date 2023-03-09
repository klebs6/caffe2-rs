crate::ix!();

impl<Context> TransposeOp<Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            axes_(this->template GetRepeatedArgument<int>("axes")) 

        // We will check the legality of axes_: it should be from 0 to axes_.size().
        std::vector<int> axes_sorted = axes_;
        std::sort(axes_sorted.begin(), axes_sorted.end());
        for (std::size_t i = 0; i < axes_sorted.size(); ++i) {
          if (axes_sorted[i] != i) {
            CAFFE_THROW("Axes should be a permutation of 0 to ndim.");
          }
        }
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            // Do the actual transpose, which is implemented in DoRunWithType().
        return DispatchHelper<TensorTypes<float, double, int, int64_t>>::call(
            this, Input(0));
        */
    }
    
    #[inline] pub fn transpose_impl<T>(
        &mut self, 
        x: &Tensor,
        y: *mut Tensor)
    {
        todo!();
        /*
            const int ndim = X.dim();
        if (axes_.empty()) {
          axes_.resize(ndim);
          std::iota(axes_.rbegin(), axes_.rend(), 0);
        } else {
          CAFFE_ENFORCE_EQ(ndim, axes_.size());
        }
        const std::vector<std::int64_t> X_dims = X.sizes().vec();
        std::vector<std::int64_t> Y_dims(ndim);
        for (int i = 0; i < ndim; ++i) {
          Y_dims[i] = X_dims[axes_[i]];
        }
        Y->Resize(Y_dims);
        math::Transpose<std::int64_t, T, Context>(
            X_dims.size(),
            X_dims.data(),
            axes_.data(),
            X.template data<T>(),
            Y->template mutable_data<T>(),
            &context_);
        */
    }
    
    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            TransposeImpl<T>(Input(0), Output(0));
        return true;
        */
    }
}
