crate::ix!();

#[inline] pub fn transpose(
    x:       &Tensor, 
    dim0:    i32, 
    dim1:    i32, 
    context: *mut CPUContext) -> Tensor 
{
    todo!();
    /*
        int ndim = X.dim();
      CAFFE_ENFORCE(ndim > dim0 && ndim > dim1, "Invalid transpose dimensions");
      std::vector<int> axes(ndim);
      std::iota(axes.begin(), axes.end(), 0);
      std::swap(axes[dim0], axes[dim1]);
      const std::vector<std::int64_t> X_dims = X.sizes().vec();
      std::vector<std::int64_t> Y_dims(ndim);
      for (int i = 0; i < ndim; ++i) {
        Y_dims[i] = X_dims[axes[i]];
      }
      Tensor Y(Y_dims, CPU);
      math::Transpose<std::int64_t, float, CPUContext>(
          ndim,
          X_dims.data(),
          axes.data(),
          X.template data<float>(),
          Y.template mutable_data<float>(),
          context);
      return Y;
    */
}
