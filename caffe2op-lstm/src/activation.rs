crate::ix!();

#[inline] pub fn sigmoid(x: &Tensor) -> Tensor {
    
    todo!();
    /*
        Tensor Y(X.sizes(), CPU);
      auto N = X.numel();
      EigenVectorArrayMap<float>(Y.template mutable_data<float>(), N) = 1.0 /
          (1.0 +
           (-ConstEigenVectorArrayMap<float>(X.template data<float>(), N)).exp());
      return Y;
    */
}

#[inline] pub fn tanh(x: &Tensor, context: *mut CPUContext) -> Tensor {
    
    todo!();
    /*
        Tensor Y(X.sizes(), CPU);
      math::Tanh<float, CPUContext>(
          X.numel(),
          X.template data<float>(),
          Y.template mutable_data<float>(),
          context);
      return Y;
    */
}
