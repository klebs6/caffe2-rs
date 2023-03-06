crate::ix!();

#[inline] pub fn mul(x: &Tensor, y: &Tensor, context: *mut CPUContext) -> Tensor {
    
    todo!();
    /*
        Tensor Z(X.sizes().vec(), CPU);
      math::Mul<float, CPUContext>(
          X.numel(),
          X.template data<float>(),
          Y.template data<float>(),
          Z.template mutable_data<float>(),
          context);
      return Z;
    */
}

#[inline] pub fn add(x: &Tensor, y: &Tensor, context: *mut CPUContext) -> Tensor {
    
    todo!();
    /*
        Tensor Z(X.sizes().vec(), CPU);
      math::Add<float, CPUContext>(
          X.numel(),
          X.template data<float>(),
          Y.template data<float>(),
          Z.template mutable_data<float>(),
          context);
      return Z;
    */
}
