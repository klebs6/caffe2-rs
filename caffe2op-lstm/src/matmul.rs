crate::ix!();

#[inline] pub fn matmul(
    x: &Tensor,
    w: &Tensor,
    context: *mut CPUContext) -> Tensor 
{
    todo!();
    /*
        const auto canonical_axis = X.canonical_axis_index(1);
      const auto M = X.size_to_dim(canonical_axis);
      const auto K = X.size_from_dim(canonical_axis);
      const auto canonical_axis_w = W.canonical_axis_index(1);
      const int N = W.size_to_dim(canonical_axis_w);
      auto output_size = X.sizes().vec();
      output_size.resize(canonical_axis + 1);
      output_size[canonical_axis] = N;
      Tensor C(output_size, CPU);
      math::Gemm<float, CPUContext>(
          CblasNoTrans,
          CblasTrans,
          M,
          N,
          K,
          1,
          X.template data<float>(),
          W.template data<float>(),
          0,
          C.template mutable_data<float>(),
          context);
      return C;
    */
}
