crate::ix!();

#[inline] pub fn linear(
    x: &Tensor,
    w: &Tensor,
    b: &Tensor,
    context: *mut CPUContext) -> Tensor 
{
    todo!();
    /*
        auto output = matmul(X, W, context);
      if (B) {
        const auto canonical_axis = X.canonical_axis_index(1);
        const auto M = X.size_to_dim(canonical_axis);
        const auto canonical_axis_w = W.canonical_axis_index(1);
        const int N = W.size_to_dim(canonical_axis_w);
        auto bias_multiplier_ = caffe2::empty({M}, CPU);
        math::Set<float, CPUContext>(
            M, 1, bias_multiplier_.template mutable_data<float>(), context);
        math::Gemm<float, CPUContext>(
            CblasNoTrans,
            CblasNoTrans,
            M,
            N,
            1,
            1,
            bias_multiplier_.template data<float>(),
            B.template data<float>(),
            1,
            output.template mutable_data<float>(),
            context);
      }
      return output;
    */
}
