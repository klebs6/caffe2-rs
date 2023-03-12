crate::ix!();

#[inline] pub fn compute_div_gradient<TGrad, TIn, TOut>(
    ndim:      i32,
    a_dims:    *const i32,
    b_dims:    *const i32,
    c_dims:    *const i32,
    dC:        *const TGrad,
    b:         *const TIn,
    c:         *const TOut,
    dA:        *mut TGrad,
    dB:        *mut TGrad,
    context:   *mut CPUContext) 
{
    todo!();
    /*
        const int A_size =
          std::accumulate(A_dims, A_dims + ndim, 1, std::multiplies<int>());
      const int B_size =
          std::accumulate(B_dims, B_dims + ndim, 1, std::multiplies<int>());
      const int C_size =
          std::accumulate(C_dims, C_dims + ndim, 1, std::multiplies<int>());
      if (dA != nullptr) {
        math::Set<TGrad, CPUContext>(A_size, TGrad(0), dA, context);
      }
      math::Set<TGrad, CPUContext>(B_size, TGrad(0), dB, context);
      std::vector<int> index(ndim, 0);
      for (int C_index = 0; C_index < C_size; ++C_index) {
        const int B_index =
            math::utils::GetIndexFromDims(ndim, B_dims, index.data());
        dB[B_index] += -dC[C_index] * C[C_index] / B[B_index];
        if (dA != nullptr) {
          const int A_index =
              math::utils::GetIndexFromDims(ndim, A_dims, index.data());
          dA[A_index] += dC[C_index] / B[B_index];
        }
        math::utils::IncreaseIndexInDims(ndim, C_dims, index.data());
      }
    */
}
