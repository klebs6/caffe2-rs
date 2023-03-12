crate::ix!();

#[inline] pub fn compute_mul_gradient_with_dims<TGrad, TIn>(
    ndim:      i32,
    a_dims:    *const i32,
    b_dims:    *const i32,
    c_dims:    *const i32,
    dC:        *const TGrad,
    a:         *const TIn,
    b:         *const TIn,
    dA:        *mut TGrad,
    dB:        *mut TGrad,
    context:   *mut CPUContext) 
{
    todo!();
    /*
       const auto A_size = c10::multiply_integers(A_dims, A_dims + ndim);
      const auto B_size = c10::multiply_integers(B_dims, B_dims + ndim);
      const auto C_size = c10::multiply_integers(C_dims, C_dims + ndim);
      math::Set<TGrad, CPUContext>(A_size, TGrad(0), dA, context);
      math::Set<TGrad, CPUContext>(B_size, TGrad(0), dB, context);
      std::vector<int> index(ndim, 0);
      for (int C_index = 0; C_index < C_size; ++C_index) {
        const int A_index =
            math::utils::GetIndexFromDims(ndim, A_dims, index.data());
        const int B_index =
            math::utils::GetIndexFromDims(ndim, B_dims, index.data());
        dA[A_index] += dC[C_index] * B[B_index];
        dB[B_index] += dC[C_index] * A[A_index];
        math::utils::IncreaseIndexInDims(ndim, C_dims, index.data());
      }
    */
}

/**
  | A : input not to broadcast whose size is
  | common_size x broadcast_size
  |
  | B : input to broadcast whose size is
  | common_size
  */
#[inline] pub fn compute_mul_gradient_with_broadcast_size(
    common_size:      i32,
    broadcast_size:   i32,
    dC:               *const f32,
    a:                *const f32,
    b:                *const f32,
    dA:               *mut f32,
    dB:               *mut f32,
    context:          *mut CPUContext)  
{

    todo!();
    /*
        for (int i = 0; i < common_size; ++i) {
        caffe2::math::Scale(
            broadcast_size,
            B[i],
            dC + i * broadcast_size,
            dA + i * broadcast_size,
            context);
        caffe2::math::Dot(
            broadcast_size,
            dC + i * broadcast_size,
            A + i * broadcast_size,
            dB + i,
            context);
      }
    */
}

#[inline] pub fn compute_mul_gradient(
    size:  i32,
    dC:    *const f32,
    a:     *const f32,
    b:     *const f32,
    dA:    *mut f32,
    dB:    *mut f32)  
{
    
    todo!();
    /*
        for (int i = 0; i < size; ++i) {
        dA[i] = dC[i] * B[i];
        dB[i] = dC[i] * A[i];
      }
    */
}
