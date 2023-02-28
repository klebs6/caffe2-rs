crate::ix!();

#[inline] pub fn sigmoid<T>(x: T) -> T {

    todo!();
    /*
        return 1 / (1 + std::exp(-x));
    */
}

#[inline] pub fn host_tanh<T>(x: T) -> T {

    todo!();
    /*
        return 2 * sigmoid(2 * x) - 1;
    */
}

#[inline] pub fn lstm_unit_impl<T>(
    n:           i32,
    d:           i32,
    t:           i32,
    h_prev:      *const T,
    c_prev:      *const T,
    x:           *const T,
    seq_lengths: *const i32,
    drop_states: bool,
    c:           *mut T,
    h:           *mut T,
    forget_bias: f32)  {

    todo!();
    /*
        const T forgetBias = convert::To<float, T>(forget_bias);
      for (int n = 0; n < N; ++n) {
        const bool valid = seqLengths == nullptr || t < seqLengths[n];
        if (!valid) {
          if (drop_states) {
            memset(H, 0, sizeof(T) * D);
            memset(C, 0, sizeof(T) * D);
          } else {
            memcpy(H, H_prev, sizeof(T) * D);
            memcpy(C, C_prev, sizeof(T) * D);
          }
        } else {
          const T* X_D = &X[D];
          const T* X_2D = &X[2 * D];
          const T* X_3D = &X[3 * D];
          VECTOR_LOOP for (int d = 0; d < D; ++d) {
            const T i = sigmoid(X[d]);
            const T f = sigmoid(X_D[d] + forgetBias);
            const T o = sigmoid(X_2D[d]);
            const T g = host_tanh(X_3D[d]);
            const T c_prev = C_prev[d];
            const T c = f * c_prev + i * g;
            C[d] = c;
            const T host_tanh_c = host_tanh(c);
            H[d] = o * host_tanh_c;
          }
        }
        H_prev += D;
        C_prev += D;
        X += 4 * D;
        C += D;
        H += D;
      }
    */
}

#[inline] pub fn lstm_unit_gradient_impl<T>(
    n:           i32,
    d:           i32,
    t:           i32,
    c_prev:      *const T,
    x:           *const T,
    seq_lengths: *const i32,
    c:           *const T,
    h:           *const T,
    c_diff:      *const T,
    h_diff:      *const T,
    drop_states: bool,
    h_prev_diff: *mut T,
    c_prev_diff: *mut T,
    x_diff:      *mut T,
    forget_bias: f32)  {

    todo!();
    /*
        const T localForgetBias = convert::To<float, T>(forget_bias);
      for (int n = 0; n < N; ++n) {
        const bool valid = seqLengths == nullptr || t < seqLengths[n];

        if (!valid) {
          if (drop_states) {
            memset(C_prev_diff, 0, sizeof(T) * D);
            memset(H_prev_diff, 0, sizeof(T) * D);
          } else {
            memcpy(H_prev_diff, H_diff, sizeof(T) * D);
            memcpy(C_prev_diff, C_diff, sizeof(T) * D);
          }
          memset(X_diff, 0, 4 * sizeof(T) * D);
        } else {
          VECTOR_LOOP for (int d = 0; d < D; ++d) {
            T* c_prev_diff = C_prev_diff + d;
            T* h_prev_diff = H_prev_diff + d;
            T* i_diff = X_diff + d;
            T* f_diff = X_diff + 1 * D + d;
            T* o_diff = X_diff + 2 * D + d;
            T* g_diff = X_diff + 3 * D + d;

            const T i = sigmoid(X[d]);
            const T f = sigmoid(X[1 * D + d] + localForgetBias);
            const T o = sigmoid(X[2 * D + d]);
            const T g = host_tanh(X[3 * D + d]);
            const T c_prev = C_prev[d];
            const T c = C[d];
            const T host_tanh_c = host_tanh(c);
            const T c_term_diff =
                C_diff[d] + H_diff[d] * o * (1 - host_tanh_c * host_tanh_c);
            *c_prev_diff = c_term_diff * f;
            *h_prev_diff = 0; // not used in 'valid' case
            *i_diff = c_term_diff * g * i * (1 - i);
            *f_diff = c_term_diff * c_prev * f * (1 - f);
            *o_diff = H_diff[d] * host_tanh_c * o * (1 - o);
            *g_diff = c_term_diff * i * (1 - g * g);
          }
        }
        C_prev += D;
        X += 4 * D;
        C += D;
        H += D;
        C_diff += D;
        H_diff += D;
        X_diff += 4 * D;
        H_prev_diff += D;
        C_prev_diff += D;
      }
    */
}
