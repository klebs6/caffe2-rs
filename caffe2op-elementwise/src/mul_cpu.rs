crate::ix!();

impl MulFunctor<CPUContext> {

    #[inline] pub fn backward<TGrad, TIn, TOut>(
        &self,
        a_dims:    &Vec<i32>,
        b_dims:    &Vec<i32>,
        dC:        *const TGrad,
        a:         *const TIn,
        b:         *const TIn,
        c:         *const TOut,
        dA:        *mut TGrad,
        dB:        *mut TGrad,
        context:   *mut CPUContext) -> bool 
    {
        todo!();
        /*
            if (A_dims == B_dims) {
            const auto size = c10::multiply_integers(A_dims);
            math::Mul(size, dC, B, dA, context);
            math::Mul(size, dC, A, dB, context);
            return true;
          }

          const int ndim = std::max(A_dims.size(), B_dims.size());
          if (ndim == 0) {
            return true;
          }

          std::vector<int> A_broadcast_dims(ndim);
          std::vector<int> B_broadcast_dims(ndim);
          std::vector<int> C_broadcast_dims(ndim);
          math::utils::ComputeBroadcastBinaryOpDims(
              A_dims.size(),
              A_dims.data(),
              B_dims.size(),
              B_dims.data(),
              A_broadcast_dims.data(),
              B_broadcast_dims.data(),
              C_broadcast_dims.data());

          const int C_size = std::accumulate(
              C_broadcast_dims.cbegin(),
              C_broadcast_dims.cbegin() + ndim,
              1,
              std::multiplies<int>());
          if (C_size == 0) {
            const auto A_size = c10::multiply_integers(A_dims);
            const auto B_size = c10::multiply_integers(B_dims);
            math::Set<TGrad, CPUContext>(A_size, TGrad(0), dA, context);
            math::Set<TGrad, CPUContext>(B_size, TGrad(0), dB, context);
            return true;
          }

          // Flatten dims as much as possible
          // We call A is broadcasted at dim d if A_broadcast_dims[d] <= 1
          // Two consecutive dims d and d+1 can be flattened if
          // A and B are broadcasted at dim d, or
          // A and B are broadcasted at dim d + 1, or
          // A is broadcasted at dim d and d + 1, or
          // B is broadcasted at dim d and d + 1, or
          // A and B are not broadcasted at dim d and d + 1
          std::vector<int> A_broadcast_dims_flattened, B_broadcast_dims_flattened,
              C_broadcast_dims_flattened;
          A_broadcast_dims_flattened.reserve(ndim);
          B_broadcast_dims_flattened.reserve(ndim);

          A_broadcast_dims_flattened.push_back(A_broadcast_dims[0]);
          B_broadcast_dims_flattened.push_back(B_broadcast_dims[0]);

          for (int i = 1; i < ndim; ++i) {
            int A_old = A_broadcast_dims_flattened.back();
            int B_old = B_broadcast_dims_flattened.back();
            int A_new = A_broadcast_dims[i];
            int B_new = B_broadcast_dims[i];
            if ((A_old == 1 && B_old == 1) || (A_new == 1 && B_new == 1) ||
                (A_old == 1 && A_new == 1) || (B_old == 1 && B_new == 1) ||
                (A_old > 1 && B_old > 1 && A_new > 1 && B_new > 1)) {
              A_broadcast_dims_flattened.back() *= A_new;
              B_broadcast_dims_flattened.back() *= B_new;
            } else {
              A_broadcast_dims_flattened.push_back(A_new);
              B_broadcast_dims_flattened.push_back(B_new);
            }
          }

          int ndim_flattened = A_broadcast_dims_flattened.size();
          C_broadcast_dims_flattened.resize(ndim_flattened);
          for (int i = 0; i < ndim_flattened; ++i) {
            C_broadcast_dims_flattened[i] =
                std::max(A_broadcast_dims_flattened[i], B_broadcast_dims_flattened[i]);
          }

          if (std::is_same<TGrad, float>::value && std::is_same<TIn, float>::value &&
              ndim_flattened <= 2 &&
              A_broadcast_dims_flattened[0] == B_broadcast_dims_flattened[0] &&
              (ndim_flattened == 1 || A_broadcast_dims_flattened[1] <= 1 ||
               B_broadcast_dims_flattened[1] <= 1)) {
            if (ndim_flattened == 2) {
              // fast path when we have 2 flattened dimensions and the second dimension
              // is broadcasted.
              bool broadcast_B = B_broadcast_dims_flattened[1] <= 1;
              ComputeMulGradient(
                  C_broadcast_dims_flattened[0],
                  C_broadcast_dims_flattened[1],
                  reinterpret_cast<const float*>(dC),
                  reinterpret_cast<const float*>(broadcast_B ? A : B),
                  reinterpret_cast<const float*>(broadcast_B ? B : A),
                  reinterpret_cast<float*>(broadcast_B ? dA : dB),
                  reinterpret_cast<float*>(broadcast_B ? dB : dA),
                  context);
            } else {
              // fast path when we have 1 flattened dimension
              assert(ndim_flattened == 1);
              ComputeMulGradient(
                  C_broadcast_dims_flattened[0],
                  reinterpret_cast<const float*>(dC),
                  reinterpret_cast<const float*>(A),
                  reinterpret_cast<const float*>(B),
                  reinterpret_cast<float*>(dA),
                  reinterpret_cast<float*>(dB));
            }
          } else {
            ComputeMulGradient<TGrad, TIn>(
                ndim_flattened,
                A_broadcast_dims_flattened.data(),
                B_broadcast_dims_flattened.data(),
                C_broadcast_dims_flattened.data(),
                dC,
                A,
                B,
                dA,
                dB,
                context);
          }

          return true;
        */
    }
}
