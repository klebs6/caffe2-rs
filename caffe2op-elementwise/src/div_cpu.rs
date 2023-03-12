crate::ix!();

impl DivFunctor<CPUContext> {

    #[inline] pub fn backward<TGrad, TIn, TOut>(
        &mut self, 
        a_dims:     &Vec<i32>,
        b_dims:     &Vec<i32>,
        dC:         *const TGrad,
        a:          *const TIn,
        b:          *const TIn,
        c:          *const TOut,
        dA:         *mut TGrad,
        dB:         *mut TGrad,
        context:    *mut CPUContext) -> bool 
    {
        todo!();
        /*
            if (A_dims == B_dims) {
            const int size = std::accumulate(
                A_dims.cbegin(), A_dims.cend(), 1, std::multiplies<int>());
            EigenVectorMap<TGrad>(dB, size) =
                -ConstEigenVectorArrayMap<TGrad>(dC, size) *
                ConstEigenVectorArrayMap<TOut>(C, size) /
                ConstEigenVectorArrayMap<TIn>(B, size);
            math::Div(size, dC, B, dA, context);
            return true;
          }
          const int ndim = std::max(A_dims.size(), B_dims.size());
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
          if (dA == dC) {
            ComputeDivGradient<TGrad, TIn, TOut>(
                ndim,
                A_broadcast_dims.data(),
                B_broadcast_dims.data(),
                C_broadcast_dims.data(),
                dC,
                B,
                C,
                nullptr,
                dB,
                context);
            math::Div(
                A_dims.size(),
                A_dims.data(),
                B_dims.size(),
                B_dims.data(),
                dC,
                B,
                dA,
                context);
          } else {
            ComputeDivGradient<TGrad, TIn, TOut>(
                ndim,
                A_broadcast_dims.data(),
                B_broadcast_dims.data(),
                C_broadcast_dims.data(),
                dC,
                B,
                C,
                dA,
                dB,
                context);
          }
          return true;
        */
    }
}
