crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cpu/moments_utils.h]

pub const K_CHUNK_SIZE: i64 = 16;

pub fn add_moments<T>(
        m0_add: i64,
        m1_add: &T,
        m2_add: &T,
        m0:     &mut i64,
        m1:     &mut T,
        m2:     &mut T)  {

    todo!();
        /*
            const i64 n = m0 + m0_add;
      const T c = n == 0 ? 0 : static_cast<T>(m0_add) / static_cast<T>(n);
      const T delta = m1_add - m1;
      m1 += c * delta;
      m2 += m2_add + delta * delta * c * static_cast<T>(m0);
      m0 = n;
        */
}

pub fn add_moments_vec<T>(
        m0_add: i64,
        m1_add: &Vectorized<T>,
        m2_add: &Vectorized<T>,
        m0:     &mut i64,
        m1:     &mut Vectorized<T>,
        m2:     &mut Vectorized<T>)  {

    todo!();
        /*
            using Vec = vec::Vectorized<T>;
      const i64 n = m0 + m0_add;
      const T c = n == 0 ? 0 : static_cast<T>(m0_add) / static_cast<T>(n);
      const Vec c_vec(c);
      const Vec delta = m1_add - m1;
      m1 += c_vec * delta;
      m2 += m2_add + delta * delta * c_vec * Vec(static_cast<T>(m0));
      m0 = n;
        */
}

/**
  | Compute rowwise moments by Welford algorithm
  | and cascade sum to improve numerical stability.
  |
  | https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
  | https://en.wikipedia.org/wiki/Pairwise_summation
  */
pub fn rowwise_moments_impl<T, const kMaxDepth: i64>(
        X: *const T,
        N: i64) -> (T,T) {

    todo!();
        /*
            using Vec = vec::Vectorized<T>;

      constexpr i64 kVecSize = Vec::size();
      const i64 n = N / kVecSize;
      const i64 m = divup(n, kChunkSize);
      const i64 depth = CeilLog2(m);

      const Vec kZeroVec(T(0));
      SmallVector<i64, kMaxDepth> m0_stk(depth, 0);
      SmallVector<Vec, kMaxDepth> m1_stk(depth, kZeroVec);
      SmallVector<Vec, kMaxDepth> m2_stk(depth, kZeroVec);

      for (i64 i = 0; i < m; ++i) {
        const T* X_ptr = X + i * kChunkSize * kVecSize;
        const i64 m0 = min(kChunkSize, n - i * kChunkSize);
        Vec m1_vec(0);
        Vec m2_vec(0);
        for (i64 j = 0; j < m0; ++j) {
          const Vec x_vec = Vec::loadu(X_ptr + j * kVecSize);
          const Vec delta_vec = x_vec - m1_vec;
          const Vec c_vec = Vec(T(1) / static_cast<T>(j + 1));
          m1_vec += delta_vec * c_vec;
          m2_vec += delta_vec * (x_vec - m1_vec);
        }
        AddMomentsVec(m0, m1_vec, m2_vec, m0_stk[0], m1_stk[0], m2_stk[0]);
        i64 mask = i + 1;
        for (i64 j = 1; j < depth && (mask & 1) == 0; ++j) {
          AddMomentsVec(
              m0_stk[j - 1],
              m1_stk[j - 1],
              m2_stk[j - 1],
              m0_stk[j],
              m1_stk[j],
              m2_stk[j]);
          m0_stk[j - 1] = 0;
          m1_stk[j - 1] = kZeroVec;
          m2_stk[j - 1] = kZeroVec;
          mask >>= 1;
        }
      }
      for (i64 i = 1; i < depth; ++i) {
        AddMomentsVec(
            m0_stk[i], m1_stk[i], m2_stk[i], m0_stk[0], m1_stk[0], m2_stk[0]);
      }

      array<T, kVecSize> m1_arr{};
      array<T, kVecSize> m2_arr{};
      m1_stk[0].store(m1_arr.data());
      m2_stk[0].store(m2_arr.data());

      i64 m0 = 0;
      T m1 = 0;
      T m2 = 0;
      for (i64 i = n * kVecSize; i < N; ++i) {
        const T delta = X[i] - m1;
        ++m0;
        m1 += delta / static_cast<T>(m0);
        m2 += delta * (X[i] - m1);
      }
      for (i64 i = 0; i < kVecSize; ++i) {
        AddMoments(n, m1_arr[i], m2_arr[i], m0, m1, m2);
      }

      return make_pair(m1, m2 / static_cast<T>(N));
        */
}

pub fn rowwise_moments<T>(
        X: *const T,
        N: i64) -> (T,T) {

    todo!();
        /*
            using Vec = vec::Vectorized<T>;
      constexpr i64 kVecSize = Vec::size();
      const i64 n = N / kVecSize;
      const i64 m = divup(n, kChunkSize);
      const i64 depth = CeilLog2(m);
      if (depth <= 4) {
        return RowwiseMomentsImpl<T, 4>(X, N);
      } else if (depth <= 8) {
        return RowwiseMomentsImpl<T, 8>(X, N);
      } else if (depth <= 16) {
        return RowwiseMomentsImpl<T, 16>(X, N);
      } else if (depth <= 32) {
        return RowwiseMomentsImpl<T, 32>(X, N);
      } else {
        return RowwiseMomentsImpl<T, 64>(X, N);
      }
        */
}
