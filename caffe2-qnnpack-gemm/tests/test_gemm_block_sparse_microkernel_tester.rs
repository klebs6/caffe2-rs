// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/test/gemm-block-sparse-microkernel-tester.h]

pub fn fill_block_sparse_weights(
    b:              *mut u8,
    N:              usize,
    K:              usize,
    row_block_size: usize,
    col_block_size: usize,
    sparsity:       f32,
    zero_points:    *const u8)  {

    todo!();
        /*
            random_device randomDevice;
        auto rng = mt19937(randomDevice());
        bernoulli_distribution dist{sparsity};
        for (u32 n = 0; n < N ; n += row_block_size) {
          for (u32 k = 0; k < K; k += col_block_size) {
            if (dist(rng)) {
              for (u32 nb = 0; (nb < row_block_size) && (n + nb < N); ++nb) {
                for (u32 kb = 0; (kb < col_block_size) && (k + kb < K); ++kb) {
                  *(b + (n + nb) * K + k + kb) = zero_points[n + nb];
                }
              }
            }
          }
        }
        */
}

/**
  | Temp Debug utils that will be removed
  | later
  |
  */
pub fn print_matrix_u8(
    name: *const u8,
    a:    *const u8,
    M:    usize,
    N:    usize)  {
    
    todo!();
        /*
            cout << "Matrix START:" << name << "...\n";
        for (u32 m = 0; m < M ; ++m) {
          for (u32 n = 0; n < N; n++) {
            cout << (const u32)(*(a + m * N + n)) << ", ";
          }
          cout << endl;
        }
        cout << "Matrix END...\n\n";
        */
}

pub fn print_matrix_f32(
    name: *const u8,
    a:    *const f32,
    M:    usize,
    N:    usize)  {
    
    todo!();
        /*
            cout << "Matrix START:" << name << "...\n";
        for (u32 m = 0; m < M ; ++m) {
          for (u32 n = 0; n < N; n++) {
            cout << (*(a + m * N + n)) << ", ";
          }
          cout << endl;
        }
        cout << "Matrix END...\n\n";
        */
}

pub struct GemmBlockSparseMicrokernelTester {
    mr:             usize, // default = { 1 }
    nr:             usize, // default = { 1 }
    m:              usize, // default = { 1 }
    n:              usize, // default = { 1 }
    k:              usize, // default = { 1 }
    ks:             usize, // default = { 1 }
    a_stride:       usize, // default = { 0 }
    c_stride:       usize, // default = { 0 }
    row_block_size: usize, // default = { 1 }
    col_block_size: usize, // default = { 4 }
    a_zero_point:   u8, // default = { 0 }
    b_zero_point:   u8, // default = { 0 }
    qmin:           u8, // default = { 0 }
    qmax:           u8, // default = { 255 }
    iterations:     usize, // default = { 10 }
    multiplier:     f32, // default = { 2.0f }
    sparsity:       f32, // default = { 0.7f }
}

impl GemmBlockSparseMicrokernelTester {
    
    #[inline] pub fn mr(&mut self, mr: usize) -> &mut GemmBlockSparseMicrokernelTester {
        
        todo!();
        /*
            this->mr_ = mr;
        return *this;
        */
    }
    
    #[inline] pub fn mr(&self) -> usize {
        
        todo!();
        /*
            return this->mr_;
        */
    }
    
    #[inline] pub fn nr(&mut self, nr: usize) -> &mut GemmBlockSparseMicrokernelTester {
        
        todo!();
        /*
            this->nr_ = nr;
        return *this;
        */
    }
    
    #[inline] pub fn nr(&self) -> usize {
        
        todo!();
        /*
            return this->nr_;
        */
    }
    
    #[inline] pub fn m(&mut self, m: usize) -> &mut GemmBlockSparseMicrokernelTester {
        
        todo!();
        /*
            this->m_ = m;
        return *this;
        */
    }
    
    #[inline] pub fn m(&self) -> usize {
        
        todo!();
        /*
            return this->m_;
        */
    }
    
    #[inline] pub fn n(&mut self, n: usize) -> &mut GemmBlockSparseMicrokernelTester {
        
        todo!();
        /*
            this->n_ = n;
        return *this;
        */
    }
    
    #[inline] pub fn n(&self) -> usize {
        
        todo!();
        /*
            return this->n_;
        */
    }
    
    #[inline] pub fn k(&mut self, k: usize) -> &mut GemmBlockSparseMicrokernelTester {
        
        todo!();
        /*
            this->k_ = k;
        return *this;
        */
    }
    
    #[inline] pub fn k(&self) -> usize {
        
        todo!();
        /*
            return this->k_;
        */
    }
    
    #[inline] pub fn ks(&mut self, ks: usize) -> &mut GemmBlockSparseMicrokernelTester {
        
        todo!();
        /*
            this->ks_ = ks;
        return *this;
        */
    }
    
    #[inline] pub fn row_block_size(&mut self, block_size: usize) -> &mut GemmBlockSparseMicrokernelTester {
        
        todo!();
        /*
            this->rowBlockSize_ = block_size;
        return *this;
        */
    }
    
    #[inline] pub fn col_block_size(&mut self, block_size: usize) -> &mut GemmBlockSparseMicrokernelTester {
        
        todo!();
        /*
            this->colBlockSize_ = block_size;
        return *this;
        */
    }
    
    #[inline] pub fn sparsity(&mut self, s: f32) -> &mut GemmBlockSparseMicrokernelTester {
        
        todo!();
        /*
            this->sparsity_ = s;
        return *this;
        */
    }
    
    #[inline] pub fn ks(&self) -> usize {
        
        todo!();
        /*
            return this->ks_;
        */
    }
    
    #[inline] pub fn row_block_size(&self) -> usize {
        
        todo!();
        /*
            return this->rowBlockSize_;
        */
    }
    
    #[inline] pub fn col_block_size(&self) -> usize {
        
        todo!();
        /*
            return this->colBlockSize_;
        */
    }
    
    #[inline] pub fn sparsity(&self) -> f32 {
        
        todo!();
        /*
            return this->sparsity_;
        */
    }
    
    #[inline] pub fn biasn(&self) -> usize {
        
        todo!();
        /*
            return n() % nr() == 0 ? n() : (n() / nr() + 1) * nr();
        */
    }
    
    #[inline] pub fn a_stride(&mut self, a_stride: usize) -> &mut GemmBlockSparseMicrokernelTester {
        
        todo!();
        /*
            this->aStride_ = aStride;
        return *this;
        */
    }
    
    #[inline] pub fn a_stride(&self) -> usize {
        
        todo!();
        /*
            return this->aStride_ == 0 ? k() : this->aStride_;
        */
    }
    
    #[inline] pub fn c_stride(&mut self, c_stride: usize) -> &mut GemmBlockSparseMicrokernelTester {
        
        todo!();
        /*
            this->cStride_ = cStride;
        return *this;
        */
    }
    
    #[inline] pub fn c_stride(&self) -> usize {
        
        todo!();
        /*
            return this->cStride_ == 0 ? n() : this->cStride_;
        */
    }
    
    #[inline] pub fn a_zero_point(&mut self, a_zero_point: u8) -> &mut GemmBlockSparseMicrokernelTester {
        
        todo!();
        /*
            this->aZeroPoint_ = aZeroPoint;
        return *this;
        */
    }
    
    #[inline] pub fn a_zero_point(&self) -> u8 {
        
        todo!();
        /*
            return this->aZeroPoint_;
        */
    }
    
    #[inline] pub fn b_zero_point(&mut self, b_zero_point: u8) -> &mut GemmBlockSparseMicrokernelTester {
        
        todo!();
        /*
            this->bZeroPoint_ = bZeroPoint;
        return *this;
        */
    }
    
    #[inline] pub fn b_zero_point(&self) -> u8 {
        
        todo!();
        /*
            return this->bZeroPoint_;
        */
    }
    
    #[inline] pub fn multiplier(&mut self, multiplier: f32) -> &mut GemmBlockSparseMicrokernelTester {
        
        todo!();
        /*
            this->multiplier_ = multiplier;
        return *this;
        */
    }
    
    #[inline] pub fn multiplier(&self) -> f32 {
        
        todo!();
        /*
            return this->multiplier_;
        */
    }
    
    #[inline] pub fn qmin(&mut self, qmin: u8) -> &mut GemmBlockSparseMicrokernelTester {
        
        todo!();
        /*
            this->qmin_ = qmin;
        return *this;
        */
    }
    
    #[inline] pub fn qmin(&self) -> u8 {
        
        todo!();
        /*
            return this->qmin_;
        */
    }
    
    #[inline] pub fn qmax(&mut self, qmax: u8) -> &mut GemmBlockSparseMicrokernelTester {
        
        todo!();
        /*
            this->qmax_ = qmax;
        return *this;
        */
    }
    
    #[inline] pub fn qmax(&self) -> u8 {
        
        todo!();
        /*
            return this->qmax_;
        */
    }
    
    #[inline] pub fn iterations(&mut self, iterations: usize) -> &mut GemmBlockSparseMicrokernelTester {
        
        todo!();
        /*
            this->iterations_ = iterations;
        return *this;
        */
    }
    
    #[inline] pub fn iterations(&self) -> usize {
        
        todo!();
        /*
            return this->iterations_;
        */
    }
    
    pub fn test(&self, qgemm: PyTorchQ8GemmDqSparseUKernelFunction)  {
        
        todo!();
        /*
            ASSERT_LE(m(), mr());
        ASSERT_LE(n(), nr());

        random_device randomDevice;
        auto rng = mt19937(randomDevice());
        auto s32rng =
            bind(uniform_int_distribution<i32>(-10000, 10000), rng);
        auto u8rng = bind(uniform_int_distribution<u8>(), rng);

        vector<u8> a((m() - 1) * aStride() + k() + 8);
        vector<u8> b(n() * k());
        vector<float, AlignedAllocator<float, 32>> bias(max<usize>(8, n()));
        vector<float> c((m() - 1) * cStride() + n());
        vector<float> acc(m() * n());

        const u8* aPtr = a.data();

        for (usize iteration = 0; iteration < iterations(); iteration++) {
          generate(a.begin(), a.end(), ref(u8rng));
          generate(bias.begin(), bias.end(), ref(s32rng));
          fill(c.begin(), c.end(), 0.0f);
          usize num_zero_points_padded = n() + 8;
          vector<u8> kernel_zero_points
            (num_zero_points_padded, bZeroPoint());
          generate(kernel_zero_points.begin(), kernel_zero_points.end(), ref(u8rng));

          // This loop to ensure the assert_ne on b mat does not fire.
          u8 max_elem, min_elem;
          do {
            generate(b.begin(), b.end(), ref(u8rng));
            fillBlockSparseWeights(
                b.data(),
                n(),
                k(),
                rowBlockSize(),
                colBlockSize(),
                sparsity(),
                kernel_zero_points.data());
            max_elem = *max_element(b.cbegin(), b.cend());
            min_elem = *min_element(b.cbegin(), b.cend());
          } while (max_elem == min_elem);

          unique_ptr<qnnpack_BCSRMatrix> bcsr_matrix =
              qnnpack_generateBlockCSRMatrix(
                  b.data(),
                  n(),
                  k(),
                  rowBlockSize(),
                  colBlockSize(),
                  kernel_zero_points.data());

          ASSERT_NE(
              *max_element(a.cbegin(), a.cend()),
              *min_element(a.cbegin(), a.cend()));
          ASSERT_NE(
              *max_element(b.cbegin(), b.cend()),
              *min_element(b.cbegin(), b.cend()));

          auto f32rng =
              bind(uniform_real_distribution<float>(1, 5), rng);
          vector<float> dequantization_scales(num_zero_points_padded);
          generate(
              dequantization_scales.begin(),
              dequantization_scales.end(),
              ref(f32rng));
          /* Compute 32-bit results and output quantization arguments */
          fill(acc.begin(), acc.end(), 0);
          for (usize mIndex = 0; mIndex < m(); mIndex++) {
            for (usize nIndex = 0; nIndex < n(); nIndex++) {
              for (usize kIndex = 0; kIndex < k(); kIndex++) {
                ASSERT_LT(mIndex * n() + nIndex, acc.size());
                ASSERT_LT(mIndex * k() + kIndex, a.size());
                acc[mIndex * n() + nIndex] +=
                    (i32(aPtr[mIndex * aStride() + kIndex]) -
                     i32(aZeroPoint())) *
                    (i32(b[nIndex * k() + kIndex]) - i32(kernel_zero_points[nIndex]));
              }
              acc[mIndex * n() + nIndex] =
                acc[mIndex * n() + nIndex] *
                dequantization_scales[nIndex] +
                bias[nIndex];
            }
          }

          const struct pytorch_qnnp_conv_dynamic_quantization_params quantizationParams{
            aZeroPoint(),
            kernel_zero_points.data(),
            dequantization_scales.data(),
          };

          qgemm(
              m(),
              n(),
              aPtr,
              aStride() * sizeof(u8),
              bcsr_matrix->values.data(),
              bcsr_matrix->row_values.data(),
              bcsr_matrix->col_indices.data(),
              bias.data(),
              c.data(),
              cStride(),
              0,
              &quantizationParams);

          for (usize mIndex = 0; mIndex < m(); mIndex++) {
            for (usize nIndex = 0; nIndex < n(); nIndex++) {
              ASSERT_EQ(
                  c[mIndex * cStride() + nIndex],
                  acc[mIndex * n() + nIndex])
                  << "at " << mIndex << ", " << nIndex
                  << ": reference = " << acc[mIndex * n() + nIndex]
                  << ", optimized = " << c[mIndex * cStride() + nIndex]
                  << ", Mr x Nr = " << mr() << " x " << nr()
                  << ", M x N x K = " << m() << " x " << n() << " x " << k();
            }
          }
        }
        */
    }
    
    pub fn test_packed(&self, 
        packa: PyTorchQ8GemmSparsePackAUKernelFunction,
        qgemm: PyTorchQ8GemmDqSparsePAckedAUKernelFunction)  {
        
        todo!();
        /*
            ASSERT_LE(m(), mr());
        ASSERT_LE(n(), nr());

        random_device randomDevice;
        auto rng = mt19937(randomDevice());
        auto s32rng =
            bind(uniform_int_distribution<i32>(-10000, 10000), rng);
        auto u8rng = bind(uniform_int_distribution<u8>(), rng);

        vector<u8> a((m() - 1) * aStride() + k() + 8);
        vector<u8> b(n() * k());
        vector<float, AlignedAllocator<float, 32>> bias(max<usize>(8, n()));
        vector<float> c((m() - 1) * cStride() + n());
        vector<float> acc(m() * n());
        auto m_blocks = (m() + mr()  - 1) / mr();
        // While colBlockSize() is what kr is, we reuse 8x4/4x4 packing kernels
        // and thus a_packed has to be allocated accordingly.
        const u32 kr_value = 4;
        auto k_blocks = (k() + kr_value  - 1) / kr_value;
        vector<u8> a_packed((m_blocks * k_blocks * mr() * kr_value) + 8, 0);

        const u8* aPtr = a.data();

        for (usize iteration = 0; iteration < iterations(); iteration++) {
          generate(a.begin(), a.end(), ref(u8rng));
          generate(bias.begin(), bias.end(), ref(s32rng));
          fill(c.begin(), c.end(), 0.0f);
          usize num_zero_points_padded = n() + 8;
          vector<u8> kernel_zero_points
            (num_zero_points_padded, bZeroPoint());

          u8 max_elem, min_elem;
          // This loop to ensure the assert_ne on b mat does not fire.
          do {
            generate(b.begin(), b.end(), ref(u8rng));
            fillBlockSparseWeights(
                b.data(),
                n(),
                k(),
                rowBlockSize(),
                colBlockSize(),
                sparsity(),
                kernel_zero_points.data());
            max_elem = *max_element(b.cbegin(), b.cend());
            min_elem = *min_element(b.cbegin(), b.cend());
          } while (max_elem == min_elem);
          unique_ptr<qnnpack_BCSRMatrix> bcsr_matrix =
            qnnpack_generateBlockCSRMatrix(
                b.data(),
                n(),
                k(),
                rowBlockSize(),
                colBlockSize(),
                kernel_zero_points.data());

          ASSERT_NE(
              *max_element(a.cbegin(), a.cend()),
              *min_element(a.cbegin(), a.cend()));
          ASSERT_NE(
              *max_element(b.cbegin(), b.cend()),
              *min_element(b.cbegin(), b.cend()));

          auto f32rng =
              bind(uniform_real_distribution<float>(1, 5), rng);
          vector<float> dequantization_scales(num_zero_points_padded, 1.f);
          generate(
              dequantization_scales.begin(),
              dequantization_scales.end(),
              ref(f32rng));
          /* Compute 32-bit results and output quantization arguments */
          fill(acc.begin(), acc.end(), 0);
          for (usize mIndex = 0; mIndex < m(); mIndex++) {
            for (usize nIndex = 0; nIndex < n(); nIndex++) {
              for (usize kIndex = 0; kIndex < k(); kIndex++) {
                ASSERT_LT(mIndex * n() + nIndex, acc.size());
                ASSERT_LT(mIndex * k() + kIndex, a.size());
                acc[mIndex * n() + nIndex] +=
                    (i32(aPtr[mIndex * aStride() + kIndex]) -
                     i32(aZeroPoint())) *
                    (i32(b[nIndex * k() + kIndex]) - i32(kernel_zero_points[nIndex]));
              }
              acc[mIndex * n() + nIndex] =
                acc[mIndex * n() + nIndex] *
                dequantization_scales[nIndex] +
                bias[nIndex];
            }
          }

          const struct pytorch_qnnp_conv_dynamic_quantization_params quantizationParams{
            aZeroPoint(),
            kernel_zero_points.data(),
            dequantization_scales.data(),
          };

          packa(
              m(),
              k(),
              aPtr,
              aStride() * sizeof(u8),
              a_packed.data()
              );

          qgemm(
              m(),
              n(),
              a_packed.data(),
              bcsr_matrix->values.data(),
              bcsr_matrix->row_values.data(),
              bcsr_matrix->col_indices.data(),
              bias.data(),
              c.data(),
              cStride(),
              0,
              &quantizationParams);

          for (usize mIndex = 0; mIndex < m(); mIndex++) {
            for (usize nIndex = 0; nIndex < n(); nIndex++) {
              ASSERT_EQ(
                  c[mIndex * cStride() + nIndex],
                  acc[mIndex * n() + nIndex])
                  << "at " << mIndex << ", " << nIndex
                  << ": reference = " << acc[mIndex * n() + nIndex]
                  << ", optimized = " << c[mIndex * cStride() + nIndex]
                  << ", Mr x Nr = " << mr() << " x " << nr()
                  << ", M x N x K = " << m() << " x " << n() << " x " << k();
            }
          }
        }
        */
    }
}
