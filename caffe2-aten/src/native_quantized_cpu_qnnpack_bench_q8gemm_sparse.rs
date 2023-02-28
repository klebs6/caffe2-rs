crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/bench/q8gemm_sparse.cc]

#[inline] pub fn divide_round_up(x: u32, q: u32) -> u32 {
    
    todo!();
        /*
            return x / q + u32(x % q != 0);
        */
}

#[inline] pub fn round_up(x: u32, q: u32) -> u32 {
    
    todo!();
        /*
            return q * divideRoundUp(x, q);
        */
}

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

pub struct Q8GEMM {
    base:                BenchmarkFixture,
    a:                   Vec<u8>,
    k:                   Vec<u8>,
    b:                   Vec<i32>,
    w:                   Vec<u8,AlignedAllocator<u8,32>>,
    c:                   Vec<u8>,
    mr:                  u32, // default = { 0 }
    nr:                  u32, // default = { 0 }
    np:                  u32, // default = { 0 }
    kr:                  u32, // default = { 0 }
    mc:                  u32, // default = { mr_ }
    nc:                  u32, // default = { nr_ }
    kc:                  u32, // default = { kr_ }
    quantization_params: PyTorchQnnpConvQuantizationParams,
}

impl Q8GEMM {
    
    pub fn new(
        mr: u32,
        nr: u32,
        np: u32,
        kr: u32) -> Self {
    
        todo!();
        /*
        : mr(mr),
        : nr(nr),
        : np(np),
        : kr(kr),
        : mc(mr),
        : nc(nr),
        : kc(kr),

        
        */
    }
    
    pub fn set_up(&mut self, _0: &BenchmarkState)  {
        
        todo!();
        /*
            random_device randomDevice;
        auto rng = mt19937(randomDevice());
        auto s32rng =
            bind(uniform_int_distribution<i32>(-10000, 10000), rng);
        auto u8rng = bind(uniform_int_distribution<u8>(), rng);

        a_.resize(mc() * kc());
        generate(a_.begin(), a_.end(), ref(u8rng));
        k_.resize(nc() * kc());
        generate(k_.begin(), k_.end(), ref(u8rng));
        b_.resize(nc());
        generate(b_.begin(), b_.end(), ref(s32rng));
        w_.resize(
            kcStride() * ncStride() +
            ncStride() * sizeof(i32) / sizeof(u8));
        fill(w_.begin(), w_.end(), 127);
        usize num_zero_points_kernel = (nc_ + (nr_ -1)) & -nr_;
        vector<u8> kernel_zero_points(num_zero_points_kernel, 127);
        vector<float> requantization_scales(num_zero_points_kernel, 0.75f);
        pytorch_pack_q8gemm_w(
            nc(),
            kc(),
            nr(),
            np(),
            kr(),
    #if !PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
            127,
            127,
    #endif
            k(),
            b(),
    #if PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
            kernel_zero_points.data(),
    #endif
            w());
        c_.resize(mc() * nc());
        fill(c_.begin(), c_.end(), 0xA5);

        quantizationParams_ = pytorch_qnnp_compute_conv_quantization_params(
            127, kernel_zero_points.data(),
            requantization_scales.data(), 127, 1, 254);
        */
    }
    
    pub fn tear_down(&mut self, state: &mut BenchmarkState)  {
        
        todo!();
        /*
            state.SetItemsProcessed(
            u64(state.iterations()) * 2 * mc() * nc() * kc());
        a_.clear();
        k_.clear();
        b_.clear();
        w_.clear();
        c_.clear();
        */
    }
    
    #[inline] pub fn a(&self) -> *const u8 {
        
        todo!();
        /*
            return a_.data();
        */
    }
    
    #[inline] pub fn k(&self) -> *const u8 {
        
        todo!();
        /*
            return k_.data();
        */
    }
    
    #[inline] pub fn b(&self) -> *const i32 {
        
        todo!();
        /*
            return b_.data();
        */
    }
    
    #[inline] pub fn w(&mut self) -> *mut u8 {
        
        todo!();
        /*
            return w_.data();
        */
    }
    
    #[inline] pub fn w(&self) -> *const u8 {
        
        todo!();
        /*
            return w_.data();
        */
    }
    
    #[inline] pub fn c(&mut self) -> *mut u8 {
        
        todo!();
        /*
            return c_.data();
        */
    }
    
    #[inline] pub fn mr(&self) -> u32 {
        
        todo!();
        /*
            return mr_;
        */
    }
    
    #[inline] pub fn mc(&self) -> u32 {
        
        todo!();
        /*
            return mc_;
        */
    }
    
    #[inline] pub fn nr(&self) -> u32 {
        
        todo!();
        /*
            return nr_;
        */
    }
    
    #[inline] pub fn np(&self) -> u32 {
        
        todo!();
        /*
            return np_;
        */
    }
    
    #[inline] pub fn nc(&self) -> u32 {
        
        todo!();
        /*
            return nc_;
        */
    }
    
    #[inline] pub fn nc_stride(&self) -> u32 {
        
        todo!();
        /*
            return roundUp(nc(), nr());
        */
    }
    
    #[inline] pub fn kr(&self) -> u32 {
        
        todo!();
        /*
            return kr_;
        */
    }
    
    #[inline] pub fn kc(&self) -> u32 {
        
        todo!();
        /*
            return kc_;
        */
    }
    
    #[inline] pub fn kc_stride(&self) -> u32 {
        
        todo!();
        /*
            return roundUp(kc(), kr());
        */
    }
    
    #[inline] pub fn quantization_params(&self) -> *const PyTorchQnnpConvQuantizationParams {
        
        todo!();
        /*
            return &quantizationParams_;
        */
    }
}

pub struct Q8GEMM_Op<const MR: u32,const NR: u32,const NP: u32,const KR: u32> {
    base: Q8GEMM,
}

impl<const MR: u32,const NR: u32,const NP: u32,const KR: u32> Q8GEMM_Op<MR,NR,NP,KR> {
    
    pub fn new() -> Self {
    
        todo!();
        /*
        : Q8GEMM(MR, NR, NP, KR),

        
        */
    }
    
    pub fn set_up(&mut self, state: &BenchmarkState)  {
        
        todo!();
        /*
            mc_ = state.range(0);
        nc_ = state.range(1);
        kc_ = state.range(2);

        Q8GEMM::SetUp(state);
        */
    }
}

pub struct Q8GEMMSparse {
    base:                BenchmarkFixture,
    a:                   Vec<u8>,
    k:                   Vec<u8>,
    b:                   Vec<f32>,
    bcsr_matrix:         Box<QnnpackBCSRMatrix>,
    c:                   Vec<f32>,
    mr:                  u32, // default = { 0 }
    nr:                  u32, // default = { 0 }
    kr:                  u32, // default = { 0 }
    mc:                  u32, // default = { mr_ }
    nc:                  u32, // default = { nr_ }
    kc:                  u32, // default = { kr_ }
    row_block_size:      u32, // default = { 1 }
    col_block_size:      u32, // default = { 4 }
    sparsity:            f32, // default = { 0.7f }
    quantization_params: PyTorchQnnpConvDynamicQuantizationParams,
}

impl Q8GEMMSparse {
    
    pub fn new(
        mr:  u32,
        nr:  u32,
        kr:  u32,
        rbs: u32,
        cbs: u32) -> Self {
    
        todo!();
        /*
        : mr(mr),
        : nr(nr),
        : kr(kr),
        : mc(mr),
        : nc(nr),
        : kc(kr),
        : row_block_size(rbs),
        : col_block_size(cbs),

        
        */
    }
    
    pub fn set_up(&mut self, _0: &BenchmarkState)  {
        
        todo!();
        /*
            random_device randomDevice;
        auto rng = mt19937(randomDevice());
        auto s32rng =
            bind(uniform_int_distribution<i32>(-10000, 10000), rng);
        auto u8rng = bind(uniform_int_distribution<u8>(), rng);
        auto f32rng =
            bind(uniform_real_distribution<float>(1, 5), rng);

        a_.resize(mc() * kc());
        generate(a_.begin(), a_.end(), ref(u8rng));
        k_.resize(nc() * kc());
        b_.resize(nc());
        generate(b_.begin(), b_.end(), ref(f32rng));
        usize num_zero_points_kernel = (nc_ + (nr_ -1)) & -nr_;
        vector<u8> kernel_zero_points(num_zero_points_kernel, 127);

        generate(k_.begin(), k_.end(), ref(u8rng));
        fillBlockSparseWeights(
            k_.data(),
            nc(),
            kc(),
            rowBlockSize(),
            colBlockSize(),
            sparsity(),
            kernel_zero_points.data());
        bcsr_matrix_ =
          qnnpack::generateBlockCSRMatrix(
              k_.data(),
              nc(),
              kc(),
              rowBlockSize(),
              colBlockSize(),
              kernel_zero_points.data());
        vector<float> dequantization_scales(num_zero_points_kernel, 0.75f);
        c_.resize(mc() * nc());
        fill(c_.begin(), c_.end(), 0xA5);

        quantizationParams_ = pytorch_qnnp_conv_dynamic_quantization_params{
          127,
          kernel_zero_points.data(),
          dequantization_scales.data(),
        };
        */
    }
    
    pub fn tear_down(&mut self, state: &mut BenchmarkState)  {
        
        todo!();
        /*
            state.SetItemsProcessed(
            u64(state.iterations()) * 2 * mc() * nc() * kc());
        a_.clear();
        k_.clear();
        b_.clear();
        c_.clear();
        */
    }
    
    #[inline] pub fn a(&self) -> *const u8 {
        
        todo!();
        /*
            return a_.data();
        */
    }
    
    #[inline] pub fn k(&self) -> *const u8 {
        
        todo!();
        /*
            return k_.data();
        */
    }
    
    #[inline] pub fn b(&self) -> *const f32 {
        
        todo!();
        /*
            return b_.data();
        */
    }
    
    #[inline] pub fn c(&mut self) -> *mut f32 {
        
        todo!();
        /*
            return c_.data();
        */
    }
    
    #[inline] pub fn mr(&self) -> u32 {
        
        todo!();
        /*
            return mr_;
        */
    }
    
    #[inline] pub fn mc(&self) -> u32 {
        
        todo!();
        /*
            return mc_;
        */
    }
    
    #[inline] pub fn nr(&self) -> u32 {
        
        todo!();
        /*
            return nr_;
        */
    }
    
    #[inline] pub fn nc(&self) -> u32 {
        
        todo!();
        /*
            return nc_;
        */
    }
    
    #[inline] pub fn nc_stride(&self) -> u32 {
        
        todo!();
        /*
            return roundUp(nc(), nr());
        */
    }
    
    #[inline] pub fn kr(&self) -> u32 {
        
        todo!();
        /*
            return kr_;
        */
    }
    
    #[inline] pub fn kc(&self) -> u32 {
        
        todo!();
        /*
            return kc_;
        */
    }
    
    #[inline] pub fn kc_stride(&self) -> u32 {
        
        todo!();
        /*
            return roundUp(kc(), kr());
        */
    }
    
    #[inline] pub fn row_block_size(&self) -> usize {
        
        todo!();
        /*
            return this->row_block_size_;
        */
    }
    
    #[inline] pub fn col_block_size(&self) -> usize {
        
        todo!();
        /*
            return this->col_block_size_;
        */
    }
    
    #[inline] pub fn sparsity(&self) -> f32 {
        
        todo!();
        /*
            return this->sparsity_;
        */
    }
    
    #[inline] pub fn quantization_params(&self) -> *const PyTorchQnnpConvDynamicQuantizationParams {
        
        todo!();
        /*
            return &quantizationParams_;
        */
    }
}

pub struct Q8GEMMSparse_Op<const MR: u32,const NR: u32,const KR: u32,const RBS: u32,const CBS: u32> {
    base: Q8GEMMSparse,
}

impl<const MR: u32,const NR: u32,const KR: u32,const RBS: u32,const CBS: u32> Q8GEMMSparse_Op<MR,NR,KR,RBS,CBS> {

    pub fn new() -> Self {
    
        todo!();
        /*
        : q8gemm_sparse(MR, NR, KR, RBS, CBS),

        
        */
    }
    
    pub fn set_up(&mut self, state: &BenchmarkState)  {
        
        todo!();
        /*
            mc_ = state.range(0);
        nc_ = state.range(1);
        kc_ = state.range(2);

        Q8GEMMSparse::SetUp(state);
        */
    }
}

pub fn sparse_gemm_bench_gemm_arguments(b: *mut Benchmark)  {
    
    todo!();
        /*
            b->ArgNames({"M", "N", "K"});

      b->Args({5, 4096, 640});
      b->Args({20, 4096, 640});
      b->Args({4, 4096, 1024});
      b->Args({3, 4096, 1024});
      b->Args({5, 1024, 640});
      b->Args({5, 4096, 1280});
      b->Args({20, 4096, 880});
      b->Args({10, 4096, 640});
      b->Args({10, 4096, 1280});
      b->Args({5, 4096, 1024});
      b->Args({6, 4096, 1024});
      b->Args({7, 4096, 1024});
      b->Args({8, 4096, 1024});
      b->Args({9, 4096, 1024});
      b->Args({7, 4096, 640});
      b->Args({4, 4096, 640});
      b->Args({28, 4096, 640});
      b->Args({16, 4096, 640});
      b->Args({10, 4096, 1024});
      b->Args({8, 4096, 640});
      b->Args({8, 4096, 1280});
      b->Args({7, 1024, 640});
      b->Args({7, 4096, 1280});
      b->Args({4, 1024, 640});
      b->Args({4, 4096, 1280});
      b->Args({28, 4096, 880});
      b->Args({16, 4096, 880});
      b->Args({14, 4096, 640});
      b->Args({14, 4096, 1280});
        */
}

#[cfg(CPUINFO_ARCH_ARM)]
lazy_static!{
    /*
    BENCHMARK_TEMPLATE_DEFINE_F(Q8GEMM_Op, 4x8__aarch32_neon, 4, 8, 8, 1)
    (BenchmarkState& state) {
      for (auto _ : state) {
        for (u32 m = 0; m < mc(); m += mr()) {
          const u32 mrr = min(mc() - m, mr());
          for (u32 n = 0, channel_offset = 0; n < nc();
              n += nr(), channel_offset += nr()) {
            const u32 nrr = min(nc() - n, nr());
            pytorch_q8gemm_ukernel_4x8__aarch32_neon(
                mrr,
                nrr,
                kc(),
                a() + m * kc(),
                kc() * sizeof(u8),
                w() + n * (kcStride() * sizeof(u8) + sizeof(i32)),
                c() + m * nc() + n,
                nc() * sizeof(u8),
                channel_offset,
                quantizationParams());
          }
        }
      }
    }

    BENCHMARK_REGISTER_F(Q8GEMM_Op, 4x8__aarch32_neon)
        ->Apply(SparseGEMMBenchGemmArguments);

    BENCHMARK_TEMPLATE_DEFINE_F(Q8GEMMSparse_Op, 4x8c1x4_prepacked__aarch32_neon, 4, 8, 4, 1, 4)
    (BenchmarkState& state) {
      for (auto _ : state) {
        auto m_blocks = (mc() + mr()  - 1) / mr();
        auto k_blocks = (kc() + 4  - 1) / 4;
        vector<u8> a_packed(m_blocks * k_blocks * mr() * 4 + 8);
        for (u32 m = 0; m < mc(); m += mr()) {
          const u32 mrr = min(mc() - m, mr());
          for (u32 n = 0, channel_offset = 0; n < nc();
              n += nr(), channel_offset += nr()) {
            const u32 nrr = min(nc() - n, nr());
            pytorch_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon(
                mrr,
                kc(),
                a() + m * kc(),
                kc() * sizeof(u8),
                a_packed.data() + (m >> 2) * (k_blocks << 2) * mr()
                );
          }
        }
        for (u32 m = 0; m < mc(); m += mr()) {
          const u32 mrr = min(mc() - m, mr());
          for (u32 n = 0, channel_offset = 0; n < nc();
              n += nr(), channel_offset += nr()) {
            const u32 nrr = min(nc() - n, nr());
            pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA__aarch32_neon(
                mrr,
                nrr,
                a_packed.data() + (m >> 2) * (k_blocks << 2) * mr(),
                bcsr_matrix_->values.data(),
                bcsr_matrix_->row_values.data() + n,
                bcsr_matrix_->col_indices.data(),
                b() + n,
                c() + m * nc() + n,
                nc(),
                channel_offset,
                quantizationParams());
          }
        }
      }
    }
    BENCHMARK_REGISTER_F(Q8GEMMSparse_Op, 4x8c1x4_prepacked__aarch32_neon)
        ->Apply(SparseGEMMBenchGemmArguments);

    BENCHMARK_TEMPLATE_DEFINE_F(Q8GEMMSparse_Op, 4x8c8x1_prepacked__aarch32_neon, 4, 8, 1, 8, 1)
    (BenchmarkState& state) {
      for (auto _ : state) {
        auto m_blocks = (mc() + mr()  - 1) / mr();
        // Still use kr of 4 because we use 4x4 packing kernel
        auto k_blocks = (kc() + 4  - 1) / 4;
        vector<u8> a_packed(m_blocks * k_blocks * mr() * 4 + 8);
        for (u32 m = 0; m < mc(); m += mr()) {
          const u32 mrr = min(mc() - m, mr());
          for (u32 n = 0, channel_offset = 0; n < nc();
              n += nr(), channel_offset += nr()) {
            const u32 nrr = min(nc() - n, nr());
            pytorch_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon(
                mrr,
                kc(),
                a() + m * kc(),
                kc() * sizeof(u8),
                a_packed.data() + (m >> 2) * (k_blocks << 2) * mr()
                );
          }
        }
        for (u32 m = 0; m < mc(); m += mr()) {
          const u32 mrr = min(mc() - m, mr());
          for (u32 n = 0, channel_offset = 0; n < nc();
              n += nr(), channel_offset += nr()) {
            const u32 nrr = min(nc() - n, nr());
            pytorch_q8gemm_dq_sparse_8x1_ukernel_4x8_packedA__aarch32_neon(
                mrr,
                nrr,
                a_packed.data() + (m >> 2) * (k_blocks << 2) * mr(),
                bcsr_matrix_->values.data(),
                bcsr_matrix_->row_values.data() + (n >> 3),
                bcsr_matrix_->col_indices.data(),
                b() + n,
                c() + m * nc() + n,
                nc(),
                channel_offset,
                quantizationParams());
          }
        }
      }
    }
    BENCHMARK_REGISTER_F(Q8GEMMSparse_Op, 4x8c8x1_prepacked__aarch32_neon)
        ->Apply(SparseGEMMBenchGemmArguments);
    */
}

#[cfg(CPUINFO_ARCH_ARM64)]
lazy_static!{
    /*
    BENCHMARK_TEMPLATE_DEFINE_F(Q8GEMM_Op, 8x8__aarch64_neon, 8, 8, 8, 1)
    (BenchmarkState& state) {
      for (auto _ : state) {
        for (u32 m = 0; m < mc(); m += mr()) {
          const u32 mrr = min(mc() - m, mr());
          for (u32 n = 0, channel_offset = 0; n < nc();
              n += nr(), channel_offset += nr()) {
            const u32 nrr = min(nc() - n, nr());
            pytorch_q8gemm_ukernel_8x8__aarch64_neon(
                mrr,
                nrr,
                kc(),
                a() + m * kc(),
                kc() * sizeof(u8),
                w() + n * (kcStride() * sizeof(u8) + sizeof(i32)),
                c() + m * nc() + n,
                nc() * sizeof(u8),
                channel_offset,
                quantizationParams());
          }
        }
      }
    }

    BENCHMARK_REGISTER_F(Q8GEMM_Op, 8x8__aarch64_neon)
        ->Apply(SparseGEMMBenchGemmArguments);

    BENCHMARK_TEMPLATE_DEFINE_F(Q8GEMMSparse_Op, 8x8c1x4_prepacked__aarch64_neon, 8, 8, 4, 1, 4)
    (BenchmarkState& state) {
      for (auto _ : state) {
        auto m_blocks = (mc() + mr()  - 1) / mr();
        auto k_blocks = (kc() + 4  - 1) / 4;
        vector<u8> a_packed(m_blocks * k_blocks * mr() * 4 + 8);
        for (u32 m = 0; m < mc(); m += mr()) {
          const u32 mrr = min(mc() - m, mr());
          for (u32 n = 0, channel_offset = 0; n < nc();
              n += nr(), channel_offset += nr()) {
            const u32 nrr = min(nc() - n, nr());
            pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch64_neon(
                mrr,
                kc(),
                a() + m * kc(),
                kc() * sizeof(u8),
                a_packed.data() + (m >> 3) * (k_blocks << 2) * mr()
                );
          }
        }
        for (u32 m = 0; m < mc(); m += mr()) {
          const u32 mrr = min(mc() - m, mr());
          for (u32 n = 0, channel_offset = 0; n < nc();
              n += nr(), channel_offset += nr()) {
            const u32 nrr = min(nc() - n, nr());
            pytorch_q8gemm_dq_sparse_1x4_ukernel_8x8_packedA__aarch64_neon(
                mrr,
                nrr,
                a_packed.data() + (m >> 3) * (k_blocks << 2) * mr(),
                bcsr_matrix_->values.data(),
                bcsr_matrix_->row_values.data(),
                bcsr_matrix_->col_indices.data(),
                b() + n,
                c() + m * nc() + n,
                nc(),
                channel_offset,
                quantizationParams());
          }
        }
      }
    }
    BENCHMARK_REGISTER_F(Q8GEMMSparse_Op, 8x8c1x4_prepacked__aarch64_neon)
        ->Apply(SparseGEMMBenchGemmArguments);

    BENCHMARK_TEMPLATE_DEFINE_F(Q8GEMMSparse_Op, 8x8c8x1_prepacked__aarch64_neon, 8, 8, 4, 8, 1)
    (BenchmarkState& state) {
      for (auto _ : state) {
        auto m_blocks = (mc() + mr()  - 1) / mr();
        // Still use kr of 4 because we use 4x4 packing kernel
        auto k_blocks = (kc() + 4  - 1) / 4;
        vector<u8> a_packed(m_blocks * k_blocks * mr() * 4 + 8);
        for (u32 m = 0; m < mc(); m += mr()) {
          const u32 mrr = min(mc() - m, mr());
          for (u32 n = 0, channel_offset = 0; n < nc();
              n += nr(), channel_offset += nr()) {
            const u32 nrr = min(nc() - n, nr());
            pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch64_neon(
                mrr,
                kc(),
                a() + m * kc(),
                kc() * sizeof(u8),
                a_packed.data() + (m >> 3) * (k_blocks << 2) * mr()
                );
          }
        }
        for (u32 m = 0; m < mc(); m += mr()) {
          const u32 mrr = min(mc() - m, mr());
          for (u32 n = 0, channel_offset = 0; n < nc();
              n += nr(), channel_offset += nr()) {
            const u32 nrr = min(nc() - n, nr());
            pytorch_q8gemm_dq_sparse_8x1_ukernel_8x8_packedA__aarch64_neon(
                mrr,
                nrr,
                a_packed.data() + (m >> 3) * (k_blocks << 2) * mr(),
                bcsr_matrix_->values.data(),
                bcsr_matrix_->row_values.data(),
                bcsr_matrix_->col_indices.data(),
                b() + n,
                c() + m * nc() + n,
                nc(),
                channel_offset,
                quantizationParams());
          }
        }
      }
    }
    BENCHMARK_REGISTER_F(Q8GEMMSparse_Op, 8x8c8x1_prepacked__aarch64_neon)
        ->Apply(SparseGEMMBenchGemmArguments);
    */
}

#[cfg(not(PYTORCH_QNNPACK_BENCHMARK_NO_MAIN))]
lazy_static!{
    /*
    BENCHMARK_MAIN();
    */
}
