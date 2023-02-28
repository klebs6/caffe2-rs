crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/bench/hgemm.cc]

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

pub struct HGEMM {
    base: BenchmarkFixture,
    a:    Vec<u16>,
    k:    Vec<u16>,
    b:    Vec<u16>,
    w:    Vec<u16,AlignedAllocator<u16,32>>,
    c:    Vec<u16>,

    /**
      | default = {0};
      |
      */
    mr:              u32,

    /**
      | default = {0};
      |
      */
    nr:              u32,

    /**
      | default = {0};
      |
      */
    kr:              u32,

    /**
      | default = {mr_};
      |
      */
    mc:              u32,

    /**
      | default = {nr_};
      |
      */
    nc:              u32,

    /**
      | default = {kr_};
      |
      */
    kc:              u32,

    /**
      | default = {0x3C00, 0x7C00, 0xFC00};
      |
      */
    clamping_params: PyTorchQnnpFp16ClampingParams,
}

impl HGEMM {
    
    pub fn new(
        mr: u32,
        nr: u32,
        kr: u32) -> Self {
    
        todo!();
        /*
        : mr(mr),
        : nr(nr),
        : kr(kr),
        : mc(mr),
        : nc(nr),
        : kc(kr),

        
        */
    }
    
    pub fn set_up(&mut self, _0: &BenchmarkState)  {
        
        todo!();
        /*
            const uint_fast32_t seed =
            chrono::system_clock::now().time_since_epoch().count();
        auto rng = bind(
            fp16_ieee_from_fp32_value,
            bind(uniform_real_distribution<float>(), mt19937(seed)));

        a_.resize(mc() * kc());
        generate(a_.begin(), a_.end(), ref(rng));
        k_.resize(nc() * kc());
        generate(k_.begin(), k_.end(), ref(rng));
        b_.resize(nc());
        generate(b_.begin(), b_.end(), ref(rng));
        w_.resize(ncStride() * kcStride() + ncStride());
        fill(w_.begin(), w_.end(), 0);
        pytorch_pack_hgemm_w(nc(), kc(), nr(), kr(), k(), b(), w());
        c_.resize(mc() * nc());
        fill(c_.begin(), c_.end(), UINT16_C(0x7E00) /* NaN */);
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
    
    #[inline] pub fn a(&self) -> *const u16 {
        
        todo!();
        /*
            return a_.data();
        */
    }
    
    #[inline] pub fn k(&self) -> *const u16 {
        
        todo!();
        /*
            return k_.data();
        */
    }
    
    #[inline] pub fn b(&self) -> *const u16 {
        
        todo!();
        /*
            return b_.data();
        */
    }
    
    #[inline] pub fn w(&mut self) -> *mut u16 {
        
        todo!();
        /*
            return w_.data();
        */
    }
    
    #[inline] pub fn w(&self) -> *const u16 {
        
        todo!();
        /*
            return w_.data();
        */
    }
    
    #[inline] pub fn c(&mut self) -> *mut u16 {
        
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
    
    #[inline] pub fn clamping_params(&self) -> *const PyTorchQnnpFp16ClampingParams {
        
        todo!();
        /*
            return &clampingParams_;
        */
    }
}

pub struct HGEMM_L1<const MR: u32,const NR: u32,const KR: u32> {
    base: HGEMM,
}

impl<const MR: u32,const NR: u32,const KR: u32> HGEMM_L1<MR,NR,KR> {
    
    pub fn new() -> Self {
    
        todo!();
        /*
        : HGEMM(MR, NR, KR),

            cpuinfo_initialize();
        const usize l1d_size = cpuinfo_get_l1d_cache(0)->size;
        const usize l1d_reserve = 512;
        kc_ = ((l1d_size - l1d_reserve) / sizeof(u16) - mr() * nr()) /
            (mr() + nr());
        if (kr() != 1) {
          kc_ = kc_ / kr() * kr();
        } else {
          kc_ = kc_ / nr() * nr();
        }
        */
    }
}

pub struct HGEMM_Op<const MR: u32,const NR: u32,const KR: u32> {
    base: HGEMM,
}

impl<const MR: u32,const NR: u32,const KR: u32> HGEMM_Op<MR,NR,KR> {
    
    pub fn new() -> Self {
    
        todo!();
        /*
        : HGEMM(MR, NR, KR),

        
        */
    }
    
    pub fn set_up(&mut self, state: &BenchmarkState)  {
        
        todo!();
        /*
            mc_ = state.range(0);
        nc_ = state.range(1);
        kc_ = state.range(2);

        HGEMM::SetUp(state);
        */
    }
}

pub fn shuffle_net_v1g1gemm_arguments(b: *mut Benchmark)  {
    
    todo!();
        /*
            b->ArgNames({"M", "N", "K"});

      /* group = 1 */
      b->Args({56 * 56, 30, 24});
      b->Args({28 * 28, 120, 30});
      b->Args({28 * 28, 36, 144});
      b->Args({28 * 28, 144, 36});
      b->Args({14 * 14, 144, 36});
      b->Args({14 * 14, 72, 288});
      b->Args({14 * 14, 288, 72});
      b->Args({7 * 7, 288, 72});
      b->Args({7 * 7, 144, 576});
      b->Args({7 * 7, 576, 144});
        */
}

pub fn shuffle_net_v1g2gemm_arguments(b: *mut Benchmark)  {
    
    todo!();
        /*
            b->ArgNames({"M", "N", "K"});

      /* group = 2 */
      b->Args({56 * 56, 22, 12});
      b->Args({28 * 28, 88, 22});
      b->Args({28 * 28, 25, 100});
      b->Args({28 * 28, 100, 25});
      b->Args({14 * 14, 100, 25});
      b->Args({14 * 14, 50, 200});
      b->Args({14 * 14, 200, 50});
      b->Args({7 * 7, 200, 50});
      b->Args({7 * 7, 100, 400});
      b->Args({7 * 7, 400, 100});
        */
}

pub fn shuffle_net_v1g3gemm_arguments(b: *mut Benchmark)  {
    
    todo!();
        /*
            b->ArgNames({"M", "N", "K"});

      /* group = 3 */
      b->Args({56 * 56, 18, 8});
      b->Args({28 * 28, 72, 18});
      b->Args({28 * 28, 20, 80});
      b->Args({28 * 28, 80, 20});
      b->Args({14 * 14, 80, 20});
      b->Args({14 * 14, 40, 160});
      b->Args({14 * 14, 160, 40});
      b->Args({7 * 7, 160, 40});
      b->Args({7 * 7, 80, 320});
      b->Args({7 * 7, 320, 80});
        */
}


pub fn shuffle_net_v1g4gemm_arguments(b: *mut Benchmark)  {
    
    todo!();
        /*
            b->ArgNames({"M", "N", "K"});

      /* group = 4 */
      b->Args({56 * 56, 15, 6});
      b->Args({28 * 28, 62, 15});
      b->Args({28 * 28, 17, 68});
      b->Args({28 * 28, 68, 17});
      b->Args({14 * 14, 68, 17});
      b->Args({14 * 14, 34, 136});
      b->Args({14 * 14, 136, 34});
      b->Args({7 * 7, 136, 34});
      b->Args({7 * 7, 68, 272});
      b->Args({7 * 7, 272, 68});
        */
}


pub fn shuffle_net_v1g8gemm_arguments(b: *mut Benchmark)  {
    
    todo!();
        /*
            b->ArgNames({"M", "N", "K"});

      /* group = 8 */
      b->Args({56 * 56, 11, 3});
      b->Args({28 * 28, 45, 11});
      b->Args({28 * 28, 12, 48});
      b->Args({28 * 28, 48, 12});
      b->Args({14 * 14, 48, 12});
      b->Args({14 * 14, 24, 96});
      b->Args({14 * 14, 96, 24});
      b->Args({7 * 7, 96, 24});
      b->Args({7 * 7, 48, 192});
      b->Args({7 * 7, 192, 48});
        */
}


pub fn mobile_net_v1gemm_arguments(b: *mut Benchmark)  {
    
    todo!();
        /*
            b->ArgNames({"M", "N", "K"});

      b->Args({112 * 112, 32, 3 * 3 * 3});
      b->Args({112 * 112, 64, 32});
      b->Args({56 * 56, 128, 64});
      b->Args({56 * 56, 128, 128});
      b->Args({28 * 28, 256, 128});
      b->Args({28 * 28, 256, 256});
      b->Args({14 * 14, 512, 256});
      b->Args({14 * 14, 512, 512});
      b->Args({7 * 7, 1024, 512});
      b->Args({7 * 7, 1024, 1024});
        */
}


pub fn squeeze_net_v10gemm_arguments(b: *mut Benchmark)  {
    
    todo!();
        /*
            b->ArgNames({"M", "N", "K"});

      /* Conv 1 */
      b->Args({111 * 111, 96, 7 * 7 * 3});
      /* Fire 2 */
      b->Args({55 * 55, 16, 96});
      b->Args({55 * 55, 64, 16});
      b->Args({55 * 55, 64, 3 * 3 * 16});
      /* Fire 3 */
      b->Args({55 * 55, 16, 128});
      b->Args({55 * 55, 64, 16});
      b->Args({55 * 55, 64, 3 * 3 * 16});
      /* Fire 4 */
      b->Args({55 * 55, 32, 128});
      b->Args({55 * 55, 128, 32});
      b->Args({55 * 55, 128, 3 * 3 * 32});
      /* Fire 5 */
      b->Args({27 * 27, 32, 256});
      b->Args({27 * 27, 128, 32});
      b->Args({27 * 27, 128, 3 * 3 * 32});
      /* Fire 6 */
      b->Args({27 * 27, 48, 256});
      b->Args({27 * 27, 192, 48});
      b->Args({27 * 27, 192, 3 * 3 * 48});
      /* Fire 7 */
      b->Args({27 * 27, 48, 384});
      b->Args({27 * 27, 192, 48});
      b->Args({27 * 27, 192, 3 * 3 * 48});
      /* Fire 8 */
      b->Args({27 * 27, 64, 384});
      b->Args({27 * 27, 256, 64});
      b->Args({27 * 27, 256, 3 * 3 * 64});
      /* Fire 9 */
      b->Args({13 * 13, 64, 512});
      b->Args({13 * 13, 256, 64});
      b->Args({13 * 13, 256, 3 * 3 * 64});
      /* Conv 10 */
      b->Args({13 * 13, 1000, 512});
        */
}


pub fn gemm_arguments(b: *mut Benchmark)  {
    
    todo!();
        /*
            for (auto S = 15; S <= 128; S *= 2) {
        for (int K = 8; K <= 1024; K *= 2) {
          b->Args({S * S, K, K});
        }
      }
        */
}

#[cfg(CPUINFO_ARCH_ARM)]
lazy_static!{
    /*
    BENCHMARK_TEMPLATE_F(HGEMM_L1, 8x8__aarch32_neonfp16arith, 8, 8, 1)
    (BenchmarkState& state) {
      if (!cpuinfo_initialize() || !cpuinfo_has_arm_neon_fp16_arith()) {
        state.SkipWithError("NEON FP16 compute is not supported");
      }
      for (auto _ : state) {
        pytorch_hgemm_ukernel_8x8__aarch32_neonfp16arith(
            mr(),
            nr(),
            kc(),
            a(),
            kc() * sizeof(u16),
            w() + nc() * (kcStride() + 1),
            c(),
            mr() * sizeof(u16),
            clampingParams());
      }
    }
    */
}

#[cfg(CPUINFO_ARCH_ARM)]
lazy_static!{
    /*
    BENCHMARK_TEMPLATE_DEFINE_F(HGEMM_Op, 8x8__aarch32_neonfp16arith, 8, 8, 1)
    (BenchmarkState& state) {
      if (!cpuinfo_initialize() || !cpuinfo_has_arm_neon_fp16_arith()) {
        state.SkipWithError("NEON FP16 compute is not supported");
      }
      for (auto _ : state) {
        for (u32 m = 0; m < mc(); m += mr()) {
          const u32 mrr = min(mc() - m, mr());
          for (u32 n = 0; n < nc(); n += nr()) {
            const u32 nrr = min(nc() - n, nr());
            pytorch_hgemm_ukernel_8x8__aarch32_neonfp16arith(
                mrr,
                nrr,
                kc(),
                a() + m * kc(),
                kc() * sizeof(u16),
                w() + n * (kcStride() + 1),
                c() + m * nc() + n,
                nc() * sizeof(u16),
                clampingParams());
          }
        }
      }
    }
    */
}

#[cfg(CPUINFO_ARCH_ARM)]
lazy_static!{
    /*
    BENCHMARK_REGISTER_F(HGEMM_Op, 8x8__aarch32_neonfp16arith)
        ->Apply(ShuffleNetV1G1GemmArguments);
    BENCHMARK_REGISTER_F(HGEMM_Op, 8x8__aarch32_neonfp16arith)
        ->Apply(MobileNetV1GemmArguments);
    BENCHMARK_REGISTER_F(HGEMM_Op, 8x8__aarch32_neonfp16arith)
        ->Apply(SqueezeNetV10GemmArguments);
    BENCHMARK_REGISTER_F(HGEMM_Op, 8x8__aarch32_neonfp16arith)
        ->Apply(GemmArguments);
    */
}

#[cfg(not(PYTORCH_QNNPACK_BENCHMARK_NO_MAIN))]
lazy_static!{
    /*
    BENCHMARK_MAIN();
    */
}
