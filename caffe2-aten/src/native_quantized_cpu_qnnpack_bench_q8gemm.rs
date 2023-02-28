crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/bench/q8gemm.cc]

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

#[cfg(PYTORCH_QNNPACK_BENCHMARK_GEMMLOWP)]
pub struct GemmlowpOutputPipeline {

}

#[cfg(PYTORCH_QNNPACK_BENCHMARK_GEMMLOWP)]
pub mod gemmlowp_output_pipeline {

    use super::*;

    pub type ColVectorMap = GemmlowpVectorMap<i32,GemmlowpVectorShapeCol>;

    pub type Pipeline = (
        GemmlowpOutputStageBiasAddition<ColVectorMap>,
        GemmlowpOutputStageQuantizeDownInt32ToUint8ScaleByFixedPoint,
        GemmlowpOutputStageClamp,
        GemmlowpOutputStageSaturatingCastToUint8
    );
}

#[cfg(PYTORCH_QNNPACK_BENCHMARK_GEMMLOWP)]
impl GemmlowpOutputPipeline {
    
    pub fn make(
        bias_data:             *const i32,
        output_rows:           i32,
        output_offset:         i32,
        output_multiplier:     i32,
        output_shift:          i32,
        output_activation_min: i32,
        output_activation_max: i32) -> Pipeline {
        
        todo!();
        /*
            ColVectorMap bias_vector(bias_data, output_rows);
        gemmlowp::OutputStageBiasAddition<ColVectorMap> bias_addition_stage;
        bias_addition_stage.bias_vector = bias_vector;
        gemmlowp::OutputStageQuantizeDownInt32ToUint8ScaleByFixedPoint
            quantize_down_stage;
        quantize_down_stage.result_offset_after_shift = output_offset;
        quantize_down_stage.result_fixedpoint_multiplier = output_multiplier;
        quantize_down_stage.result_shift = output_shift;
        gemmlowp::OutputStageClamp clamp_stage;
        clamp_stage.min = output_activation_min;
        clamp_stage.max = output_activation_max;
        gemmlowp::OutputStageSaturatingCastToUint8 saturating_cast_stage;
        return make_tuple(
            bias_addition_stage,
            quantize_down_stage,
            clamp_stage,
            saturating_cast_stage);
        */
    }
}

pub struct Q8GEMM {

    base:                BenchmarkFixture,

    a:                   Vec<u8>,
    k:                   Vec<u8>,
    b:                   Vec<i32>,
    w:                   Vec<u8,AlignedAllocator<u8,32>>,
    c:                   Vec<u8>,

    /**
      | default = {0};
      |
      */
    mr:                  u32,


    /**
      | default = {0};
      |
      */
    nr:                  u32,


    /**
      | default = {0};
      |
      */
    np:                  u32,


    /**
      | default = {0};
      |
      */
    kr:                  u32,


    /**
      | default = {mr_};
      |
      */
    mc:                  u32,


    /**
      | default = {nr_};
      |
      */
    nc:                  u32,


    /**
      | default = {kr_};
      |
      */
    kc:                  u32,

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

pub struct Q8GEMM_L1<const MR: u32,const NR: u32,const NP: u32,const KR: u32> {
    base: Q8GEMM,
}

impl<const MR: u32,const NR: u32,const NP: u32,const KR: u32> Q8GEMM_L1<MR,NR,NP,KR> {
    
    pub fn new() -> Self {
    
        todo!();
        /*
        : Q8GEMM(MR, NR, NP, KR),

            cpuinfo_initialize();
        const usize l1d_size = cpuinfo_get_l1d_cache(0)->size;
        const usize l1d_reserve = 512;
        kc_ = ((l1d_size - l1d_reserve) / sizeof(u8) - mr() * nr()) /
            (mr() + nr());
        if (kr() != 1) {
          kc_ = kc_ / kr() * kr();
        } else {
          kc_ = kc_ / nr() * nr();
        }
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

pub struct Q8GEMM_XZP {
    base:                  Q8GEMM,
    a_row_sums:            Vec<i32>,
    requantization_params: PyTorchQnnpQ31RequantizationParams,
}

impl Q8GEMM_XZP {

    pub fn new(
        mr: u32,
        nr: u32,
        np: u32,
        kr: u32) -> Self {
    
        todo!();
        /*
        : Q8GEMM(mr, nr, np, kr),

        
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
        k_.resize(ncStride() * kcStride());
        generate(k_.begin(), k_.end(), ref(u8rng));
        b_.resize(roundUp(nc(), nr()));
        generate(b_.begin(), b_.end(), ref(s32rng));
        w_.resize(ncStride() * (kcStride() + sizeof(i32) / sizeof(u8)));
        fill(w_.begin(), w_.end(), 127);
        pytorch_pack_swizzle_q8gemm_b(
            nc(),
            kc(),
            np(),
            kr(),
            8,
    #if !PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
            127,
            127,
    #endif
            k(),
            b(),
            w());
        c_.resize(mc() * nc());
        fill(c_.begin(), c_.end(), 0xA5);
        aRowSums_.resize(roundUp(mc(), mr()));
        fill(aRowSums_.begin(), aRowSums_.end(), 0xFE01);

        requantizationParams_ =
            pytorch_qnnp_compute_requantization_params(0.75f, 127, 1, 254);
        */
    }
    
    pub fn tear_down(&mut self, state: &mut BenchmarkState)  {
        
        todo!();
        /*
            state.SetItemsProcessed(
            u64(state.iterations()) * 2 * mc() * nc() * kc());
        a_.clear();
        k_.clear();
        c_.clear();
        aRowSums_.clear();
        */
    }
    
    #[inline] pub fn a_row_sums(&mut self) -> *mut i32 {
        
        todo!();
        /*
            return aRowSums_.data();
        */
    }
    
    #[inline] pub fn a_row_sums(&self) -> *const i32 {
        
        todo!();
        /*
            return aRowSums_.data();
        */
    }
    
    #[inline] pub fn requantization_params(&self) -> *const PyTorchQnnpQ31RequantizationParams {
        
        todo!();
        /*
            return &requantizationParams_;
        */
    }
}

pub struct Q8GEMM_XZP_L1<const MR: u32,const NR: u32,const NP: u32,const KR: u32> {
    base: Q8GEMM_XZP,
}

impl<const MR: u32,const NR: u32,const NP: u32,const KR: u32> Q8GEMM_XZP_L1<MR,NR,NP,KR> {
    
    pub fn new() -> Self {
    
        todo!();
        /*
        : Q8GEMM_XZP(MR, NR, NP, KR),

            cpuinfo_initialize();
        const usize l1d_size = cpuinfo_get_l1d_cache(0)->size;
        const usize l1d_reserve = 512;
        kc_ = ((l1d_size - l1d_reserve) / sizeof(u8) - mr() * nr()) /
            (mr() + nr());
        if (kr() != 1) {
          kc_ = kc_ / kr() * kr();
        } else {
          kc_ = kc_ / nr() * nr();
        }
        */
    }
}

pub struct Q8GEMM_XZP_Op<const MR: u32,const NR: u32,const NP: u32,const KR: u32> {
    base: Q8GEMM_XZP,
}

impl<const MR: u32,const NR: u32,const NP: u32,const KR: u32> Q8GEMM_XZP_Op<MR,NR,NP,KR> {

    pub fn new() -> Self {
    
        todo!();
        /*
        : Q8GEMM_XZP(MR, NR, NP, KR),

        
        */
    }
    
    pub fn set_up(&mut self, state: &BenchmarkState)  {
        
        todo!();
        /*
            mc_ = state.range(0);
        nc_ = state.range(1);
        kc_ = state.range(2);

        Q8GEMM_XZP::SetUp(state);
        */
    }
}

pub struct COMPUTE_ROW_SUM_Op<const MR: u32,const NR: u32,const NP: u32,const KR: u32> {
    base: Q8GEMM_XZP,
}

impl<const MR: u32,const NR: u32,const NP: u32,const KR: u32> COMPUTE_ROW_SUM_Op<MR,NR,NP,KR> {
    
    pub fn new() -> Self {
    
        todo!();
        /*
        : Q8GEMM_XZP(MR, NR, NP, KR),

        
        */
    }
    
    pub fn set_up(&mut self, state: &BenchmarkState)  {
        
        todo!();
        /*
            mc_ = state.range(0);
        nc_ = state.range(1);
        kc_ = state.range(2);

        Q8GEMM_XZP::SetUp(state);
        */
    }
    
    pub fn tear_down(&mut self, state: &mut BenchmarkState)  {
        
        todo!();
        /*
            state.SetItemsProcessed(u64(state.iterations()) * (mc() * kc()));
        a_.clear();
        k_.clear();
        b_.clear();
        c_.clear();
        aRowSums_.clear();
        */
    }
}

#[cfg(PYTORCH_QNNPACK_BENCHMARK_GEMMLOWP)]
pub struct GEMMLOWP {
    base:              BenchmarkFixture,
    threading_context: GemmlowpMultiThreadGemmContext,
    a:                 Vec<u8>,
    k:                 Vec<u8>,
    b:                 Vec<i32>,
    c:                 Vec<u8>,
    mc:                u32,
    nc:                u32,
    kc:                u32,
}

#[cfg(PYTORCH_QNNPACK_BENCHMARK_GEMMLOWP)]
impl GEMMLOWP {
    
    pub fn set_up(&mut self, state: &BenchmarkState)  {
        
        todo!();
        /*
            const uint_fast32_t seed =
            chrono::system_clock::now().time_since_epoch().count();
        auto rng =
            bind(uniform_int_distribution<u8>(), mt19937(seed));

        mc_ = state.range(0);
        nc_ = state.range(1);
        kc_ = state.range(2);

        a_.resize(mc() * kc());
        generate(a_.begin(), a_.end(), ref(rng));
        k_.resize(nc() * kc());
        generate(k_.begin(), k_.end(), ref(rng));
        b_.resize(nc());
        generate(b_.begin(), b_.end(), ref(rng));
        c_.resize(mc() * nc());
        fill(c_.begin(), c_.end(), 0xA5);

        threadingContext.set_max_num_threads(1);
        */
    }
    
    pub fn tear_down(&mut self, state: &mut BenchmarkState)  {
        
        todo!();
        /*
            state.SetItemsProcessed(
            u64(state.iterations()) * 2 * mc() * nc() * kc());
        a_.clear();
        k_.clear();
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
    
    #[inline] pub fn c(&mut self) -> *mut u8 {
        
        todo!();
        /*
            return c_.data();
        */
    }
    
    #[inline] pub fn mc(&self) -> u32 {
        
        todo!();
        /*
            return mc_;
        */
    }
    
    #[inline] pub fn nc(&self) -> u32 {
        
        todo!();
        /*
            return nc_;
        */
    }
    
    #[inline] pub fn kc(&self) -> u32 {
        
        todo!();
        /*
            return kc_;
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
            b->ArgNames({"M", "N", "K"});

      for (auto S = 15; S <= 128; S *= 2) {
        for (int K = 8; K <= 1024; K *= 2) {
          b->Args({S * S, K, K});
        }
      }
        */
}

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
pub fn q8gemm_compute_row_sum(
        a:          *const u8,
        m:          usize,
        k:          usize,
        stride:     usize,
        multiplier: i32,
        row_sum:    *mut i32)  {
    
    todo!();
        /*
            const usize block_size = 4;
      for (usize block_start = 0; block_start < m; block_start += block_size) {
        pytorch_q8sumrows_ukernel_4x__neon(
            a + block_start * stride,
            min(block_size, m - block_start),
            k,
            stride,
            multiplier,
            row_sum + block_start);
      }
        */
}

#[cfg(CPUINFO_ARCH_ARM)]
lazy_static!{
    /*
    BENCHMARK_TEMPLATE_F(Q8GEMM_L1, 4x8__aarch32_neon, 4, 8, 8, 1)
    (BenchmarkState& state) {
      for (auto _ : state) {
        pytorch_q8gemm_ukernel_4x8__aarch32_neon(
            mr(),
            nr(),
            kc(),
            a(),
            kc() * sizeof(u8),
            w(),
            c(),
            mr() * sizeof(u8),
            0,
            quantizationParams());
      }
    }
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
    */
}

#[cfg(CPUINFO_ARCH_ARM)]
lazy_static!{
    /*
    BENCHMARK_REGISTER_F(Q8GEMM_Op, 4x8__aarch32_neon)
        ->Apply(ShuffleNetV1G1GemmArguments);
    BENCHMARK_REGISTER_F(Q8GEMM_Op, 4x8__aarch32_neon)
        ->Apply(MobileNetV1GemmArguments);
    BENCHMARK_REGISTER_F(Q8GEMM_Op, 4x8__aarch32_neon)
        ->Apply(SqueezeNetV10GemmArguments);
    BENCHMARK_REGISTER_F(Q8GEMM_Op, 4x8__aarch32_neon)->Apply(GemmArguments);
    */
}

#[cfg(CPUINFO_ARCH_ARM)]
lazy_static!{
    /*
    BENCHMARK_TEMPLATE_F(Q8GEMM_XZP_L1, 4x8c2__aarch32_neon, 4, 8, 8, 2)
    (BenchmarkState& state) {
      for (auto _ : state) {
        q8gemm_compute_row_sum(a(), mr(), kc(), kc(), -64, aRowSums());
        pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon(
            mr(),
            nr(),
            kc(),
            a(),
            kc(),
            aRowSums(),
            w(),
            c(),
            mr(),
            requantizationParams());
      }
    }
    */
}

#[cfg(CPUINFO_ARCH_ARM)]
lazy_static!{
    /*
    BENCHMARK_TEMPLATE_DEFINE_F(Q8GEMM_XZP_Op, 4x8c2__aarch32_neon, 4, 8, 8, 2)
    (BenchmarkState& state) {
      for (auto _ : state) {
        q8gemm_compute_row_sum(a(), mc(), kc(), kc(), -64, aRowSums());
        for (u32 m = 0; m < mc(); m += mr()) {
          const u32 mrr = min(mc() - m, mr());
          for (u32 n = 0; n < nc(); n += nr()) {
            const u32 nrr = min(nc() - n, nr());
            pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon(
                mrr,
                nrr,
                kc(),
                a() + m * kc(),
                kc(),
                aRowSums() + m,
                w() + n * (kcStride() + sizeof(i32) / sizeof(u8)),
                c() + m * nc() + n,
                nc(),
                requantizationParams());
          }
        }
      }
    }
    */
}

#[cfg(CPUINFO_ARCH_ARM)]
lazy_static!{
    /*
    BENCHMARK_REGISTER_F(Q8GEMM_XZP_Op, 4x8c2__aarch32_neon)
        ->Apply(ShuffleNetV1G1GemmArguments);
    BENCHMARK_REGISTER_F(Q8GEMM_XZP_Op, 4x8c2__aarch32_neon)
        ->Apply(MobileNetV1GemmArguments);
    BENCHMARK_REGISTER_F(Q8GEMM_XZP_Op, 4x8c2__aarch32_neon)
        ->Apply(SqueezeNetV10GemmArguments);
    BENCHMARK_REGISTER_F(Q8GEMM_XZP_Op, 4x8c2__aarch32_neon)->Apply(GemmArguments);
    */
}

#[cfg(CPUINFO_ARCH_ARM64)]
lazy_static!{
    /*
    BENCHMARK_TEMPLATE_F(Q8GEMM_L1, 8x8__aarch64_neon, 8, 8, 8, 1)
    (BenchmarkState& state) {
      for (auto _ : state) {
        pytorch_q8gemm_ukernel_8x8__aarch64_neon(
            mr(),
            nr(),
            kc(),
            a(),
            kc() * sizeof(u8),
            w(),
            c(),
            mr() * sizeof(u8),
            0,
            quantizationParams());
      }
    }
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
    */
}

#[cfg(CPUINFO_ARCH_ARM64)]
lazy_static!{
    /*
    BENCHMARK_REGISTER_F(Q8GEMM_Op, 8x8__aarch64_neon)
        ->Apply(ShuffleNetV1G1GemmArguments);
    BENCHMARK_REGISTER_F(Q8GEMM_Op, 8x8__aarch64_neon)
        ->Apply(MobileNetV1GemmArguments);
    BENCHMARK_REGISTER_F(Q8GEMM_Op, 8x8__aarch64_neon)
        ->Apply(SqueezeNetV10GemmArguments);
    BENCHMARK_REGISTER_F(Q8GEMM_Op, 8x8__aarch64_neon)->Apply(GemmArguments);
    */
}

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
lazy_static!{
    /*
    BENCHMARK_TEMPLATE_F(Q8GEMM_L1, 4x8__neon, 4, 8, 8, 1)
    (BenchmarkState& state) {
      for (auto _ : state) {
        pytorch_q8gemm_ukernel_4x8__neon(
            mr(),
            nr(),
            kc(),
            a(),
            kc() * sizeof(u8),
            w(),
            c(),
            mr() * sizeof(u8),
            0,
            quantizationParams());
      }
    }
    */
}

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
lazy_static!{
    /*
    BENCHMARK_TEMPLATE_F(Q8GEMM_L1, 8x8__neon, 8, 8, 8, 1)
    (BenchmarkState& state) {
      for (auto _ : state) {
        pytorch_q8gemm_ukernel_8x8__neon(
            mr(),
            nr(),
            kc(),
            a(),
            kc() * sizeof(u8),
            w(),
            c(),
            mr() * sizeof(u8),
            0,
            quantizationParams());
      }
    }
    */
}

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
lazy_static!{
    /*
    BENCHMARK_TEMPLATE_DEFINE_F(Q8GEMM_Op, 4x8__neon, 4, 8, 8, 1)
    (BenchmarkState& state) {
      for (auto _ : state) {
        for (u32 m = 0; m < mc(); m += mr()) {
          const u32 mrr = min(mc() - m, mr());
          for (u32 n = 0, channel_offset = 0; n < nc();
              n += nr(), channel_offset += nr()) {
            const u32 nrr = min(nc() - n, nr());
            pytorch_q8gemm_ukernel_4x8__neon(
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
    */
}


#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
lazy_static!{
    /*
    BENCHMARK_REGISTER_F(Q8GEMM_Op, 4x8__neon)->Apply(ShuffleNetV1G1GemmArguments);
    BENCHMARK_REGISTER_F(Q8GEMM_Op, 4x8__neon)->Apply(MobileNetV1GemmArguments);
    BENCHMARK_REGISTER_F(Q8GEMM_Op, 4x8__neon)->Apply(SqueezeNetV10GemmArguments);
    BENCHMARK_REGISTER_F(Q8GEMM_Op, 4x8__neon)->Apply(GemmArguments);
    */
}

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
lazy_static!{
    /*
    BENCHMARK_TEMPLATE_DEFINE_F(Q8GEMM_Op, 8x8__neon, 8, 8, 8, 1)
    (BenchmarkState& state) {
      for (auto _ : state) {
        for (u32 m = 0; m < mc(); m += mr()) {
          const u32 mrr = min(mc() - m, mr());
          for (u32 n = 0, channel_offset = 0; n < nc();
              n += nr(), channel_offset += nr()) {
            const u32 nrr = min(nc() - n, nr());
            pytorch_q8gemm_ukernel_8x8__neon(
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
    */
}

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
lazy_static!{
    /*
    BENCHMARK_REGISTER_F(Q8GEMM_Op, 8x8__neon)->Apply(ShuffleNetV1G1GemmArguments);
    BENCHMARK_REGISTER_F(Q8GEMM_Op, 8x8__neon)->Apply(MobileNetV1GemmArguments);
    BENCHMARK_REGISTER_F(Q8GEMM_Op, 8x8__neon)->Apply(SqueezeNetV10GemmArguments);
    BENCHMARK_REGISTER_F(Q8GEMM_Op, 8x8__neon)->Apply(GemmArguments);
    */
}


#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
lazy_static!{
    /*
    BENCHMARK_TEMPLATE_F(Q8GEMM_XZP_L1, 4x8c2_neon, 4, 8, 8, 2)
    (BenchmarkState& state) {
      for (auto _ : state) {
        q8gemm_compute_row_sum(a(), mr(), kc(), kc(), -64, aRowSums());
        pytorch_q8gemm_xzp_ukernel_4x8c2__neon(
            mr(),
            nr(),
            kc(),
            a(),
            kc(),
            aRowSums(),
            w(),
            c(),
            mr(),
            requantizationParams());
      }
    }
    */
}

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
lazy_static!{
    /*
    BENCHMARK_TEMPLATE_DEFINE_F(Q8GEMM_XZP_Op, 4x8c2_neon, 4, 8, 8, 2)
    (BenchmarkState& state) {
      for (auto _ : state) {
        q8gemm_compute_row_sum(a(), mc(), kc(), kc(), -64, aRowSums());
        for (u32 m = 0; m < mc(); m += mr()) {
          const u32 mrr = min(mc() - m, mr());
          for (u32 n = 0; n < nc(); n += nr()) {
            const u32 nrr = min(nc() - n, nr());
            pytorch_q8gemm_xzp_ukernel_4x8c2__neon(
                mrr,
                nrr,
                kc(),
                a() + m * kc(),
                kc(),
                aRowSums() + m,
                w() + n * (kcStride() + sizeof(i32) / sizeof(u8)),
                c() + m * nc() + n,
                nc(),
                requantizationParams());
          }
        }
      }
    }
    */
}

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
lazy_static!{
    /*
    BENCHMARK_REGISTER_F(Q8GEMM_XZP_Op, 4x8c2_neon)
        ->Apply(ShuffleNetV1G1GemmArguments);
    BENCHMARK_REGISTER_F(Q8GEMM_XZP_Op, 4x8c2_neon)
        ->Apply(MobileNetV1GemmArguments);
    BENCHMARK_REGISTER_F(Q8GEMM_XZP_Op, 4x8c2_neon)
        ->Apply(SqueezeNetV10GemmArguments);
    BENCHMARK_REGISTER_F(Q8GEMM_XZP_Op, 4x8c2_neon)->Apply(GemmArguments);
    */
}

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
lazy_static!{
    /*
    BENCHMARK_TEMPLATE_DEFINE_F(
        COMPUTE_ROW_SUM_Op,
        compute_row_sum_neon,
        4,
        8,
        8,
        2)
    (BenchmarkState& state) {
      for (auto _ : state) {
        const usize block_size = 4;
        for (usize block_start = 0; block_start < mc();
             block_start += block_size) {
          pytorch_q8sumrows_ukernel_4x__neon(
              a() + block_start * kc(),
              min(block_size, mc() - block_start),
              kc(),
              kc(),
              0x11,
              aRowSums() + block_start);
        }
      }
    }
    */
}


#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
lazy_static!{
    /*
    BENCHMARK_REGISTER_F(COMPUTE_ROW_SUM_Op, compute_row_sum_neon)
        ->Apply(ShuffleNetV1G1GemmArguments);
    BENCHMARK_REGISTER_F(COMPUTE_ROW_SUM_Op, compute_row_sum_neon)
        ->Apply(MobileNetV1GemmArguments);
    BENCHMARK_REGISTER_F(COMPUTE_ROW_SUM_Op, compute_row_sum_neon)
        ->Apply(SqueezeNetV10GemmArguments);
    BENCHMARK_REGISTER_F(COMPUTE_ROW_SUM_Op, compute_row_sum_neon)
        ->Apply(GemmArguments);
    */
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
lazy_static!{
    /*
    BENCHMARK_TEMPLATE_F(Q8GEMM_L1, 2x4c8__sse2, 2, 4, 1, 8)
    (BenchmarkState& state) {
      for (auto _ : state) {
        pytorch_q8gemm_ukernel_2x4c8__sse2(
            mr(),
            nr(),
            kc(),
            a(),
            kc() * sizeof(u8),
            w(),
            c(),
            mr() * sizeof(u8),
            0,
            quantizationParams());
      }
    }

    BENCHMARK_TEMPLATE_F(Q8GEMM_L1, 4x4c2__sse2, 4, 4, 4, 2)
    (BenchmarkState& state) {
      for (auto _ : state) {
        pytorch_q8gemm_ukernel_4x4c2__sse2(
            mr(),
            nr(),
            kc(),
            a(),
            kc() * sizeof(u8),
            w(),
            c(),
            mr() * sizeof(u8),
            0,
            quantizationParams());
      }
    }

    BENCHMARK_TEMPLATE_DEFINE_F(Q8GEMM_Op, 2x4c8__sse2, 2, 4, 1, 8)
    (BenchmarkState& state) {
      for (auto _ : state) {
        for (u32 m = 0; m < mc(); m += mr()) {
          const u32 mrr = min(mc() - m, mr());
          for (u32 n = 0, channel_offset = 0; n < nc();
              n += nr(), channel_offset += nr()) {
            const u32 nrr = min(nc() - n, nr());
            pytorch_q8gemm_ukernel_2x4c8__sse2(
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

    BENCHMARK_REGISTER_F(Q8GEMM_Op, 2x4c8__sse2)
        ->Apply(ShuffleNetV1G1GemmArguments);
    BENCHMARK_REGISTER_F(Q8GEMM_Op, 2x4c8__sse2)->Apply(MobileNetV1GemmArguments);
    BENCHMARK_REGISTER_F(Q8GEMM_Op, 2x4c8__sse2)->Apply(SqueezeNetV10GemmArguments);
    BENCHMARK_REGISTER_F(Q8GEMM_Op, 2x4c8__sse2)->Apply(GemmArguments);

    BENCHMARK_TEMPLATE_DEFINE_F(Q8GEMM_Op, 4x4c2__sse2, 4, 4, 4, 2)
    (BenchmarkState& state) {
      for (auto _ : state) {
        for (u32 m = 0; m < mc(); m += mr()) {
          const u32 mrr = min(mc() - m, mr());
          for (u32 n = 0, channel_offset = 0; n < nc();
              n += nr(), channel_offset += nr()) {
            const u32 nrr = min(nc() - n, nr());
            pytorch_q8gemm_ukernel_4x4c2__sse2(
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

    BENCHMARK_REGISTER_F(Q8GEMM_Op, 4x4c2__sse2)
        ->Apply(ShuffleNetV1G1GemmArguments);
    BENCHMARK_REGISTER_F(Q8GEMM_Op, 4x4c2__sse2)->Apply(MobileNetV1GemmArguments);
    BENCHMARK_REGISTER_F(Q8GEMM_Op, 4x4c2__sse2)->Apply(SqueezeNetV10GemmArguments);
    BENCHMARK_REGISTER_F(Q8GEMM_Op, 4x4c2__sse2)->Apply(GemmArguments);
    */
}

#[cfg(PYTORCH_QNNPACK_BENCHMARK_GEMMLOWP)]
lazy_static!{
    /*
    BENCHMARK_DEFINE_F(GEMMLOWP, single_threaded)(BenchmarkState& state) {
      for (auto _ : state) {
        gemmlowp::MatrixMap<const u8, gemmlowp::MapOrder::RowMajor> AM(
            a(), mc(), kc(), kc());
        gemmlowp::MatrixMap<const u8, gemmlowp::MapOrder::ColMajor> BM(
            k(), kc(), nc(), kc());
        gemmlowp::MatrixMap<u8, gemmlowp::MapOrder::RowMajor> CM(
            c(), mc(), nc(), nc());
        const auto& output_pipeline =
            GemmlowpOutputPipeline::Make(b(), nc(), 127, 1, 2, 0, 255);
        gemmlowp::GemmWithOutputPipeline<
            u8,
            u8,
            gemmlowp::L8R8WithLhsNonzeroBitDepthParams>(
            &threadingContext, AM, BM, &CM, 2, 1, output_pipeline);
      }
    }

    BENCHMARK_REGISTER_F(GEMMLOWP, single_threaded)
        ->Apply(ShuffleNetV1G1GemmArguments);
    BENCHMARK_REGISTER_F(GEMMLOWP, single_threaded)
        ->Apply(MobileNetV1GemmArguments);
    BENCHMARK_REGISTER_F(GEMMLOWP, single_threaded)
        ->Apply(SqueezeNetV10GemmArguments);
    BENCHMARK_REGISTER_F(GEMMLOWP, single_threaded)->Apply(GemmArguments);
    */
}

#[cfg(not(PYTORCH_QNNPACK_BENCHMARK_NO_MAIN))]
lazy_static!{
    /*
    BENCHMARK_MAIN();
    */
}
