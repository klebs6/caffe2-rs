crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/bench/requantization.cc]

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

#[inline] pub fn min(a: u32, b: u32) -> u32 {
    
    todo!();
        /*
            return a < b ? a : b;
        */
}

pub struct Requantization {
    base:   BenchmarkFixture,
    input:  Vec<i32,AlignedAllocator<i32,32>>,
    output: Vec<u8>,
    n:      usize,
}

impl Requantization {
    
    pub fn new() -> Self {
    
        todo!();
        /*


            cpuinfo_initialize();
        const usize l1d_size = cpuinfo_get_l1d_cache(0)->size;
        const usize l1d_reserve = 1024;
        n_ = (l1d_size - l1d_reserve) / (sizeof(i32) + sizeof(u8));
        n_ = n_ / 16 * 16;
        */
    }
    
    pub fn set_up(&mut self, _0: &BenchmarkState)  {
        
        todo!();
        /*
            const uint_fast32_t seed =
            chrono::system_clock::now().time_since_epoch().count();
        auto rng =
            bind(uniform_int_distribution<i32>(), mt19937(seed));

        input_.resize(n());
        generate(input_.begin(), input_.end(), ref(rng));
        output_.resize(n());
        fill(output_.begin(), output_.end(), 0xA5);
        */
    }
    
    pub fn tear_down(&mut self, state: &mut BenchmarkState)  {
        
        todo!();
        /*
            state.SetItemsProcessed(u64(state.iterations()) * n());
        state.SetBytesProcessed(
            u64(state.iterations()) * n() *
            (sizeof(i32) + sizeof(u8)));
        input_.clear();
        output_.clear();
        */
    }
    
    #[inline] pub fn input(&self) -> *const i32 {
        
        todo!();
        /*
            return input_.data();
        */
    }
    
    #[inline] pub fn output(&mut self) -> *mut u8 {
        
        todo!();
        /*
            return output_.data();
        */
    }
    
    #[inline] pub fn n(&self) -> usize {
        
        todo!();
        /*
            return n_;
        */
    }
}

lazy_static!{
    /*
    BENCHMARK_F(Requantization, precise__scalar_unsigned32)
    (BenchmarkState& state) {
      for (auto _ : state) {
        pytorch_qnnp_requantize_precise__scalar_unsigned32(
            n(),
            input(),
            0x1.0p-12f /* scale */,
            128 /* zero point */,
            1 /* qmin */,
            254 /* qmax */,
            output());
      }
    }

    BENCHMARK_F(Requantization, precise__scalar_unsigned64)
    (BenchmarkState& state) {
      for (auto _ : state) {
        pytorch_qnnp_requantize_precise__scalar_unsigned64(
            n(),
            input(),
            0x1.0p-12f /* scale */,
            128 /* zero point */,
            1 /* qmin */,
            254 /* qmax */,
            output());
      }
    }

    BENCHMARK_F(Requantization, precise__scalar_signed64)(BenchmarkState& state) {
      for (auto _ : state) {
        pytorch_qnnp_requantize_precise__scalar_signed64(
            n(),
            input(),
            0x1.0p-12f /* scale */,
            128 /* zero point */,
            1 /* qmin */,
            254 /* qmax */,
            output());
      }
    }

    BENCHMARK_F(Requantization, fp32__scalar_lrintf)(BenchmarkState& state) {
      for (auto _ : state) {
        pytorch_qnnp_requantize_fp32__scalar_lrintf(
            n(),
            input(),
            0x1.0p-12f /* scale */,
            128 /* zero point */,
            1 /* qmin */,
            254 /* qmax */,
            output());
      }
    }

    BENCHMARK_F(Requantization, fp32__scalar_magic)(BenchmarkState& state) {
      for (auto _ : state) {
        pytorch_qnnp_requantize_fp32__scalar_magic(
            n(),
            input(),
            0x1.0p-12f /* scale */,
            128 /* zero point */,
            1 /* qmin */,
            254 /* qmax */,
            output());
      }
    }

    BENCHMARK_F(Requantization, gemmlowp__scalar)(BenchmarkState& state) {
      for (auto _ : state) {
        pytorch_qnnp_requantize_gemmlowp__scalar(
            n(),
            input(),
            0x1.0p-12f /* scale */,
            128 /* zero point */,
            1 /* qmin */,
            254 /* qmax */,
            output());
      }
    }

    BENCHMARK_F(Requantization, precise__psimd)(BenchmarkState& state) {
      for (auto _ : state) {
        pytorch_qnnp_requantize_precise__psimd(
            n(),
            input(),
            0x1.0p-12f /* scale */,
            128 /* zero point */,
            1 /* qmin */,
            254 /* qmax */,
            output());
      }
    }

    BENCHMARK_F(Requantization, fp32__psimd)(BenchmarkState& state) {
      for (auto _ : state) {
        pytorch_qnnp_requantize_fp32__psimd(
            n(),
            input(),
            0x1.0p-12f /* scale */,
            128 /* zero point */,
            1 /* qmin */,
            254 /* qmax */,
            output());
      }
    }
    */
}

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
lazy_static!{
    /*
    BENCHMARK_F(Requantization, precise__neon)(BenchmarkState& state) {
      for (auto _ : state) {
        pytorch_qnnp_requantize_precise__neon(
            n(),
            input(),
            0x1.0p-12f /* scale */,
            128 /* zero point */,
            1 /* qmin */,
            254 /* qmax */,
            output());
      }
    }

    BENCHMARK_F(Requantization, fp32__neon)(BenchmarkState& state) {
      for (auto _ : state) {
        pytorch_qnnp_requantize_fp32__neon(
            n(),
            input(),
            0x1.0p-12f /* scale */,
            128 /* zero point */,
            1 /* qmin */,
            254 /* qmax */,
            output());
      }
    }

    BENCHMARK_F(Requantization, q31__neon)(BenchmarkState& state) {
      for (auto _ : state) {
        pytorch_qnnp_requantize_q31__neon(
            n(),
            input(),
            0x1.0p-12f /* scale */,
            128 /* zero point */,
            1 /* qmin */,
            254 /* qmax */,
            output());
      }
    }

    BENCHMARK_F(Requantization, gemmlowp__neon)(BenchmarkState& state) {
      for (auto _ : state) {
        pytorch_qnnp_requantize_gemmlowp__neon(
            n(),
            input(),
            0x1.0p-12f /* scale */,
            128 /* zero point */,
            1 /* qmin */,
            254 /* qmax */,
            output());
      }
    }
    */
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
lazy_static!{
    /*
    BENCHMARK_F(Requantization, precise__sse2)(BenchmarkState& state) {
      for (auto _ : state) {
        pytorch_qnnp_requantize_precise__sse2(
            n(),
            input(),
            0x1.0p-12f /* scale */,
            128 /* zero point */,
            1 /* qmin */,
            254 /* qmax */,
            output());
      }
    }

    BENCHMARK_F(Requantization, precise__ssse3)(BenchmarkState& state) {
      for (auto _ : state) {
        pytorch_qnnp_requantize_precise__ssse3(
            n(),
            input(),
            0x1.0p-12f /* scale */,
            128 /* zero point */,
            1 /* qmin */,
            254 /* qmax */,
            output());
      }
    }

    BENCHMARK_F(Requantization, precise__sse4)(BenchmarkState& state) {
      for (auto _ : state) {
        pytorch_qnnp_requantize_precise__sse4(
            n(),
            input(),
            0x1.0p-12f /* scale */,
            128 /* zero point */,
            1 /* qmin */,
            254 /* qmax */,
            output());
      }
    }

    BENCHMARK_F(Requantization, fp32__sse2)(BenchmarkState& state) {
      for (auto _ : state) {
        pytorch_qnnp_requantize_fp32__sse2(
            n(),
            input(),
            0x1.0p-12f /* scale */,
            128 /* zero point */,
            1 /* qmin */,
            254 /* qmax */,
            output());
      }
    }

    BENCHMARK_F(Requantization, q31__sse2)(BenchmarkState& state) {
      for (auto _ : state) {
        pytorch_qnnp_requantize_q31__sse2(
            n(),
            input(),
            0x1.0p-12f /* scale */,
            128 /* zero point */,
            1 /* qmin */,
            254 /* qmax */,
            output());
      }
    }

    BENCHMARK_F(Requantization, q31__ssse3)(BenchmarkState& state) {
      for (auto _ : state) {
        pytorch_qnnp_requantize_q31__ssse3(
            n(),
            input(),
            0x1.0p-12f /* scale */,
            128 /* zero point */,
            1 /* qmin */,
            254 /* qmax */,
            output());
      }
    }

    BENCHMARK_F(Requantization, q31__sse4)(BenchmarkState& state) {
      for (auto _ : state) {
        pytorch_qnnp_requantize_q31__sse4(
            n(),
            input(),
            0x1.0p-12f /* scale */,
            128 /* zero point */,
            1 /* qmin */,
            254 /* qmax */,
            output());
      }
    }

    BENCHMARK_F(Requantization, gemmlowp__sse2)(BenchmarkState& state) {
      for (auto _ : state) {
        pytorch_qnnp_requantize_gemmlowp__sse2(
            n(),
            input(),
            0x1.0p-12f /* scale */,
            128 /* zero point */,
            1 /* qmin */,
            254 /* qmax */,
            output());
      }
    }

    BENCHMARK_F(Requantization, gemmlowp__ssse3)(BenchmarkState& state) {
      for (auto _ : state) {
        pytorch_qnnp_requantize_gemmlowp__ssse3(
            n(),
            input(),
            0x1.0p-12f /* scale */,
            128 /* zero point */,
            1 /* qmin */,
            254 /* qmax */,
            output());
      }
    }

    BENCHMARK_F(Requantization, gemmlowp__sse4)(BenchmarkState& state) {
      for (auto _ : state) {
        pytorch_qnnp_requantize_gemmlowp__sse4(
            n(),
            input(),
            0x1.0p-12f /* scale */,
            128 /* zero point */,
            1 /* qmin */,
            254 /* qmax */,
            output());
      }
    }
    */
}

#[cfg(not(PYTORCH_QNNPACK_BENCHMARK_NO_MAIN))]
lazy_static!{
    /*
    BENCHMARK_MAIN();
    */
}
