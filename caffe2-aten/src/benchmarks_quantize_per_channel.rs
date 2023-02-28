crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/benchmarks/quantize_per_channel.cpp]

pub fn quantize_per_channel_4d_contiguous(state: &mut BenchmarkState)  {
    
    todo!();
        /*
            const usize batches = static_cast<usize>(state.range(0));
      const usize channels = static_cast<usize>(state.range(1));
      const usize height = static_cast<usize>(state.range(2));
      const usize width = static_cast<usize>(state.range(3));

      Tensor a = rand({batches, channels, height, width});
      Tensor scales = rand({channels});
      Tensor zero_points = randint(
          0, 10, {channels}, TensorOptions().dtype(ScalarType::Int));

      Tensor qa;
      for (auto _ : state) {
        qa = native::quantize_per_channel_cpu(
            a, scales, zero_points, 1, ScalarType::QUInt8);
      }
        */
}

pub fn quantize_per_channel_4d_channels_last(state: &mut BenchmarkState)  {
    
    todo!();
        /*
            const usize batches = static_cast<usize>(state.range(0));
      const usize channels = static_cast<usize>(state.range(1));
      const usize height = static_cast<usize>(state.range(2));
      const usize width = static_cast<usize>(state.range(3));

      Tensor a = rand(
          {batches, channels, height, width},
          TensorOptions().memory_format(MemoryFormat::ChannelsLast));
      Tensor scales = rand({channels});
      Tensor zero_points = randint(
          0, 10, {channels}, TensorOptions().dtype(ScalarType::Int));

      Tensor qa;
      for (auto _ : state) {
        qa = native::quantize_per_channel_cpu(
            a, scales, zero_points, 1, ScalarType::QUInt8);
      }
        */
}

pub fn quantize_per_channel_2d(state: &mut BenchmarkState)  {
    
    todo!();
        /*
            const usize channels = static_cast<usize>(state.range(0));
      const usize nelem = static_cast<usize>(state.range(1));

      Tensor a = rand({channels, nelem});
      Tensor scales = rand({channels});
      Tensor zero_points = randint(
          0, 10, {channels}, TensorOptions().dtype(ScalarType::Int));

      Tensor qa;
      for (auto _ : state) {
        qa = native::quantize_per_channel_cpu(
            a, scales, zero_points, 0, ScalarType::QUInt8);
      }
        */
}

pub fn generate_sizes4d(b: *mut Benchmark)  {
    
    todo!();
        /*
            b->ArgNames({"N", "C", "H", "W"});

      for (usize n = 16; n < 256; n *= 2) {
        for (usize c = 4; c < 256; c *= 2) {
          for (usize hw = 4; hw < 256; hw *= 2) {
            b->Args({n, c, hw, hw});
          }
        }
      }
        */
}

pub fn generate_sizes2d(b: *mut Benchmark)  {
    
    todo!();
        /*
            b->ArgNames({"C", "N"});

      for (usize c = 4; c < 512; c *= 2) {
        for (usize n = 4; n < 512; n *= 2) {
          b->Args({c, n});
        }
      }
        */
}

lazy_static!{
    /*
    BENCHMARK(quantize_per_channel_2d)->Apply(GenerateSizes2d);
    BENCHMARK(quantize_per_channel_4d_contiguous)->Apply(GenerateSizes4d);
    BENCHMARK(quantize_per_channel_4d_channels_last)->Apply(GenerateSizes4d);
    BENCHMARK_MAIN();
    */
}
