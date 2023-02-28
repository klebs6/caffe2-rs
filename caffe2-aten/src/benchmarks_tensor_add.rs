crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/benchmarks/tensor_add.cpp]

pub fn tensor_add(state: &mut BenchmarkState)  {
    
    todo!();
        /*
            const usize batchSize = static_cast<usize>(state.range(0));
      const usize channels = static_cast<usize>(state.range(1));

      at::Tensor a = at::rand({batchSize, channels});
      at::Tensor b = at::rand({batchSize, channels});
      at::Tensor c;
      for (auto _ : state) {
        c = a + b;
      }
        */
}

pub fn generate_sizes(b: *mut Benchmark)  {
    
    todo!();
        /*
            b->ArgNames({"N", "C"});

      for (usize n = 8; n < 1024;) {
        for (usize c = 8; c < 1024;) {
          b->Args({n, c});
          c *= 2;
        }
        n *= 2;
      }
        */
}

lazy_static!{
    /*
    BENCHMARK(tensor_add)->Apply(GenerateSizes);
    BENCHMARK_MAIN();
    */
}
