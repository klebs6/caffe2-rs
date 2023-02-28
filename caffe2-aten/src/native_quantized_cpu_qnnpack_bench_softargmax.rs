crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/bench/softargmax.cc]

pub fn softargmax_q8(state: &mut BenchmarkState)  {
    
    todo!();
        /*
            const usize batchSize = static_cast<usize>(state.range(0));
      const usize channels = static_cast<usize>(state.range(1));

      random_device randomDevice;
      auto rng = mt19937(randomDevice());
      auto u8rng = bind(uniform_int_distribution<u8>(), rng);

      vector<u8> input(batchSize * channels);
      vector<u8> output(batchSize * channels);
      generate(input.begin(), input.end(), ref(u8rng));
      fill(output.begin(), output.end(), 0xA5);

      pytorch_qnnp_status status = pytorch_qnnp_initialize();
      if (status != pytorch_qnnp_status_success) {
        state.SkipWithError("failed to initialize QNNPACK");
      }

      pytorch_qnnp_operator_t softArgMaxOperator = nullptr;
      status = pytorch_qnnp_create_softargmax_nc_q8(
          channels,
          1.0f /* input scale */,
          0 /* output zero point */,
          1.0f / 256.0f /* output scale */,
          0 /* flags */,
          &softArgMaxOperator);
      if (status != pytorch_qnnp_status_success || softArgMaxOperator == nullptr) {
        state.SkipWithError("failed to create SoftArgMax operator");
      }

      status = pytorch_qnnp_setup_softargmax_nc_q8(
          softArgMaxOperator,
          batchSize,
          input.data(),
          channels /* input:stride */,
          output.data(),
          channels /* output:stride */);
      if (status != pytorch_qnnp_status_success) {
        state.SkipWithError("failed to setup SoftArgMax operator");
      }

      for (auto _ : state) {
        status = pytorch_qnnp_run_operator(
            softArgMaxOperator, nullptr /* thread pool */);
        if (status != pytorch_qnnp_status_success) {
          state.SkipWithError("failed to run SoftArgMax operator");
        }
      }

      const usize itemsPerIteration = batchSize * channels;
      state.SetItemsProcessed(
          i64(state.iterations()) * i64(itemsPerIteration));

      const usize bytesPerIteration = 2 * itemsPerIteration * sizeof(u8);
      state.SetBytesProcessed(
          i64(state.iterations()) * i64(bytesPerIteration));

      status = pytorch_qnnp_delete_operator(softArgMaxOperator);
      if (status != pytorch_qnnp_status_success) {
        state.SkipWithError("failed to delete SoftArgMax operator");
      }
        */
}


pub fn characteristic_arguments(b: *mut Benchmark)  {
    
    todo!();
        /*
            b->ArgNames({"N", "C"});

      /* CIFAR-10 */
      b->Args({1, 10});
      /* CIFAR-100 */
      b->Args({1, 100});
      /* ImageNet-1K */
      b->Args({1, 1000});
      /* ImageNet-1K+1 */
      b->Args({1, 1001});
      /* ImageNet-22K */
      b->Args({1, 21841});
        */
}

lazy_static!{
    /*
    BENCHMARK(softargmax_q8)->Apply(CharacteristicArguments);
    */
}

#[cfg(not(PYTORCH_QNNPACK_BENCHMARK_NO_MAIN))]
lazy_static!{
    /*
    BENCHMARK_MAIN();
    */
}
