crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/bench/sigmoid.cc]

pub fn sigmoid_q8(state: &mut BenchmarkState)  {
    
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

      pytorch_qnnp_operator_t sigmoidOperator = nullptr;
      status = pytorch_qnnp_create_sigmoid_nc_q8(
          channels,
          127 /* input zero point */,
          1.0f /* input scale */,
          0 /* output zero point */,
          1.0f / 256.0f /* output scale */,
          0 /* output min */,
          255 /* output max */,
          0 /* flags */,
          &sigmoidOperator);
      if (status != pytorch_qnnp_status_success || sigmoidOperator == nullptr) {
        state.SkipWithError("failed to create Sigmoid operator");
      }

      status = pytorch_qnnp_setup_sigmoid_nc_q8(
          sigmoidOperator,
          batchSize,
          input.data(),
          channels /* input:stride */,
          output.data(),
          channels /* output:stride */);
      if (status != pytorch_qnnp_status_success) {
        state.SkipWithError("failed to setup Sigmoid operator");
      }

      for (auto _ : state) {
        status =
            pytorch_qnnp_run_operator(sigmoidOperator, nullptr /* thread pool */);
        if (status != pytorch_qnnp_status_success) {
          state.SkipWithError("failed to run Sigmoid operator");
        }
      }

      const usize itemsPerIteration = batchSize * channels;
      state.SetItemsProcessed(
          i64(state.iterations()) * i64(itemsPerIteration));

      const usize bytesPerIteration = 2 * itemsPerIteration * sizeof(u8);
      state.SetBytesProcessed(
          i64(state.iterations()) * i64(bytesPerIteration));

      status = pytorch_qnnp_delete_operator(sigmoidOperator);
      if (status != pytorch_qnnp_status_success) {
        state.SkipWithError("failed to delete Sigmoid operator");
      }
        */
}


pub fn characteristic_arguments(b: *mut Benchmark)  {
    
    todo!();
        /*
            b->ArgNames({"N", "C"});

      i32 c = 16;
      for (i32 n = 224; n >= 7; n /= 2) {
        b->Args({n * n, c});
        c *= 2;
      }
        */
}

lazy_static!{
    /*
    BENCHMARK(sigmoid_q8)->Apply(CharacteristicArguments);
    */
}

#[cfg(not(PYTORCH_QNNPACK_BENCHMARK_NO_MAIN))]
lazy_static!{
    /*
    BENCHMARK_MAIN();
    */
}
