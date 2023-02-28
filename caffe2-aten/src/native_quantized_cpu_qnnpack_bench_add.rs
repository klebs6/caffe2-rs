crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/bench/add.cc]

pub fn add_nc_q8(state: &mut BenchmarkState)  {
    
    todo!();
        /*
            const usize batchSize = static_cast<usize>(state.range(0));
      const usize channels = static_cast<usize>(state.range(1));

      random_device randomDevice;
      auto rng = mt19937(randomDevice());
      auto u8rng = bind(uniform_int_distribution<u8>(), rng);

      vector<u8> a(batchSize * channels);
      vector<u8> b(batchSize * channels);
      vector<u8> y(batchSize * channels);
      generate(a.begin(), a.end(), ref(u8rng));
      generate(b.begin(), b.end(), ref(u8rng));

      pytorch_qnnp_status status = pytorch_qnnp_initialize();
      if (status != pytorch_qnnp_status_success) {
        state.SkipWithError("failed to initialize QNNPACK");
      }

      pytorch_qnnp_operator_t addOperator = nullptr;
      status = pytorch_qnnp_create_add_nc_q8(
          channels,
          127 /* a:zero point */,
          1.0f /* a:scale */,
          127 /* b:zero point */,
          1.0f /* b:scale */,
          127 /* y:zero point */,
          1.0f /* y:scale */,
          1 /* y:min */,
          254 /* y:max */,
          0 /* flags */,
          &addOperator);
      if (status != pytorch_qnnp_status_success || addOperator == nullptr) {
        state.SkipWithError("failed to create Q8 Add operator");
      }

      status = pytorch_qnnp_setup_add_nc_q8(
          addOperator,
          batchSize,
          a.data(),
          channels /* a:stride */,
          b.data(),
          channels /* b:stride */,
          y.data(),
          channels /* y:stride */);
      if (status != pytorch_qnnp_status_success) {
        state.SkipWithError("failed to setup Q8 Add operator");
      }

      for (auto _ : state) {
        status = pytorch_qnnp_run_operator(addOperator, nullptr /* thread pool */);
        if (status != pytorch_qnnp_status_success) {
          state.SkipWithError("failed to run Q8 Add operator");
        }
      }

      const usize itemsPerIteration = batchSize * channels;
      state.SetItemsProcessed(
          i64(state.iterations()) * i64(itemsPerIteration));

      const usize bytesPerIteration = 3 * itemsPerIteration * sizeof(u8);
      state.SetBytesProcessed(
          i64(state.iterations()) * i64(bytesPerIteration));

      status = pytorch_qnnp_delete_operator(addOperator);
      if (status != pytorch_qnnp_status_success) {
        state.SkipWithError("failed to delete Q8 Add operator");
      }
        */
}

pub fn add_nc_q8_inplace(state: &mut BenchmarkState)  {
    
    todo!();
        /*
            const usize batchSize = static_cast<usize>(state.range(0));
      const usize channels = static_cast<usize>(state.range(1));

      random_device randomDevice;
      auto rng = mt19937(randomDevice());
      auto u8rng = bind(uniform_int_distribution<u8>(), rng);

      vector<u8> a(batchSize * channels);
      vector<u8> y(batchSize * channels);
      generate(a.begin(), a.end(), ref(u8rng));

      pytorch_qnnp_status status = pytorch_qnnp_initialize();
      if (status != pytorch_qnnp_status_success) {
        state.SkipWithError("failed to initialize QNNPACK");
      }

      pytorch_qnnp_operator_t addOperator = nullptr;
      status = pytorch_qnnp_create_add_nc_q8(
          channels,
          127 /* a:zero point */,
          1.0f /* a:scale */,
          127 /* b:zero point */,
          1.0f /* b:scale */,
          127 /* y:zero point */,
          1.0f /* y:scale */,
          1 /* y:min */,
          254 /* y:max */,
          0 /* flags */,
          &addOperator);
      if (status != pytorch_qnnp_status_success || addOperator == nullptr) {
        state.SkipWithError("failed to create Q8 Add operator");
      }

      status = pytorch_qnnp_setup_add_nc_q8(
          addOperator,
          batchSize,
          a.data(),
          channels /* a:stride */,
          y.data(),
          channels /* b:stride */,
          y.data(),
          channels /* y:stride */);
      if (status != pytorch_qnnp_status_success) {
        state.SkipWithError("failed to setup Q8 Add operator");
      }

      for (auto _ : state) {
        status = pytorch_qnnp_run_operator(addOperator, nullptr /* thread pool */);
        if (status != pytorch_qnnp_status_success) {
          state.SkipWithError("failed to run Q8 Add operator");
        }
      }

      const usize itemsPerIteration = batchSize * channels;
      state.SetItemsProcessed(
          i64(state.iterations()) * i64(itemsPerIteration));

      const usize bytesPerIteration = 3 * itemsPerIteration * sizeof(u8);
      state.SetBytesProcessed(
          i64(state.iterations()) * i64(bytesPerIteration));

      status = pytorch_qnnp_delete_operator(addOperator);
      if (status != pytorch_qnnp_status_success) {
        state.SkipWithError("failed to delete Q8 Add operator");
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
    BENCHMARK(add_nc_q8)->Apply(CharacteristicArguments);
    BENCHMARK(add_nc_q8_inplace)->Apply(CharacteristicArguments);

    #ifndef PYTORCH_QNNPACK_BENCHMARK_NO_MAIN
    BENCHMARK_MAIN();
    #endif
    */
}
