crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/bench/global-average-pooling.cc]

pub fn global_average_pooling_q8(state: &mut BenchmarkState)  {
    
    todo!();
        /*
            const usize batchSize = state.range(0);
      const usize inputHeight = state.range(1);
      const usize inputWidth = state.range(2);
      const usize channels = state.range(3);

      random_device randomDevice;
      auto rng = mt19937(randomDevice());
      auto u8rng = bind(uniform_int_distribution<u8>(), rng);

      const usize inputPixelStride = channels;
      const usize outputPixelStride = channels;

      vector<u8> input(
          batchSize * inputHeight * inputWidth * inputPixelStride);
      generate(input.begin(), input.end(), ref(u8rng));
      vector<u8> output(batchSize * outputPixelStride);

      pytorch_qnnp_status status = pytorch_qnnp_initialize();
      if (status != pytorch_qnnp_status_success) {
        state.SkipWithError("failed to initialize QNNPACK");
      }

      pytorch_qnnp_operator_t globalPoolingOperator = nullptr;
      status = pytorch_qnnp_create_global_average_pooling_nwc_q8(
          channels,
          127 /* input zero point */,
          0.75f /* input scale */,
          127 /* output zero point */,
          1.25f /* output scale */,
          0,
          255,
          0 /* flags */,
          &globalPoolingOperator);
      if (status != pytorch_qnnp_status_success) {
        state.SkipWithError("failed to create Global Average Pooling operator");
      }

      status = pytorch_qnnp_setup_global_average_pooling_nwc_q8(
          globalPoolingOperator,
          batchSize,
          inputHeight * inputWidth,
          input.data(),
          inputPixelStride,
          output.data(),
          outputPixelStride);
      if (status != pytorch_qnnp_status_success) {
        state.SkipWithError("failed to setup Global Average Pooling operator");
      }

      for (auto _ : state) {
        pytorch_qnnp_run_operator(globalPoolingOperator, nullptr /* thread pool */);
      }

      status = pytorch_qnnp_delete_operator(globalPoolingOperator);
      if (status != pytorch_qnnp_status_success) {
        state.SkipWithError("failed to delete Global Average Pooling operator");
      }
      globalPoolingOperator = nullptr;

      state.SetBytesProcessed(
          u64(state.iterations()) * batchSize *
          (inputHeight * inputWidth + 1) * channels * sizeof(u8));
        */
}

pub fn image_net_arguments(b: *mut Benchmark)  {
    
    todo!();
        /*
            b->ArgNames({"N", "H", "W", "C"});

      /*       N  IH  IW    C */
      b->Args({1, 7, 7, 1000});
      b->Args({1, 13, 13, 1000});
        */
}

lazy_static!{
    /*
    BENCHMARK(global_average_pooling_q8)->Apply(ImageNetArguments);

    #ifndef PYTORCH_QNNPACK_BENCHMARK_NO_MAIN
    BENCHMARK_MAIN();
    #endif
    */
}

