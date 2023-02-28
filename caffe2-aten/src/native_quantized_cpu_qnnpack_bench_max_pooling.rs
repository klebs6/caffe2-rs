crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/bench/max-pooling.cc]

pub fn max_pooling_u8(
        state: &mut BenchmarkState,
        net:   *const u8)  {
    
    todo!();
        /*
            const usize batchSize = state.range(0);
      const usize inputHeight = state.range(1);
      const usize inputWidth = state.range(2);
      const usize poolingSize = state.range(3);
      const usize paddingSize = state.range(4);
      const usize stride = state.range(5);
      const usize channels = state.range(6);

      std::random_device randomDevice;
      auto rng = std::mt19937(randomDevice());
      auto u8rng = std::bind(std::uniform_int_distribution<u8>(), rng);

      const usize inputPixelStride = channels;
      const usize outputPixelStride = channels;
      const usize outputHeight =
          (2 * paddingSize + inputHeight - poolingSize) / stride + 1;
      const usize outputWidth =
          (2 * paddingSize + inputWidth - poolingSize) / stride + 1;

      std::vector<u8> input(
          batchSize * inputHeight * inputWidth * inputPixelStride);
      std::generate(input.begin(), input.end(), std::ref(u8rng));
      std::vector<u8> output(
          batchSize * outputHeight * outputWidth * outputPixelStride);
      std::fill(output.begin(), output.end(), 0xA5);

      pytorch_qnnp_status status = pytorch_qnnp_initialize();
      if (status != pytorch_qnnp_status_success) {
        state.SkipWithError("failed to initialize QNNPACK");
      }

      pytorch_qnnp_operator_t poolingOperator = nullptr;
      status = pytorch_qnnp_create_max_pooling2d_nhwc_u8(
          paddingSize,
          paddingSize,
          paddingSize,
          paddingSize,
          poolingSize,
          poolingSize,
          stride,
          stride,
          1 /* dilation height */,
          1 /* dilation width */,
          channels,
          0,
          255,
          0 /* flags */,
          &poolingOperator);
      if (status != pytorch_qnnp_status_success) {
        state.SkipWithError("failed to create Max Pooling operator");
      }

      status = pytorch_qnnp_setup_max_pooling2d_nhwc_u8(
          poolingOperator,
          batchSize,
          inputHeight,
          inputWidth,
          input.data(),
          inputPixelStride,
          output.data(),
          outputPixelStride,
          nullptr /* thread pool */);
      if (status != pytorch_qnnp_status_success) {
        state.SkipWithError("failed to setup Max Pooling operator");
      }

      for (auto _ : state) {
        status =
            pytorch_qnnp_run_operator(poolingOperator, nullptr /* thread pool */);
        if (status != pytorch_qnnp_status_success) {
          state.SkipWithError("failed to run Max Pooling operator");
        }
      }

      status = pytorch_qnnp_delete_operator(poolingOperator);
      if (status != pytorch_qnnp_status_success) {
        state.SkipWithError("failed to delete Max Pooling operator");
      }
      poolingOperator = nullptr;

      state.SetBytesProcessed(
          u64(state.iterations()) * batchSize *
          (inputHeight * inputWidth + outputHeight * outputWidth) * channels *
          sizeof(u8));
        */
}

/**
  | ShuffleNet v1/v2
  |
  */
pub fn shuffle_net(b: *mut Benchmark)  {
    
    todo!();
        /*
            b->ArgNames({"N", "H", "W", "K", "P", "S", "C"});

      /*       N   H   W    K  P  S   C */
      b->Args({1, 112, 112, 3, 1, 2, 24});
        */
}

/**
  | SqueezeNet 1.0
  |
  */
pub fn squeeze_netv10(b: *mut Benchmark)  {
    
    todo!();
        /*
            b->ArgNames({"N", "H", "W", "K", "P", "S", "C"});

      /*********** MaxPool 1 ************/
      /*       N   H    W   K  P  S   C */
      b->Args({1, 111, 111, 3, 0, 2, 96});
      /*********** MaxPool 4 ************/
      /*       N   H    W   K  P  S   C */
      b->Args({1, 27, 27, 3, 0, 2, 256});
      /*********** MaxPool 8 ************/
      /*       N   H    W   K  P  S   C */
      b->Args({1, 13, 13, 3, 0, 2, 512});
        */
}

/**
  | SqueezeNet 1.1
  |
  */
pub fn squeeze_netv11(b: *mut Benchmark)  {
    
    todo!();
        /*
            b->ArgNames({"N", "H", "W", "K", "P", "S", "C"});

      /*********** MaxPool 1 ***********/
      /*       N   H    W   K  P  S   C */
      b->Args({1, 111, 111, 3, 0, 2, 64});
      /*********** MaxPool 3 ************/
      /*       N   H    W   K  P  S   C */
      b->Args({1, 55, 55, 3, 0, 2, 128});
      /*********** MaxPool 5 ************/
      /*       N   H    W   K  P  S   C */
      b->Args({1, 13, 13, 3, 0, 2, 256});
        */
}

pub fn VGG(b: *mut Benchmark)  {
    
    todo!();
        /*
            b->ArgNames({"N", "H", "W", "K", "P", "S", "C"});

      /*       N   H    W   K  P  S   C */
      b->Args({1, 224, 224, 2, 1, 2, 64});
      b->Args({1, 112, 112, 2, 1, 2, 128});
      b->Args({1, 56, 56, 2, 1, 2, 256});
      b->Args({1, 28, 28, 2, 1, 2, 512});
      b->Args({1, 14, 14, 2, 1, 2, 512});
        */
}

lazy_static!{
    /*
    BENCHMARK_CAPTURE(max_pooling_u8, shufflenet, "ShuffleNet v1/v2")
        ->Apply(ShuffleNet);
    BENCHMARK_CAPTURE(max_pooling_u8, squeezenet_v10, "SqueezeNet v1.0")
        ->Apply(SqueezeNetV10);
    BENCHMARK_CAPTURE(max_pooling_u8, squeezenet_v11, "SqueezeNet v1.1")
        ->Apply(SqueezeNetV11);
    BENCHMARK_CAPTURE(max_pooling_u8, vgg, "VGG")->Apply(VGG);

    #ifndef PYTORCH_QNNPACK_BENCHMARK_NO_MAIN
    BENCHMARK_MAIN();
    #endif
    */
}

