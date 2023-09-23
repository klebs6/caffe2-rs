crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/bench/channel-shuffle.cc]

pub fn channel_shuffle_x8(
        state: &mut BenchmarkState,
        net:   *const u8)  {
    
    todo!();
        /*
            const usize batchSize = static_cast<usize>(state.range(0));
      const usize groups = static_cast<usize>(state.range(1));
      const usize groupChannels = static_cast<usize>(state.range(2));

      random_device randomDevice;
      auto rng = mt19937(randomDevice());
      auto u8rng = bind(uniform_int_distribution<u8>(), rng);

      vector<u8> input(batchSize * groups * groupChannels);
      vector<u8> output(batchSize * groups * groupChannels);
      generate(input.begin(), input.end(), ref(u8rng));

      pytorch_qnnp_status status = pytorch_qnnp_initialize();
      if (status != pytorch_qnnp_status_success) {
        state.SkipWithError("failed to initialize QNNPACK");
      }

      pytorch_qnnp_operator_t channelShuffleOperator = nullptr;
      status = pytorch_qnnp_create_channel_shuffle_nc_x8(
          groups, groupChannels, 0 /* flags */, &channelShuffleOperator);
      if (status != pytorch_qnnp_status_success ||
          channelShuffleOperator == nullptr) {
        state.SkipWithError("failed to create X8 Channel Shuffle operator");
      }

      status = pytorch_qnnp_setup_channel_shuffle_nc_x8(
          channelShuffleOperator,
          batchSize,
          input.data(),
          groups * groupChannels /* input:stride */,
          output.data(),
          groups * groupChannels /* output:stride */);
      if (status != pytorch_qnnp_status_success) {
        state.SkipWithError("failed to setup X8 Channel Shuffle operator");
      }

      for (auto _ : state) {
        status = pytorch_qnnp_run_operator(
            channelShuffleOperator, nullptr /* thread pool */);
        if (status != pytorch_qnnp_status_success) {
          state.SkipWithError("failed to run X8 Channel Shuffle operator");
        }
      }

      const usize itemsPerIteration = batchSize * groups * groupChannels;
      state.SetItemsProcessed(
          i64(state.iterations()) * i64(itemsPerIteration));

      const usize bytesPerIteration = 2 * itemsPerIteration * sizeof(u8);
      state.SetBytesProcessed(
          i64(state.iterations()) * i64(bytesPerIteration));

      status = pytorch_qnnp_delete_operator(channelShuffleOperator);
      if (status != pytorch_qnnp_status_success) {
        state.SkipWithError("failed to delete X8 Channel Shuffle operator");
      }
        */
}


pub fn shuffle_net_v1g2arguments(b: *mut Benchmark)  {
    
    todo!();
        /*
            b->ArgNames({"N", "G", "GC"});

      /******** Stage 2 ********/
      /*        H    W  G   CG */
      b->Args({56 * 56, 2, 25});
      b->Args({28 * 28, 2, 25});

      /******** Stage 3 ********/
      /*        H    W  G   CG */
      b->Args({28 * 28, 2, 50});
      b->Args({14 * 14, 2, 50});

      /******** Stage 4 ********/
      /*        H    W  G   CG */
      b->Args({14 * 14, 2, 100});
      b->Args({7 * 7, 2, 100});
        */
}


pub fn shuffle_net_v1g3arguments(b: *mut Benchmark)  {
    
    todo!();
        /*
            b->ArgNames({"N", "G", "GC"});

      /******** Stage 2 *******/
      /*        H    W  G  CG */
      b->Args({56 * 56, 3, 20});
      b->Args({28 * 28, 3, 20});

      /******** Stage 3 *******/
      /*        H    W  G  CG */
      b->Args({28 * 28, 3, 40});
      b->Args({14 * 14, 3, 40});

      /******** Stage 4 *******/
      /*        H    W  G  CG */
      b->Args({14 * 14, 3, 80});
      b->Args({7 * 7, 3, 80});
        */
}


pub fn shuffle_net_v1g4arguments(b: *mut Benchmark)  {
    
    todo!();
        /*
            b->ArgNames({"N", "G", "GC"});

      /******** Stage 2 *******/
      /*        H    W  G  CG */
      b->Args({56 * 56, 4, 17});
      b->Args({28 * 28, 4, 17});

      /******** Stage 3 *******/
      /*        H    W  G  CG */
      b->Args({28 * 28, 4, 34});
      b->Args({14 * 14, 4, 34});

      /******** Stage 4 *******/
      /*        H    W  G  CG */
      b->Args({14 * 14, 4, 68});
      b->Args({7 * 7, 4, 68});
        */
}


pub fn shuffle_net_v1g8arguments(b: *mut Benchmark)  {
    
    todo!();
        /*
            b->ArgNames({"N", "G", "GC"});

      /******** Stage 2 *******/
      /*        H    W  G  CG */
      b->Args({56 * 56, 8, 12});
      b->Args({28 * 28, 8, 12});

      /******** Stage 3 *******/
      /*        H    W  G  CG */
      b->Args({28 * 28, 8, 24});
      b->Args({14 * 14, 8, 24});

      /******** Stage 4 *******/
      /*        H    W  G  CG */
      b->Args({14 * 14, 8, 48});
      b->Args({7 * 7, 8, 48});
        */
}


pub fn shuffle_net_v2x0_5arguments(b: *mut Benchmark)  {
    
    todo!();
        /*
            b->ArgNames({"N", "G", "GC"});

      /******** Stage 2 *******/
      /*        H    W  G  CG */
      b->Args({28 * 28, 2, 24});

      /******** Stage 3 *******/
      /*        H    W  G  CG */
      b->Args({14 * 14, 2, 48});

      /******** Stage 4 *******/
      /*        H    W  G  CG */
      b->Args({7 * 7, 2, 96});
        */
}


pub fn shuffle_net_v2x1_0arguments(b: *mut Benchmark)  {
    
    todo!();
        /*
            b->ArgNames({"N", "G", "GC"});

      /******** Stage 2 ********/
      /*        H    W  G   CG */
      b->Args({28 * 28, 2, 58});

      /******** Stage 3 ********/
      /*        H    W  G   CG */
      b->Args({14 * 14, 2, 116});

      /******** Stage 4 ********/
      /*        H    W  G   CG */
      b->Args({7 * 7, 2, 232});
        */
}


pub fn shuffle_net_v2x1_5arguments(b: *mut Benchmark)  {
    
    todo!();
        /*
            b->ArgNames({"N", "G", "GC"});

      /******** Stage 2 ********/
      /*        H    W  G   CG */
      b->Args({28 * 28, 2, 88});

      /******** Stage 3 ********/
      /*        H    W  G   CG */
      b->Args({14 * 14, 2, 176});

      /******** Stage 4 ********/
      /*        H    W  G   CG */
      b->Args({7 * 7, 2, 352});
        */
}


pub fn shuffle_net_v2x2_0arguments(b: *mut Benchmark)  {
    
    todo!();
        /*
            b->ArgNames({"N", "G", "GC"});

      /******** Stage 2 ********/
      /*        H    W  G   CG */
      b->Args({28 * 28, 2, 122});

      /******** Stage 3 ********/
      /*        H    W  G   CG */
      b->Args({14 * 14, 2, 244});

      /******** Stage 4 ********/
      /*        H    W  G   CG */
      b->Args({7 * 7, 2, 488});
        */
}

lazy_static!{
    /*
    BENCHMARK_CAPTURE(
        channel_shuffle_x8,
        shufflenet_v1_g2,
        "ShuffleNet v1 (2 groups)")
        ->Apply(ShuffleNetV1G2Arguments);
    BENCHMARK_CAPTURE(
        channel_shuffle_x8,
        shufflenet_v1_g3,
        "ShuffleNet v1 (3 groups)")
        ->Apply(ShuffleNetV1G3Arguments);
    BENCHMARK_CAPTURE(
        channel_shuffle_x8,
        shufflenet_v1_g4,
        "ShuffleNet v1 (4 groups)")
        ->Apply(ShuffleNetV1G4Arguments);
    BENCHMARK_CAPTURE(
        channel_shuffle_x8,
        shufflenet_v1_g8,
        "ShuffleNet v1 (8 groups)")
        ->Apply(ShuffleNetV1G8Arguments);
    BENCHMARK_CAPTURE(channel_shuffle_x8, shufflenet_v2_x05, "ShuffleNet v2 x0.5")
        ->Apply(ShuffleNetV2x0_5Arguments);
    BENCHMARK_CAPTURE(channel_shuffle_x8, shufflenet_v2_x10, "ShuffleNet v2 x1.0")
        ->Apply(ShuffleNetV2x1_0Arguments);
    BENCHMARK_CAPTURE(channel_shuffle_x8, shufflenet_v2_x15, "ShuffleNet v2 x1.5")
        ->Apply(ShuffleNetV2x1_5Arguments);
    BENCHMARK_CAPTURE(channel_shuffle_x8, shufflenet_v2_x20, "ShuffleNet v2 x2.0")
        ->Apply(ShuffleNetV2x2_0Arguments);

    #ifndef PYTORCH_QNNPACK_BENCHMARK_NO_MAIN
    BENCHMARK_MAIN();
    #endif
    */
}

