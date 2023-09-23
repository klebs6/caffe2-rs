crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/benchmarks/stateful_conv1d.cpp]

pub fn stateful_conv1d(state: &mut BenchmarkState)  {
    
    todo!();
        /*
            const usize input_channels = static_cast<usize>(state.range(0));
      const usize output_channels = static_cast<usize>(state.range(1));
      const usize kernel = static_cast<usize>(state.range(2));
      const usize batch_size = static_cast<usize>(state.range(3));
      const usize width = static_cast<usize>(state.range(4));
      const bool optimized = static_cast<bool>(state.range(5));

      torch::jit::Module m("m");
      m.register_parameter("weight_1", torch::rand({output_channels, input_channels, kernel}), false);
      m.register_parameter("bias_1", torch::rand({output_channels}), false);
      m.register_parameter("weight_2", torch::rand({output_channels, output_channels, kernel}), false);
      m.register_parameter("bias_2", torch::rand({output_channels}), false);
      m.register_parameter("weight_3", torch::rand({output_channels, output_channels, kernel}), false);
      m.register_parameter("bias_3", torch::rand({output_channels}), false);
      m.register_parameter("weight_4", torch::rand({output_channels, output_channels, kernel}), false);
      m.register_parameter("bias_4", torch::rand({output_channels}), false);

      m.define(R"(
        def forward(self, x):
          x = torch.conv1d(x, self.weight_1, self.bias_1, 1, 0, 1, 1)
          x = torch.conv1d(x, self.weight_2, self.bias_2, 1, 0, 1, 1)
          x = torch.conv1d(x, self.weight_3, self.bias_3, 1, 0, 1, 1)
          x = torch.conv1d(x, self.weight_4, self.bias_4, 1, 0, 1, 1)
          return x
      )");

      std::vector<std::vector<torch::jit::IValue>> inputs;
      for (int i = 0; i < 10; ++i) {
        std::vector<torch::jit::IValue> input;
        input.push_back(torch::rand({batch_size, input_channels, width}));
        inputs.push_back(input);
      }

      auto m_cloned = m.clone();
      torch::jit::transformConv1dToConv2d(m_cloned);
      auto m_optimized = torch::jit::optimizeForMobile(m_cloned);
      torch::jit::IValue output;

      if (!optimized) {
        for (auto _ : state) {
          for (const auto& input : inputs) {
            output = m.forward(input);
          }
        }
      } else {
        for (auto _ : state) {
          for (const auto& input : inputs) {
            output = m_optimized.forward(input);
          }
        }
      }
        */
}

pub fn generate_sizes(b: *mut Benchmark)  {
    
    todo!();
        /*
            b->ArgNames({"Input Channels",
                   "Output Channels",
                   "Kernel",
                   "Batch usize",
                   "Width",
                   "Optimized"});

      for (usize input_channels = 32; input_channels < 256; input_channels *= 2) {
        for (usize output_channels = 32; output_channels < 256; output_channels *= 2) {
          for (usize kernel = 3; kernel < 8; ++kernel) {
            for (usize batch_size = 1; batch_size < 5; ++batch_size) {
              for (usize width = 32; width < 256; width *= 2) {
                b->Args({input_channels, output_channels, kernel, batch_size, width, true});
                b->Args({input_channels, output_channels, kernel, batch_size, width, false});
              }
            }
          }
        }
      }
        */
}

lazy_static!{
    /*
    BENCHMARK(stateful_conv1d)->Apply(GenerateSizes);
    BENCHMARK_MAIN();
    */
}
