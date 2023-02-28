crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/test/vulkan_api_test.cpp]

#[cfg(USE_VULKAN_API)]
mod use_vulkan_api {
    use super::*;

    // TODO: These functions should move to a common place.
    pub fn check_rtol(
        diff:   &Tensor,
        inputs: &Vec<Tensor>) -> bool {
        
        todo!();
            /*
                float maxValue = 0.0f;

          for (const auto& tensor : inputs) {
            maxValue = fmax(tensor.abs().max().item<float>(), maxValue);
          }

        #ifdef USE_VULKAN_FP16_INFERENCE
          constexpr float tolerance = 1e-2;
        #else
          constexpr float tolerance = 1e-5;
        #endif

          return diff.abs().max().item<float>() < (tolerance * maxValue);
            */
    }

    pub fn almost_equal(
        a: &Tensor,
        b: &Tensor) -> bool {
        
        todo!();
            /*
                return checkRtol(a - b, {a, b});
            */
    }

    pub fn exactly_equal(
            a: &Tensor,
            b: &Tensor) -> bool {
        
        todo!();
            /*
                return (a - b).abs().max().item<float>() == 0.0f;
            */
    }

    pub fn show_rtol(
            a: &Tensor,
            b: &Tensor)  {
        
        todo!();
            /*
                const auto diff = (a - b).abs();

              float maxValue = a.abs().max().item<float>();
              maxValue = fmax(b.abs().max().item<float>(), maxValue);

            #ifdef USE_VULKAN_FP16_INFERENCE
              constexpr float tolerance = 1e-2;
            #else
              constexpr float tolerance = 1e-5;
            #endif

              const float maxDiff = maxValue * tolerance;
              cout << "Max Diff allowed: " << maxDiff << endl;
              if (diff.sizes().size() == 2) {
                for (int y = 0; y < diff.sizes()[0]; y++) {
                  cout << y << ":";
                  for (int x = 0; x < diff.sizes()[1]; x++) {
                    float diff_xy = diff[y][x].item<float>();
                    if (diff_xy > maxDiff) {
                      cout << setw(5) << x;
                    }
                    else {
                      cout << setw(5) << " ";
                    }
                  }
                  cout << endl;
                }
              }
            */
    }

    #[test] fn vulkan_api_test_adaptive_avg_pool2d() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }
          InferenceMode mode;

          const auto in_cpu = rand({5, 7, 47, 31}, TensorOptions(kCPU).dtype(kFloat));
          const auto out_cpu = adaptive_avg_pool2d(in_cpu, {3, 3});
          const auto out_vulkan = adaptive_avg_pool2d(in_cpu.vulkan(), {3, 3});

          const auto check = almostEqual(out_cpu, out_vulkan.cpu());
          if (!check) {
            showRtol(out_cpu, out_vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_add() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          const auto a_cpu = rand({11, 7, 139, 109}, device(kCPU).dtype(kFloat));
          const auto a_vulkan = a_cpu.vulkan();

          const auto b_cpu = rand({11, 7, 139, 109}, device(kCPU).dtype(kFloat));
          const auto b_vulkan = b_cpu.vulkan();

          const auto c_cpu = add(a_cpu, b_cpu, 2.1f);
          const auto c_vulkan = add(a_vulkan, b_vulkan, 2.1f);

          const auto check = almostEqual(c_cpu, c_vulkan.cpu());
          if (!check) {
            showRtol(c_cpu, c_vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_add_broadcast0() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          const auto a_cpu = rand({3, 5, 179, 221}, device(kCPU).dtype(kFloat));
          const auto a_vulkan = a_cpu.vulkan();

          const auto b_cpu = rand({3, 5, 1, 1}, device(kCPU).dtype(kFloat));
          const auto b_vulkan = b_cpu.vulkan();

          const auto c_cpu = add(a_cpu, b_cpu, 1.8f);
          const auto c_vulkan = add(a_vulkan, b_vulkan, 1.8f);

          const auto check = almostEqual(c_cpu, c_vulkan.cpu());
          if (!check) {
            showRtol(c_cpu, c_vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_add_broadcast1() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          const auto a_cpu = rand({3, 5, 179, 221}, device(kCPU).dtype(kFloat));
          const auto a_vulkan = a_cpu.vulkan();

          const auto b_cpu = rand({3, 5, 1, 221}, device(kCPU).dtype(kFloat));
          const auto b_vulkan = b_cpu.vulkan();

          const auto c_cpu = add(a_cpu, b_cpu, 1.8f);
          const auto c_vulkan = add(a_vulkan, b_vulkan, 1.8f);

          const auto check = almostEqual(c_cpu, c_vulkan.cpu());
          if (!check) {
            showRtol(c_cpu, c_vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_add_broadcast2() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          const auto a_cpu = rand({3, 4, 179, 221}, device(kCPU).dtype(kFloat));
          const auto a_vulkan = a_cpu.vulkan();

          const auto b_cpu = rand({4, 1, 1}, device(kCPU).dtype(kFloat));
          const auto b_vulkan = b_cpu.vulkan();

          const auto c_cpu = add(a_cpu, b_cpu, 2.5f);
          const auto c_vulkan = add(a_vulkan, b_vulkan, 2.5f);

          const auto check = almostEqual(c_cpu, c_vulkan.cpu());
          if (!check) {
            showRtol(c_cpu, c_vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_add() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          auto a_cpu = rand({61, 17, 29, 83}, device(kCPU).dtype(kFloat));
          auto a_vulkan = a_cpu.vulkan();

          const auto b_cpu = rand({61, 17, 29, 83}, device(kCPU).dtype(kFloat));
          const auto b_vulkan = b_cpu.vulkan();

          a_cpu.add_(b_cpu, 2.1f);
          a_vulkan.add_(b_vulkan, 2.1f);

          const auto check = almostEqual(a_cpu, a_vulkan.cpu());
          if (!check) {
            showRtol(b_cpu, b_vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_add_broadcast0() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          auto a_cpu = rand({16, 17, 29, 83}, device(kCPU).dtype(kFloat));
          auto a_vulkan = a_cpu.vulkan();

          const auto b_cpu = rand({16, 17, 29, 1}, device(kCPU).dtype(kFloat));
          const auto b_vulkan = b_cpu.vulkan();

          a_cpu.add_(b_cpu, 2.1f);
          a_vulkan.add_(b_vulkan, 2.1f);

          const auto check = almostEqual(a_cpu, a_vulkan.cpu());
          if (!check) {
            showRtol(b_cpu, b_vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_add_broadcast1() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          auto a_cpu = rand({3, 8, 29, 83}, device(kCPU).dtype(kFloat));
          auto a_vulkan = a_cpu.vulkan();

          const auto b_cpu = rand({3, 8, 1, 1}, device(kCPU).dtype(kFloat));
          const auto b_vulkan = b_cpu.vulkan();

          a_cpu.add_(b_cpu, 2.1f);
          a_vulkan.add_(b_vulkan, 2.1f);

          const auto check = almostEqual(a_cpu, a_vulkan.cpu());
          if (!check) {
            showRtol(b_cpu, b_vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_add_scalar() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          const auto a_cpu = rand({13, 23, 59, 73}, device(kCPU).dtype(kFloat));
          const auto a_vulkan = a_cpu.vulkan();

          const float b_scalar = 3.1415f;

          const auto c_cpu = add(a_cpu, b_scalar, 2.1f);
          const auto c_vulkan = add(a_vulkan, b_scalar, 2.1f);

          const auto check = almostEqual(c_cpu, c_vulkan.cpu());
          if (!check) {
            showRtol(c_cpu, c_vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_add_scalar() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          auto a_cpu = rand({47, 2, 23, 97}, device(kCPU).dtype(kFloat));
          auto a_vulkan = a_cpu.vulkan();

          const float b_scalar = 3.1415f;

          a_cpu.add_(b_scalar, 2.1f);
          a_vulkan.add_(b_scalar, 2.1f);

          const auto check = almostEqual(a_cpu, a_vulkan.cpu());
          if (!check) {
            showRtol(a_cpu, a_vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_addmm() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          constexpr float alpha = 2.1f;
          constexpr float beta = 103.24;

          const auto bias_cpu = rand({179, 163}, device(kCPU).dtype(kFloat));
          const auto m1_cpu = rand({179, 67}, device(kCPU).dtype(kFloat));
          const auto m2_cpu = rand({67, 163}, device(kCPU).dtype(kFloat));
          const auto out_cpu = addmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);

          const auto m1_vulkan = m1_cpu.vulkan();
          const auto out_vulkan = addmm(bias_cpu, m1_vulkan, m2_cpu, beta, alpha);

          const auto check = almostEqual(out_cpu, out_vulkan.cpu());
          if (!check) {
            showRtol(out_cpu, out_vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_addmm_expand() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          constexpr float alpha = 2.1f;
          constexpr float beta = 103.24;

          const auto bias_cpu = rand({1000}, device(kCPU).dtype(kFloat));
          const auto m1_cpu = rand({1, 1280}, device(kCPU).dtype(kFloat));
          const auto m2_cpu = rand({1280, 1000}, device(kCPU).dtype(kFloat));
          const auto out_cpu = addmm(bias_cpu, m1_cpu, m2_cpu, beta, alpha);

          const auto m1_vulkan = m1_cpu.vulkan();
          const auto out_vulkan = addmm(bias_cpu, m1_vulkan, m2_cpu, beta, alpha);

          const auto check = almostEqual(out_cpu, out_vulkan.cpu());
          if (!check) {
            showRtol(out_cpu, out_vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_avg_pool2d() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          const auto in_cpu = rand({3, 19, 43, 79}, TensorOptions(kCPU).dtype(kFloat));
          const auto out_cpu = avg_pool2d(in_cpu, {5, 3}, {1, 2}, {2, 0}, true);
          const auto out_vulkan = avg_pool2d(in_cpu.vulkan(), {5, 3}, {1, 2}, {2, 0}, true);

          const auto check = almostEqual(out_cpu, out_vulkan.cpu());
          if (!check) {
            showRtol(out_cpu, out_vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_clamp() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          const auto in_cpu = rand({17, 197, 302, 5}, device(kCPU).dtype(kFloat));
          const auto in_vulkan = in_cpu.vulkan();

          const float min_value = 0.2f;
          const float max_value = 0.8f;

          const auto out_cpu = clamp(in_cpu, min_value, max_value);
          const auto out_vulkan = clamp(in_vulkan, min_value, max_value);

          const auto check = almostEqual(out_cpu, out_vulkan.cpu());
          if (!check) {
            showRtol(out_cpu, out_vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_clamp() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          const auto cpu = rand({17, 197, 302, 5}, device(kCPU).dtype(kFloat));
          const auto vulkan = cpu.vulkan();

          const float min_value = 0.2f;
          const float max_value = 0.8f;

          cpu.clamp_(min_value, max_value);
          vulkan.clamp_(min_value, max_value);

          const auto check = almostEqual(cpu, vulkan.cpu());
          if (!check) {
            showRtol(cpu, vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_conv2d() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          constexpr i64 groups = 1;
          constexpr array<i64, 2u> stride{2, 2};
          constexpr array<i64, 2u> padding{1, 1};
          //TODO: Support conv2d with dilation != 1
          constexpr array<i64, 2u> dilation{1, 1};

          constexpr struct {
            u32 batches;
            u32 channels;
            u32 width;
            u32 height;

            array<i64, 4u> size() const {
              return {
                batches,
                channels,
                width,
                height,
              };
            }
          } input {1, 3, 8, 8};

          constexpr struct {
            u32 output_channels;
            u32 input_channels;
            u32 width;
            u32 height;

            array<i64, 4u> size() const {
              return {
                output_channels,
                input_channels,
                width,
                height,
              };
            }
          } weights {1, input.channels, 3, 3};

          const auto input_cpu = randn(input.size(), device(kCPU).dtype(kFloat));
          const auto weights_cpu = randn(weights.size(), device(kCPU).dtype(kFloat));
          const auto bias_cpu = randn({weights.output_channels}, device(kCPU).dtype(kFloat));

          const auto output_cpu = conv2d(
              input_cpu,
              weights_cpu,
              bias_cpu,
              stride,
              padding,
              dilation,
              groups);

          const auto output_vulkan = conv2d(
              input_cpu.vulkan(),
              weights_cpu,
              bias_cpu,
              stride,
              padding,
              dilation,
              groups).cpu();

          const bool check = almostEqual(output_cpu, output_vulkan);
          if (!check) {
            showRtol(output_cpu, output_vulkan);
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_conv2d_dw() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          constexpr i64 groups = 7;
          constexpr array<i64, 2u> stride{2, 3};
          constexpr array<i64, 2u> padding{0, 4};
          constexpr array<i64, 2u> dilation{3, 1};

          constexpr struct {
            u32 batches;
            u32 channels;
            u32 width;
            u32 height;

            array<i64, 4u> size() const {
              return {
                batches,
                channels,
                width,
                height,
              };
            }
          } input {1, groups, 137, 199};

          constexpr struct {
            u32 output_channels;
            u32 input_channels;
            u32 width;
            u32 height;

            array<i64, 4u> size() const {
              return {
                output_channels,
                input_channels,
                width,
                height,
              };
            }
          } weights {groups, 1, 17, 7};

          const auto input_cpu = rand(input.size(), device(kCPU).dtype(kFloat));
          const auto weights_cpu = rand(weights.size(), device(kCPU).dtype(kFloat));
          const auto bias_cpu = rand({weights.output_channels}, device(kCPU).dtype(kFloat));

          const auto output_cpu = conv2d(
              input_cpu,
              weights_cpu,
              bias_cpu,
              stride,
              padding,
              dilation,
              groups);

          const auto output_vulkan = conv2d(
              input_cpu.vulkan(),
              weights_cpu,
              bias_cpu,
              stride,
              padding,
              dilation,
              groups);

          const bool check = almostEqual(output_cpu, output_vulkan.cpu());
          if (!check) {
            showRtol(output_cpu, output_vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_conv2d_pw() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          constexpr i64 groups = 1;
          constexpr array<i64, 2u> stride{1, 1};
          constexpr array<i64, 2u> padding{0, 0};
          constexpr array<i64, 2u> dilation{1, 1};

          constexpr struct {
            u32 batches;
            u32 channels;
            u32 width;
            u32 height;

            array<i64, 4u> size() const {
              return {
                batches,
                channels,
                width,
                height,
              };
            }
          } input {1, 17, 127, 397};

          constexpr struct {
            u32 output_channels;
            u32 input_channels;
            u32 width;
            u32 height;

            array<i64, 4u> size() const {
              return {
                output_channels,
                input_channels,
                width,
                height,
              };
            }
          } weights {29, input.channels, 1, 1};

          const auto input_cpu = randn(input.size(), device(kCPU).dtype(kFloat));
          const auto weights_cpu = randn(weights.size(), device(kCPU).dtype(kFloat));
          const auto bias_cpu = randn({weights.output_channels}, device(kCPU).dtype(kFloat));

          const auto output_cpu = conv2d(
              input_cpu,
              weights_cpu,
              bias_cpu,
              stride,
              padding,
              dilation,
              groups);

          const auto output_vulkan = conv2d(
              input_cpu.vulkan(),
              weights_cpu,
              bias_cpu,
              stride,
              padding,
              dilation,
              groups);

          const bool check = almostEqual(output_cpu, output_vulkan.cpu());
          if (!check) {
            showRtol(output_cpu, output_vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_conv2d_winograd() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          constexpr i64 groups = 1;
          constexpr array<i64, 2u> stride{1, 1};
          constexpr array<i64, 2u> padding{2, 2};
          constexpr array<i64, 2u> dilation{1, 1};

          constexpr struct {
            u32 batches;
            u32 channels;
            u32 width;
            u32 height;

            array<i64, 4u> size() const {
              return {
                batches,
                channels,
                width,
                height,
              };
            }
          } input {1, 10, 177, 232};

          constexpr struct {
            u32 output_channels;
            u32 input_channels;
            u32 width;
            u32 height;

            array<i64, 4u> size() const {
              return {
                output_channels,
                input_channels,
                width,
                height,
              };
            }
          } weights {13, input.channels, 3, 3};

          const auto input_cpu = rand(input.size(), device(kCPU).dtype(kFloat));
          const auto weights_cpu = rand(weights.size(), device(kCPU).dtype(kFloat));
          const auto bias_cpu = rand({weights.output_channels}, device(kCPU).dtype(kFloat));

          const auto output_cpu = conv2d(
              input_cpu,
              weights_cpu,
              bias_cpu,
              stride,
              padding,
              dilation,
              groups);

          const auto output_vulkan = conv2d(
              input_cpu.vulkan(),
              weights_cpu,
              bias_cpu,
              stride,
              padding,
              dilation,
              groups).cpu();

          const bool check = almostEqual(output_cpu, output_vulkan);
          if (!check) {
            showRtol(output_cpu, output_vulkan);
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_copy_() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          const auto cpu = rand({13, 17, 37, 19}, device(kCPU).dtype(kFloat));
          const auto vulkan = cpu.vulkan();

          const auto check = exactlyEqual(cpu, vulkan.cpu());
          if (!check) {
            showRtol(cpu, vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_div() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          const auto a_cpu = rand({11, 7, 139, 109}, device(kCPU).dtype(kFloat))+0.01;
          const auto a_vulkan = a_cpu.vulkan();

          const auto b_cpu = rand({11, 7, 139, 109}, device(kCPU).dtype(kFloat))+0.01;
          const auto b_vulkan = b_cpu.vulkan();

          const auto c_cpu = div(a_cpu, b_cpu);
          const auto c_vulkan = div(a_vulkan, b_vulkan);

          const auto check = almostEqual(c_cpu, c_vulkan.cpu());
          if (!check) {
            showRtol(c_cpu, c_vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_div_broadcast0() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          const auto a_cpu = rand({3, 5, 1, 1}, device(kCPU).dtype(kFloat))+0.01;
          const auto a_vulkan = a_cpu.vulkan();

          const auto b_cpu = rand({3, 5, 179, 221}, device(kCPU).dtype(kFloat))+0.01;
          const auto b_vulkan = b_cpu.vulkan();

          const auto c_cpu = div(a_cpu, b_cpu);
          const auto c_vulkan = div(a_vulkan, b_vulkan);

          const auto check = almostEqual(c_cpu, c_vulkan.cpu());
          if (!check) {
            showRtol(c_cpu, c_vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_div_broadcast1() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          const auto a_cpu = rand({3, 5, 179, 221}, device(kCPU).dtype(kFloat))+0.01;
          const auto a_vulkan = a_cpu.vulkan();

          const auto b_cpu = rand({3, 5, 1, 221}, device(kCPU).dtype(kFloat))+0.01;
          const auto b_vulkan = b_cpu.vulkan();

          const auto c_cpu = div(a_cpu, b_cpu);
          const auto c_vulkan = div(a_vulkan, b_vulkan);

          const auto check = almostEqual(c_cpu, c_vulkan.cpu());
          if (!check) {
            showRtol(c_cpu, c_vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_div_broadcast2() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          const auto a_cpu = rand({3, 4, 179, 221}, device(kCPU).dtype(kFloat))+0.01;
          const auto a_vulkan = a_cpu.vulkan();

          const auto b_cpu = rand({4, 1, 1}, device(kCPU).dtype(kFloat))+0.01;
          const auto b_vulkan = b_cpu.vulkan();

          const auto c_cpu = div(a_cpu, b_cpu);
          const auto c_vulkan = div(a_vulkan, b_vulkan);

          const auto check = almostEqual(c_cpu, c_vulkan.cpu());
          if (!check) {
            showRtol(c_cpu, c_vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_div() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          auto a_cpu = rand({61, 17, 29, 83}, device(kCPU).dtype(kFloat))+0.01;
          auto a_vulkan = a_cpu.vulkan();

          const auto b_cpu = rand({61, 17, 29, 83}, device(kCPU).dtype(kFloat))+0.01;
          const auto b_vulkan = b_cpu.vulkan();

          a_cpu.div_(b_cpu);
          a_vulkan.div_(b_vulkan);

          const auto check = almostEqual(a_cpu, a_vulkan.cpu());
          if (!check) {
            showRtol(b_cpu, b_vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_div_broadcast0() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          auto a_cpu = rand({12, 17, 29, 83}, device(kCPU).dtype(kFloat))+0.01;
          auto a_vulkan = a_cpu.vulkan();

          const auto b_cpu = rand({12, 17, 29, 1}, device(kCPU).dtype(kFloat))+0.01;
          const auto b_vulkan = b_cpu.vulkan();

          a_cpu.div_(b_cpu);
          a_vulkan.div_(b_vulkan);

          const auto check = almostEqual(a_cpu, a_vulkan.cpu());
          if (!check) {
            showRtol(b_cpu, b_vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_div_broadcast1() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          auto a_cpu = rand({3, 8, 29, 83}, device(kCPU).dtype(kFloat))+0.01;
          auto a_vulkan = a_cpu.vulkan();

          const auto b_cpu = rand({8, 1, 1}, device(kCPU).dtype(kFloat))+0.01;
          const auto b_vulkan = b_cpu.vulkan();

          a_cpu.div_(b_cpu);
          a_vulkan.div_(b_vulkan);

          const auto check = almostEqual(a_cpu, a_vulkan.cpu());
          if (!check) {
            showRtol(b_cpu, b_vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_div_scalar() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          const auto a_cpu = rand({17, 213, 213, 7}, device(kCPU).dtype(kFloat));
          const auto a_vulkan = a_cpu.vulkan();

          const float b_scalar = 3.1415f;

          const auto c_cpu = div(a_cpu, b_scalar);
          const auto c_vulkan = div(a_vulkan, b_scalar);

          const auto check = almostEqual(c_cpu, c_vulkan.cpu());
          if (!check) {
            showRtol(c_cpu, c_vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_div_scalar() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          auto a_cpu = rand({11, 7, 139, 109}, device(kCPU).dtype(kFloat));
          auto a_vulkan = a_cpu.vulkan();

          const float b_scalar = 3.1415f;

          a_cpu.div_(b_scalar);
          a_vulkan.div_(b_scalar);

          const auto check = almostEqual(a_cpu, a_vulkan.cpu());
          if (!check) {
            showRtol(a_cpu, a_vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_empty() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          ASSERT_NO_THROW(empty({1, 17, 41, 53}, device(kVulkan).dtype(kFloat)));

        */
    }

    #[test] fn vulkan_api_test_hardsigmoid() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          const auto in_cpu = rand({17, 197, 302, 5}, device(kCPU).dtype(kFloat))*12 - 6;
          const auto in_vulkan = in_cpu.vulkan();

          const auto out_cpu = hardsigmoid(in_cpu);
          const auto out_vulkan = hardsigmoid(in_vulkan);

          const auto check = almostEqual(out_cpu, out_vulkan.cpu());
          if (!check) {
            showRtol(out_cpu, out_vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_hardsigmoid() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          auto cpu = rand({17, 197, 302, 5}, device(kCPU).dtype(kFloat))*12 - 6;
          auto vulkan = cpu.vulkan();

          hardsigmoid_(cpu);
          hardsigmoid_(vulkan);

          const auto check = almostEqual(cpu, vulkan.cpu());
          if (!check) {
            showRtol(cpu, vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_hardswish() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          const auto in_cpu = rand({17, 197, 302, 5}, device(kCPU).dtype(kFloat))*12 - 6;
          const auto in_vulkan = in_cpu.vulkan();

          const auto out_cpu = hardswish(in_cpu);
          const auto out_vulkan = hardswish(in_vulkan);

          const auto check = almostEqual(out_cpu, out_vulkan.cpu());
          if (!check) {
            showRtol(out_cpu, out_vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_hardswish() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          auto cpu = rand({17, 197, 302, 5}, device(kCPU).dtype(kFloat))*12 - 6;
          auto vulkan = cpu.vulkan();

          native::hardswish_(cpu);
          hardswish_(vulkan);

          const auto check = almostEqual(cpu, vulkan.cpu());
          if (!check) {
            showRtol(cpu, vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_max_pool2d() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }
          InferenceMode mode;

          const auto in_cpu = rand({5, 13, 55, 68}, TensorOptions(kCPU).dtype(kFloat));
          const auto out_cpu = max_pool2d(in_cpu, {3, 4}, {2, 1}, {1, 1}, {1, 1}, false);
          const auto out_vulkan = max_pool2d(in_cpu.vulkan(), {3, 4}, {2, 1}, {1, 1}, {1,1}, false);

          const auto check = almostEqual(out_cpu, out_vulkan.cpu());
          if (!check) {
            showRtol(out_cpu, out_vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_mean() {
        todo!();
        /*
        
          const auto in_cpu = rand({17, 3, 79, 53}, TensorOptions(kCPU).dtype(kFloat));
          const auto out_cpu = mean(in_cpu, {-1, -2}, true);

          const auto in_vulkan = in_cpu.vulkan();
          const auto out_vulkan = mean(in_vulkan, {-1, -2}, true);

          const auto check = almostEqual(out_cpu, out_vulkan.cpu());
          if (!check) {
            showRtol(out_cpu, out_vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_mean2d() {
        todo!();
        /*
        
          const auto in_cpu = rand({11, 7, 173, 37}, TensorOptions(kCPU).dtype(kFloat));
          const auto out_cpu = mean(in_cpu, {-1, -2}, false);

          const auto in_vulkan = in_cpu.vulkan();
          const auto out_vulkan = mean(in_vulkan, {-1, -2}, false);

          const auto check = almostEqual(out_cpu, out_vulkan.cpu());
          if (!check) {
            showRtol(out_cpu, out_vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_mm() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          const auto m1_cpu = rand({179, 67}, device(kCPU).dtype(kFloat));
          const auto m2_cpu = rand({67, 163}, device(kCPU).dtype(kFloat));
          const auto out_cpu = m1_cpu.mm(m2_cpu);

          const auto m1_vulkan = m1_cpu.vulkan();
          const auto out_vulkan = m1_vulkan.mm(m2_cpu);

          const auto check = almostEqual(out_cpu, out_vulkan.cpu());
          if (!check) {
            showRtol(out_cpu, out_vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_mul() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          const auto a_cpu = rand({11, 7, 139, 109}, device(kCPU).dtype(kFloat));
          const auto a_vulkan = a_cpu.vulkan();

          const auto b_cpu = rand({11, 7, 139, 109}, device(kCPU).dtype(kFloat));
          const auto b_vulkan = b_cpu.vulkan();

          const auto c_cpu = mul(a_cpu, b_cpu);
          const auto c_vulkan = mul(a_vulkan, b_vulkan);

          const auto check = almostEqual(c_cpu, c_vulkan.cpu());
          if (!check) {
            showRtol(c_cpu, c_vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_mul_broadcast0() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          const auto a_cpu = rand({3, 5, 1, 1}, device(kCPU).dtype(kFloat));
          const auto a_vulkan = a_cpu.vulkan();

          const auto b_cpu = rand({3, 5, 179, 221}, device(kCPU).dtype(kFloat));
          const auto b_vulkan = b_cpu.vulkan();

          const auto c_cpu = mul(a_cpu, b_cpu);
          const auto c_vulkan = mul(a_vulkan, b_vulkan);

          const auto check = almostEqual(c_cpu, c_vulkan.cpu());
          if (!check) {
            showRtol(c_cpu, c_vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_mul_broadcast1() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          const auto a_cpu = rand({3, 5, 179, 221}, device(kCPU).dtype(kFloat));
          const auto a_vulkan = a_cpu.vulkan();

          const auto b_cpu = rand({3, 5, 1, 221}, device(kCPU).dtype(kFloat));
          const auto b_vulkan = b_cpu.vulkan();

          const auto c_cpu = mul(a_cpu, b_cpu);
          const auto c_vulkan = mul(a_vulkan, b_vulkan);

          const auto check = almostEqual(c_cpu, c_vulkan.cpu());
          if (!check) {
            showRtol(c_cpu, c_vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_mul_broadcast2() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          const auto a_cpu = rand({3, 4, 179, 221}, device(kCPU).dtype(kFloat));
          const auto a_vulkan = a_cpu.vulkan();

          const auto b_cpu = rand({4, 1, 1}, device(kCPU).dtype(kFloat));
          const auto b_vulkan = b_cpu.vulkan();

          const auto c_cpu = mul(a_cpu, b_cpu);
          const auto c_vulkan = mul(a_vulkan, b_vulkan);

          const auto check = almostEqual(c_cpu, c_vulkan.cpu());
          if (!check) {
            showRtol(c_cpu, c_vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_mul() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          auto a_cpu = rand({61, 17, 29, 83}, device(kCPU).dtype(kFloat));
          auto a_vulkan = a_cpu.vulkan();

          const auto b_cpu = rand({61, 17, 29, 83}, device(kCPU).dtype(kFloat));
          const auto b_vulkan = b_cpu.vulkan();

          a_cpu.mul_(b_cpu);
          a_vulkan.mul_(b_vulkan);

          const auto check = almostEqual(a_cpu, a_vulkan.cpu());
          if (!check) {
            showRtol(b_cpu, b_vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_mul_broadcast0() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          auto a_cpu = rand({12, 17, 29, 83}, device(kCPU).dtype(kFloat));
          auto a_vulkan = a_cpu.vulkan();

          const auto b_cpu = rand({12, 17, 29, 1}, device(kCPU).dtype(kFloat));
          const auto b_vulkan = b_cpu.vulkan();

          a_cpu.mul_(b_cpu);
          a_vulkan.mul_(b_vulkan);

          const auto check = almostEqual(a_cpu, a_vulkan.cpu());
          if (!check) {
            showRtol(b_cpu, b_vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_mul_broadcast1() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          auto a_cpu = rand({3, 8, 29, 83}, device(kCPU).dtype(kFloat));
          auto a_vulkan = a_cpu.vulkan();

          const auto b_cpu = rand({8, 1, 1}, device(kCPU).dtype(kFloat));
          const auto b_vulkan = b_cpu.vulkan();

          a_cpu.mul_(b_cpu);
          a_vulkan.mul_(b_vulkan);

          const auto check = almostEqual(a_cpu, a_vulkan.cpu());
          if (!check) {
            showRtol(b_cpu, b_vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_mul_scalar() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          const auto a_cpu = rand({17, 213, 213, 7}, device(kCPU).dtype(kFloat));
          const auto a_vulkan = a_cpu.vulkan();

          const float b_scalar = 3.1415f;

          const auto c_cpu = mul(a_cpu, b_scalar);
          const auto c_vulkan = mul(a_vulkan, b_scalar);

          const auto check = almostEqual(c_cpu, c_vulkan.cpu());
          if (!check) {
            showRtol(c_cpu, c_vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_mul_scalar() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          auto a_cpu = rand({11, 7, 139, 109}, device(kCPU).dtype(kFloat));
          auto a_vulkan = a_cpu.vulkan();

          const float b_scalar = 3.1415f;

          a_cpu.mul_(b_scalar);
          a_vulkan.mul_(b_scalar);

          const auto check = almostEqual(a_cpu, a_vulkan.cpu());
          if (!check) {
            showRtol(a_cpu, a_vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_reflection_pad2d() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          const auto a_cpu = rand({2, 3, 47, 63}, device(kCPU).dtype(kFloat));
          const auto a_vulkan = a_cpu.vulkan();

          const auto out_cpu = reflection_pad2d(a_cpu, {9,8,5,12});
          const auto out_vulkan = reflection_pad2d(a_vulkan, {9,8,5,12}).cpu();

          const auto check = almostEqual(out_cpu, out_vulkan);
          if (!check) {
            showRtol(out_cpu, out_vulkan);
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_reshape() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }
          InferenceMode mode;

          const auto in_cpu = rand({47, 11, 83, 97}, device(kCPU).dtype(kFloat));
          const auto in_vulkan = in_cpu.vulkan();

          const array<i64, 2> shape{47 * 83, 11 * 97};

          const auto out_cpu = reshape(in_cpu, shape);
          const auto out_vulkan = reshape(in_vulkan, shape);

          const auto check = almostEqual(out_cpu, out_vulkan.cpu());
          if (!check) {
            showRtol(out_cpu, out_vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_reshape() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }
          InferenceMode mode;

          const auto cpu = rand({59, 41, 19, 67}, device(kCPU).dtype(kFloat));
          const auto vulkan = cpu.vulkan();

          const array<i64, 3> shape{59, 41 * 67, 19};

          cpu.reshape(shape);
          vulkan.reshape(shape);

          const auto check = almostEqual(cpu, vulkan.cpu());
          if (!check) {
            showRtol(cpu, vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_sigmoid() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          const auto in_cpu = rand({17, 197, 302, 5}, device(kCPU).dtype(kFloat));
          const auto in_vulkan = in_cpu.vulkan();

          const auto out_cpu = sigmoid(in_cpu);
          const auto out_vulkan = sigmoid(in_vulkan);

          const auto check = almostEqual(out_cpu, out_vulkan.cpu());
          if (!check) {
            showRtol(out_cpu, out_vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_sigmoid() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          auto cpu = rand({17, 197, 302, 5}, device(kCPU).dtype(kFloat));
          auto vulkan = cpu.vulkan();

          sigmoid_(cpu);
          sigmoid_(vulkan);

          const auto check = almostEqual(cpu, vulkan.cpu());
          if (!check) {
            showRtol(cpu, vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_sub() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          const auto a_cpu = rand({11, 7, 139, 109}, device(kCPU).dtype(kFloat));
          const auto a_vulkan = a_cpu.vulkan();

          const auto b_cpu = rand({11, 7, 139, 109}, device(kCPU).dtype(kFloat));
          const auto b_vulkan = b_cpu.vulkan();

          const auto c_cpu = sub(a_cpu, b_cpu, 2.1f);
          const auto c_vulkan = sub(a_vulkan, b_vulkan, 2.1f);

          const auto check = almostEqual(c_cpu, c_vulkan.cpu());
          if (!check) {
            showRtol(c_cpu, c_vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_sub_broadcast0() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          const auto a_cpu = rand({3, 5, 179, 221}, device(kCPU).dtype(kFloat));
          const auto a_vulkan = a_cpu.vulkan();

          const auto b_cpu = rand({3, 5, 1, 1}, device(kCPU).dtype(kFloat));
          const auto b_vulkan = b_cpu.vulkan();

          const auto c_cpu = sub(a_cpu, b_cpu, 1.8f);
          const auto c_vulkan = sub(a_vulkan, b_vulkan, 1.8f);

          const auto check = almostEqual(c_cpu, c_vulkan.cpu());
          if (!check) {
            showRtol(c_cpu, c_vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_sub_broadcast1() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          const auto a_cpu = rand({3, 5, 179, 221}, device(kCPU).dtype(kFloat));
          const auto a_vulkan = a_cpu.vulkan();

          const auto b_cpu = rand({3, 5, 1, 221}, device(kCPU).dtype(kFloat));
          const auto b_vulkan = b_cpu.vulkan();

          const auto c_cpu = sub(a_cpu, b_cpu, 1.8f);
          const auto c_vulkan = sub(a_vulkan, b_vulkan, 1.8f);

          const auto check = almostEqual(c_cpu, c_vulkan.cpu());
          if (!check) {
            showRtol(c_cpu, c_vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_sub_broadcast2() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          const auto a_cpu = rand({3, 4, 179, 221}, device(kCPU).dtype(kFloat));
          const auto a_vulkan = a_cpu.vulkan();

          const auto b_cpu = rand({4, 1, 1}, device(kCPU).dtype(kFloat));
          const auto b_vulkan = b_cpu.vulkan();

          const auto c_cpu = sub(a_cpu, b_cpu, 2.5f);
          const auto c_vulkan = sub(a_vulkan, b_vulkan, 2.5f);

          const auto check = almostEqual(c_cpu, c_vulkan.cpu());
          if (!check) {
            showRtol(c_cpu, c_vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_sub() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          auto a_cpu = rand({61, 17, 29, 83}, device(kCPU).dtype(kFloat));
          auto a_vulkan = a_cpu.vulkan();

          const auto b_cpu = rand({61, 17, 29, 83}, device(kCPU).dtype(kFloat));
          const auto b_vulkan = b_cpu.vulkan();

          a_cpu.sub_(b_cpu, 2.1f);
          a_vulkan.sub_(b_vulkan, 2.1f);

          const auto check = almostEqual(a_cpu, a_vulkan.cpu());
          if (!check) {
            showRtol(b_cpu, b_vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_sub_broadcast0() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          auto a_cpu = rand({16, 17, 29, 83}, device(kCPU).dtype(kFloat));
          auto a_vulkan = a_cpu.vulkan();

          const auto b_cpu = rand({16, 17, 29, 1}, device(kCPU).dtype(kFloat));
          const auto b_vulkan = b_cpu.vulkan();

          a_cpu.sub_(b_cpu, 2.1f);
          a_vulkan.sub_(b_vulkan, 2.1f);

          const auto check = almostEqual(a_cpu, a_vulkan.cpu());
          if (!check) {
            showRtol(b_cpu, b_vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_sub_broadcast1() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          auto a_cpu = rand({3, 8, 29, 83}, device(kCPU).dtype(kFloat));
          auto a_vulkan = a_cpu.vulkan();

          const auto b_cpu = rand({3, 8, 1, 1}, device(kCPU).dtype(kFloat));
          const auto b_vulkan = b_cpu.vulkan();

          a_cpu.sub_(b_cpu, 2.1f);
          a_vulkan.sub_(b_vulkan, 2.1f);

          const auto check = almostEqual(a_cpu, a_vulkan.cpu());
          if (!check) {
            showRtol(b_cpu, b_vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_api_test_upsample_nearest2d() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }

          const auto in_cpu = rand({1, 2, 2, 3}, TensorOptions(kCPU).dtype(kFloat));
          const auto out_cpu = upsample_nearest2d(in_cpu, {4, 6});

          const auto in_vulkan = in_cpu.vulkan();
          const auto out_vulkan = upsample_nearest2d(in_vulkan, {4, 6});

          const auto check = almostEqual(out_cpu, out_vulkan.cpu());
          if (!check) {
            showRtol(out_cpu, out_vulkan.cpu());
          }

          ASSERT_TRUE(check);

        */
    }

    pub enum OpType {
        addmm,
        conv2d,
        hardtanh_,
        mean,
    }

    pub trait BaseOpInterface:
    Run
    + ToString {}

    pub trait Run {

        fn run(&self, _0: &mut Tensor) -> Tensor;
    }

    pub trait ToString {

        fn to_string(&self) -> String;
    }


    pub struct Addmm {
        base:  BaseOp,
        m2:    Tensor,
        b:     Tensor,
        beta:  f32,
        alpha: f32,
    }

    impl Addmm {
        
        pub fn new(
            m1h:   i64,
            m1w:   i64,
            m2w:   i64,
            beta:  f32,
            alpha: f32) -> Self {
        
            todo!();
            /*


                : BaseOp(OpType::addmm),
              m2_(rand(IntArrayRef({m1W, m2W}), device(kCPU).dtype(kFloat))),
              b_(rand(IntArrayRef({m1H, m2W}), device(kCPU).dtype(kFloat))),
              beta_(beta),
              alpha_(alpha)
            */
        }
        
        pub fn run(&self, t: &mut Tensor) -> Tensor {
            
            todo!();
            /*
                if (t.is_vulkan()) {
              return addmm(b_, t, m2_, beta_, alpha_);
            }

            return addmm(b_, t, m2_, beta_, alpha_);
            */
        }
        
        pub fn to_string(&self) -> String {
            
            todo!();
            /*
                return "addmm";
            */
        }
    }

    pub struct Conv2d {
        base:    BaseOp,
        groups:  i64,
        stride:  i64,
        padding: i64,
        w:       Tensor,
        b:       Tensor,
    }

    impl Conv2d {

        pub fn new(
            wsizes:  &[i32],
            groups:  i64,
            stride:  i64,
            padding: i64) -> Self {
        
            todo!();
            /*


                : BaseOp(OpType::conv2d),
                groups_(groups),
                stride_(stride),
                padding_(padding),
                w_(rand(wsizes, device(kCPU).dtype(kFloat))),
                b_(rand(wsizes[0], device(kCPU).dtype(kFloat)))
            */
        }
        
        pub fn run(&self, t: &mut Tensor) -> Tensor {
            
            todo!();
            /*
                return conv2d(t, w_, b_, {stride_}, {padding_}, {1}, groups_);
            */
        }
        
        pub fn to_string(&self) -> String {
            
            todo!();
            /*
                return "conv2d";
            */
        }
    }

    pub struct Hardtanh_ {
        base: BaseOp,
    }

    impl Default for Hardtanh_ {
        
        fn default() -> Self {
            todo!();
            /*
            : base_op(OpType::hardtanh_),

            
            */
        }
    }

    impl Hardtanh_ {
        
        pub fn run(&self, input: &mut Tensor) -> Tensor {
            
            todo!();
            /*
                return hardtanh_(input, 0, 6);
            */
        }
        
        pub fn to_string(&self) -> String {
            
            todo!();
            /*
                return "hardtanh_";
            */
        }
    }


    pub struct Mean {
        base: BaseOp,
    }

    impl Default for Mean {
        
        fn default() -> Self {
            todo!();
            /*
            : base_op(OpType::mean),

            
            */
        }
    }

    impl Mean {
        
        pub fn run(&self, input: &mut Tensor) -> Tensor {
            
            todo!();
            /*
                return mean(input, {2, 3}, false);
            */
        }
        
        pub fn to_string(&self) -> String {
            
            todo!();
            /*
                return "mean";
            */
        }
    }

    pub struct OpsList {
        ops: Vec<Box<BaseOp>>,
    }

    impl OpsList {
        
        pub fn new(ops: Vec<Box<BaseOp>>) -> Self {
        
            todo!();
            /*


                : ops_(move(ops))
            */
        }
        
        pub fn run(&mut self, input: &Tensor) -> Auto {
            
            todo!();
            /*
                Tensor output = input;

            for (const auto& op : ops_) {
              output = op->run(output);
            }

            return output;
            */
        }
        
        pub fn run(&mut self, 
            input:   &Tensor,
            v_input: &Tensor) -> Auto {
            
            todo!();
            /*
                Tensor output = input;
            Tensor v_output = v_input;

            for (const auto& op : ops_) {
              output = op->run(output);
              v_output = op->run(v_output);
            }

            return make_pair(output, v_output);
            */
        }
    }

    pub struct MobileNetV2 {
        base: OpsList,
    }

    impl Default for MobileNetV2 {
        
        fn default() -> Self {
            todo!();
            /*


                ops_.emplace_back(new Conv2d({32, 3, 3, 3}, 1, 2, 1));
            ops_.emplace_back(new Hardtanh_());
            ops_.emplace_back(new Conv2d({32, 1, 3, 3}, 32, 1, 1));
            ops_.emplace_back(new Hardtanh_());
            ops_.emplace_back(new Conv2d({16, 32, 1, 1}, 1, 1, 0));
            ops_.emplace_back(new Conv2d({96, 16, 1, 1}, 1, 1, 0));
            ops_.emplace_back(new Hardtanh_());
            ops_.emplace_back(new Conv2d({96, 1, 3, 3}, 96, 2, 1));
            ops_.emplace_back(new Hardtanh_());
            ops_.emplace_back(new Conv2d({24, 96, 1, 1}, 1, 1, 0));
            ops_.emplace_back(new Conv2d({144, 24, 1, 1}, 1, 1, 0));
            ops_.emplace_back(new Hardtanh_());
            ops_.emplace_back(new Conv2d({144, 1, 3, 3}, 144, 1, 1));
            ops_.emplace_back(new Hardtanh_());
            ops_.emplace_back(new Conv2d({24, 144, 1, 1}, 1, 1, 0));
            ops_.emplace_back(new Conv2d({144, 24, 1, 1}, 1, 1, 0));
            ops_.emplace_back(new Hardtanh_());
            ops_.emplace_back(new Conv2d({144, 1, 3, 3}, 144, 2, 1));
            ops_.emplace_back(new Hardtanh_());
            ops_.emplace_back(new Conv2d({32, 144, 1, 1}, 1, 1, 0));
            ops_.emplace_back(new Conv2d({192, 32, 1, 1}, 1, 1, 0));
            ops_.emplace_back(new Hardtanh_());
            ops_.emplace_back(new Conv2d({192, 1, 3, 3}, 192, 1, 1));
            ops_.emplace_back(new Hardtanh_());
            ops_.emplace_back(new Conv2d({32, 192, 1, 1}, 1, 1, 0));
            ops_.emplace_back(new Conv2d({192, 32, 1, 1}, 1, 1, 0));
            ops_.emplace_back(new Hardtanh_());
            ops_.emplace_back(new Conv2d({192, 1, 3, 3}, 192, 1, 1));
            ops_.emplace_back(new Hardtanh_());
            ops_.emplace_back(new Conv2d({32, 192, 1, 1}, 1, 1, 0));
            ops_.emplace_back(new Conv2d({192, 32, 1, 1}, 1, 1, 0));
            ops_.emplace_back(new Hardtanh_());
            ops_.emplace_back(new Conv2d({192, 1, 3, 3}, 192, 2, 1));
            ops_.emplace_back(new Hardtanh_());
            ops_.emplace_back(new Conv2d({64, 192, 1, 1}, 1, 1, 0));
            ops_.emplace_back(new Conv2d({384, 64, 1, 1}, 1, 1, 0));
            ops_.emplace_back(new Hardtanh_());
            ops_.emplace_back(new Conv2d({384, 1, 3, 3}, 384, 1, 1));
            ops_.emplace_back(new Hardtanh_());
            ops_.emplace_back(new Conv2d({64, 384, 1, 1}, 1, 1, 0));
            ops_.emplace_back(new Conv2d({384, 64, 1, 1}, 1, 1, 0));
            ops_.emplace_back(new Hardtanh_());
            ops_.emplace_back(new Conv2d({384, 1, 3, 3}, 384, 1, 1));
            ops_.emplace_back(new Hardtanh_());
            ops_.emplace_back(new Conv2d({64, 384, 1, 1}, 1, 1, 0));
            ops_.emplace_back(new Conv2d({384, 64, 1, 1}, 1, 1, 0));
            ops_.emplace_back(new Hardtanh_());
            ops_.emplace_back(new Conv2d({384, 1, 3, 3}, 384, 1, 1));
            ops_.emplace_back(new Hardtanh_());
            ops_.emplace_back(new Conv2d({64, 384, 1, 1}, 1, 1, 0));
            ops_.emplace_back(new Conv2d({384, 64, 1, 1}, 1, 1, 0));
            ops_.emplace_back(new Hardtanh_());
            ops_.emplace_back(new Conv2d({384, 1, 3, 3}, 384, 1, 1));
            ops_.emplace_back(new Hardtanh_());
            ops_.emplace_back(new Conv2d({96, 384, 1, 1}, 1, 1, 0));
            ops_.emplace_back(new Conv2d({576, 96, 1, 1}, 1, 1, 0));
            ops_.emplace_back(new Hardtanh_());
            ops_.emplace_back(new Conv2d({576, 1, 3, 3}, 576, 1, 1));
            ops_.emplace_back(new Hardtanh_());
            ops_.emplace_back(new Conv2d({96, 576, 1, 1}, 1, 1, 0));
            ops_.emplace_back(new Conv2d({576, 96, 1, 1}, 1, 1, 0));
            ops_.emplace_back(new Hardtanh_());
            ops_.emplace_back(new Conv2d({576, 1, 3, 3}, 576, 1, 1));
            ops_.emplace_back(new Hardtanh_());
            ops_.emplace_back(new Conv2d({96, 576, 1, 1}, 1, 1, 0));
            ops_.emplace_back(new Conv2d({576, 96, 1, 1}, 1, 1, 0));
            ops_.emplace_back(new Hardtanh_());
            ops_.emplace_back(new Conv2d({576, 1, 3, 3}, 576, 2, 1));
            ops_.emplace_back(new Hardtanh_());
            ops_.emplace_back(new Conv2d({160, 576, 1, 1}, 1, 1, 0));
            ops_.emplace_back(new Conv2d({960, 160, 1, 1}, 1, 1, 0));
            ops_.emplace_back(new Hardtanh_());
            ops_.emplace_back(new Conv2d({960, 1, 3, 3}, 960, 1, 1));
            ops_.emplace_back(new Hardtanh_());
            ops_.emplace_back(new Conv2d({160, 960, 1, 1}, 1, 1, 0));
            ops_.emplace_back(new Conv2d({960, 160, 1, 1}, 1, 1, 0));
            ops_.emplace_back(new Hardtanh_());
            ops_.emplace_back(new Conv2d({960, 1, 3, 3}, 960, 1, 1));
            ops_.emplace_back(new Hardtanh_());
            ops_.emplace_back(new Conv2d({160, 960, 1, 1}, 1, 1, 0));
            ops_.emplace_back(new Conv2d({960, 160, 1, 1}, 1, 1, 0));
            ops_.emplace_back(new Hardtanh_());
            ops_.emplace_back(new Conv2d({960, 1, 3, 3}, 960, 1, 1));
            ops_.emplace_back(new Hardtanh_());
            ops_.emplace_back(new Conv2d({320, 960, 1, 1}, 1, 1, 0));
            ops_.emplace_back(new Conv2d({1280, 320, 1, 1}, 1, 1, 0));
            ops_.emplace_back(new Hardtanh_());
            ops_.emplace_back(new Mean());
            ops_.emplace_back(new Addmm(1, 1280, 1000, 0, 1));
            */
        }
    }

    #[test] fn vulkan_api_test_mobilenetv2() {
        todo!();
        /*
        
          if (!is_vulkan_available()) {
            return;
          }
          InferenceMode mode;

          MobileNetV2 mn2;

          const auto input = rand({1, 3, 224, 224}, device(kCPU).dtype(kFloat));
          const auto output = mn2.run(input, input.vulkan());

          const auto check = almostEqual(output.first, output.second.cpu());
          if (!check) {
            showRtol(output.first, output.second.cpu());
          }

          ASSERT_TRUE(check);

        */
    }
}
