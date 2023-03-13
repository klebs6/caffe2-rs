crate::ix!();

#[test] fn roi_align_test_check_cpugpu_equal() {
    todo!();
    /*
      if (!caffe2::HasCudaGPU())
        return;

      Tensor y_cpu(CPU);
      Tensor y_gpu(CPU);
      Tensor y_cpu_nhwc(CPU);

      // tests using FAIR example
      {
        TestParams test_params;
        CreateAndRun<CPUContext>(&y_cpu, "NCHW", test_params, false);
        CreateAndRun<CUDAContext>(&y_gpu, "NCHW", test_params, false);
        CreateAndRun<CPUContext>(&y_cpu_nhwc, "NHWC", test_params, false);

        EXPECT_EQ(y_cpu.sizes(), y_gpu.sizes());
        EXPECT_EQ(y_cpu.sizes(), y_cpu_nhwc.sizes());
        ConstEigenVectorMap<float> y_cpu_vec(y_cpu.data<float>(), y_cpu.numel());
        ConstEigenVectorMap<float> y_gpu_vec(y_gpu.data<float>(), y_gpu.numel());
        ConstEigenVectorMap<float> y_cpu_nhwc_vec(
            y_cpu_nhwc.data<float>(), y_cpu_nhwc.numel());
        int max_diff_idx = -1;
        (y_cpu_vec - y_gpu_vec).cwiseAbs().maxCoeff(&max_diff_idx);
        EXPECT_FLOAT_EQ(y_cpu_vec[max_diff_idx], y_gpu_vec[max_diff_idx]);

        max_diff_idx = -1;
        (y_cpu_vec - y_cpu_nhwc_vec).cwiseAbs().maxCoeff(&max_diff_idx);
        EXPECT_FLOAT_EQ(y_cpu_vec[max_diff_idx], y_cpu_nhwc_vec[max_diff_idx]);
      }

      // random tests
      const int random_test_numbers = 100;
      for (int i = 0; i < random_test_numbers; i++) {
        const int N = randInt(1, 5);
        const int C = randInt(1, 5);
        const int H = randInt(1, 50);
        const int W = randInt(1, 50);
        const int n_rois = randInt(1, 30);
        vector<float> rois_array;
        for (int n = 0; n < n_rois; n++) {
          rois_array.push_back(randInt(0, N - 1));
          int w1 = randInt(-20, W + 20);
          int w2 = randInt(-20, W + 20);
          int h1 = randInt(-20, H + 20);
          int h2 = randInt(-20, H + 20);
          rois_array.push_back(std::min(w1, w2));
          rois_array.push_back(std::max(h1, h2));
          rois_array.push_back(std::min(w1, w2));
          rois_array.push_back(std::max(h1, h2));
        }
        TestParams test_params{N, C, H, W, n_rois, rois_array};

        CreateAndRun<CPUContext>(&y_cpu, "NCHW", test_params, true);
        CreateAndRun<CUDAContext>(&y_gpu, "NCHW", test_params, true);
        CreateAndRun<CPUContext>(&y_cpu_nhwc, "NHWC", test_params, true);

        EXPECT_EQ(y_cpu.sizes(), y_gpu.sizes());
        EXPECT_EQ(y_cpu.sizes(), y_cpu_nhwc.sizes());
        ConstEigenVectorMap<float> y_cpu_vec(y_cpu.data<float>(), y_cpu.numel());
        ConstEigenVectorMap<float> y_gpu_vec(y_gpu.data<float>(), y_gpu.numel());
        ConstEigenVectorMap<float> y_cpu_nhwc_vec(
            y_cpu_nhwc.data<float>(), y_cpu_nhwc.numel());
        int max_diff_idx = -1;
        (y_cpu_vec - y_gpu_vec).cwiseAbs().maxCoeff(&max_diff_idx);
        EXPECT_NEAR(y_cpu_vec[max_diff_idx], y_gpu_vec[max_diff_idx], 1e-1);

        max_diff_idx = -1;
        (y_cpu_vec - y_cpu_nhwc_vec).cwiseAbs().maxCoeff(&max_diff_idx);
        EXPECT_FLOAT_EQ(y_cpu_vec[max_diff_idx], y_cpu_nhwc_vec[max_diff_idx]);
      }
  */
}
