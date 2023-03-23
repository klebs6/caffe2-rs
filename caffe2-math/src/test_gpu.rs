crate::ix!();

#[test] fn math_util_gpu_test_add_striped_batch() {
    todo!();
    /*
      if (!HasCudaGPU())
        return;
      Workspace ws;
      DeviceOption option;
      option.set_device_type(PROTO_CUDA);
      CUDAContext context(option);
      Blob* blobx = ws.CreateBlob("X");
      Blob* bloby = ws.CreateBlob("Y");
      Blob* bloby_host = ws.CreateBlob("Y_host");

      vector<int> shapex{33 * 9, 25};
      vector<int> shapey{33, 25};

      auto* tensorx = BlobGetMutableTensor(blobx, CUDA);
      tensorx->Resize(shapex);
      int stripe = 33 * 25;
      vector<float> tot(33, 0.0);
      for (int j = 0; j < 9; j++) {
        // Have different values for each line
        for (int k = 0; k < 33; k++) {
          math::Set<float, CUDAContext>(
              33,
              1.0 + j + k,
              tensorx->mutable_data<float>() + j * stripe + k * 25,
              &context);
          tot[k] += 1.0 + j + k;
        }
      }

      auto* tensory = BlobGetMutableTensor(bloby, CUDA);
      tensory->Resize(shapey);
      math::Set<float, CUDAContext>(
          stripe, 0.0, tensory->mutable_data<float>(), &context);

      math::AddStripedBatch<float, CUDAContext>(
          stripe,
          tensorx->template data<float>(),
          tensory->mutable_data<float>(),
          stripe,
          9,
          &context);
      context.FinishDeviceComputation();

      // Copy result to CPU so we can inspect it
      auto* tensory_host = BlobGetMutableTensor(bloby_host, CPU);
      tensory_host->CopyFrom(*tensory);

      for (int k = 0; k < 33; k++) {
        for (int i = 0; i < 25; i++) {
          EXPECT_EQ(tensory_host->data<float>()[k * 25 + i], tot[k]);
        }
      }
  */
}

#[test] fn math_util_gpu_test_reduce_min() {
    todo!();
    /*
      executeGpuBinaryOpTest(
          6,
          1,
          1,
          [](int /*i*/) { return 11.0f; },
          [](int /*i*/) { return 0.0f; },
          [](int N0,
             int /*N1*/,
             const float* src0,
             const float* /*src1*/,
             float* dst,
             CUDAContext* context) {
            Tensor aux(CUDA);
            math::ReduceMin<float, CUDAContext>(N0, src0, dst, &aux, context);
          },
          [](int /*i*/) { return 11.0f; });
      executeGpuBinaryOpTest(
          6,
          1,
          1,
          [](int i) { return i == 3 ? 11.0f : 17.0f; },
          [](int /*i*/) { return 0.0f; },
          [](int N0,
             int /*N1*/,
             const float* src0,
             const float* /*src1*/,
             float* dst,
             CUDAContext* context) {
            Tensor aux(CUDA);
            math::ReduceMin<float, CUDAContext>(N0, src0, dst, &aux, context);
          },
          [](int /*i*/) { return 11.0f; });
  */
}

#[test] fn math_util_gpu_test_reduce_max() {
    todo!();
    /*
      executeGpuBinaryOpTest(
          6,
          1,
          1,
          [](int /*i*/) { return 11.0f; },
          [](int /*i*/) { return 0.0f; },
          [](int N0,
             int /*N1*/,
             const float* src0,
             const float* /*src1*/,
             float* dst,
             CUDAContext* context) {
            Tensor aux(CUDA);
            math::ReduceMax<float, CUDAContext>(N0, src0, dst, &aux, context);
          },
          [](int /*i*/) { return 11.0f; });
      executeGpuBinaryOpTest(
          6,
          1,
          1,
          [](int i) { return i == 3 ? 17.0f : 11.0f; },
          [](int /*i*/) { return 0.0f; },
          [](int N0,
             int /*N1*/,
             const float* src0,
             const float* /*src1*/,
             float* dst,
             CUDAContext* context) {
            Tensor aux(CUDA);
            math::ReduceMax<float, CUDAContext>(N0, src0, dst, &aux, context);
          },
          [](int /*i*/) { return 17.0f; });
  */
}

#[test] fn math_util_gpu_test_copy_vector() {
    todo!();
    /*
      executeGpuBinaryOpTest(
          6,
          1,
          6,
          [](int i) { return 5.0f - i; },
          [](int /*i*/) { return 0.0f; },
          [](int N0,
             int /*N1*/,
             const float* src0,
             const float* /*src1*/,
             float* dst,
             CUDAContext* context) {
            math::CopyVector<float, CUDAContext>(N0, src0, dst, context);
          },
          [](int i) { return 5.0f - i; });
  */
}

#[test] fn gemm_batched_gpu_test_gemm_batched_gpu_float_test() {
    todo!();
    /*
      if (!HasCudaGPU()) {
        return;
      }
      RunGemmBatched(1.0f, 0.0f);
      VerifyOutput(10.0f);
      RunGemmBatched(1.0f, 0.5f);
      VerifyOutput(15.0f);
      RunGemmBatched(0.5f, 1.0f);
      VerifyOutput(20.0f);
  */
}

#[test] fn gemm_batched_gpu_test_gemm_strided_batched_gpu_float_test() {
    todo!();
    /*
      if (!HasCudaGPU()) {
        return;
      }
      RunGemmStridedBatched(1.0f, 0.0f);
      VerifyOutput(10.0f);
      RunGemmStridedBatched(1.0f, 0.5f);
      VerifyOutput(15.0f);
      RunGemmStridedBatched(0.5f, 1.0f);
      VerifyOutput(20.0f);
  */
}
