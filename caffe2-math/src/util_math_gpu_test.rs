crate::ix!();

#[inline] pub fn execute_gpu_binary_op_test(
    shapex0:        i32,
    shapex1:        i32,
    shapey:         i32,
    input0:         fn(unnamed_0: i32) -> f32,
    input1:         fn(unnamed_0: i32) -> f32,
    operation:      fn(
        n0:      i32,
        n1:      i32,
        src0:    *const f32,
        src1:    *const f32,
        dst:     *mut f32,
        context: *mut CUDAContext
    ) -> (),
    correct_output: fn(unnamed_0: i32) -> f32)  {

    todo!();
    /*
        if (!HasCudaGPU())
        return;
      Workspace ws;
      DeviceOption option;
      option.set_device_type(PROTO_CUDA);
      CUDAContext context(option);

      Blob* blobx0 = ws.CreateBlob("X0");
      Blob* blobx1 = ws.CreateBlob("X1");
      Blob* bloby = ws.CreateBlob("Y");
      Blob* bloby_host = ws.CreateBlob("Y_host");

      auto* tensorx0 = BlobGetMutableTensor(blobx0, CUDA);
      auto* tensorx1 = BlobGetMutableTensor(blobx1, CUDA);
      auto* tensory = BlobGetMutableTensor(bloby, CUDA);

      vector<int> shapex0_vector{shapex0};
      vector<int> shapex1_vector{shapex1};
      vector<int> shapey_vector{shapey};

      tensorx0->Resize(shapex0_vector);
      tensorx1->Resize(shapex1_vector);
      tensory->Resize(shapey_vector);

      for (int i = 0; i < shapex0; i++) {
        math::Set<float, CUDAContext>(
            1, input0(i), tensorx0->mutable_data<float>() + i, &context);
      }
      for (int i = 0; i < shapex1; i++) {
        math::Set<float, CUDAContext>(
            1, input1(i), tensorx1->mutable_data<float>() + i, &context);
      }
      operation(
          shapex0,
          shapex1,
          tensorx0->template data<float>(),
          tensorx1->template data<float>(),
          tensory->mutable_data<float>(),
          &context);
      context.FinishDeviceComputation();

      // Copy result to CPU so we can inspect it
      auto* tensory_host = BlobGetMutableTensor(bloby_host, CPU);
      tensory_host->CopyFrom(*tensory);

      for (int i = 0; i < shapey; ++i) {
        EXPECT_EQ(tensory_host->data<float>()[i], correct_output(i));
      }
    */
}


#[test] fn MathUtilGPUTest_testAddStripedBatch() {
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


#[test] fn MathUtilGPUTest_testReduceMin() {
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


#[test] fn MathUtilGPUTest_testReduceMax() {
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


#[test] fn MathUtilGPUTest_testCopyVector() {
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

pub const kEps: f32 = 1e-5;

pub struct GemmBatchedGPUTest {
    /*
       : public testing::TestWithParam<testing::tuple<bool, bool>> {
       */

    ws:            Workspace,
    option:        DeviceOption,
    cuda_context:  Box<CUDAContext>,
    x:             *mut Tensor, // default = nullptr
    w:             *mut Tensor, // default = nullptr
    y:             *mut Tensor, // default = nullptr
    trans_X:       bool,
    trans_W:       bool,
}

impl GemmBatchedGPUTest {

    #[inline] pub fn set_up(&mut self)  {

        todo!();
        /*
            if (!HasCudaGPU()) {
          return;
        }
        option_.set_device_type(PROTO_CUDA);
        cuda_context_ = make_unique<CUDAContext>(option_);
        Blob* X_blob = ws_.CreateBlob("X");
        Blob* W_blob = ws_.CreateBlob("W");
        Blob* Y_blob = ws_.CreateBlob("Y");
        X_ = BlobGetMutableTensor(X_blob, CUDA);
        W_ = BlobGetMutableTensor(W_blob, CUDA);
        Y_ = BlobGetMutableTensor(Y_blob, CUDA);
        X_->Resize(std::vector<int64_t>{3, 5, 10});
        W_->Resize(std::vector<int64_t>{3, 6, 10});
        Y_->Resize(std::vector<int64_t>{3, 5, 6});
        math::Set<float, CUDAContext>(
            X_->numel(), 1.0f, X_->mutable_data<float>(), cuda_context_.get());
        math::Set<float, CUDAContext>(
            W_->numel(), 1.0f, W_->mutable_data<float>(), cuda_context_.get());
        trans_X_ = std::get<0>(GetParam());
        trans_W_ = std::get<1>(GetParam());
        */
    }
    
    #[inline] pub fn run_gemm_batched(&mut self, alpha: f32, beta: f32)  {
        
        todo!();
        /*
            const float* X_data = X_->template data<float>();
        const float* W_data = W_->template data<float>();
        float* Y_data = Y_->template mutable_data<float>();
        const int X_stride = 5 * 10;
        const int W_stride = 6 * 10;
        const int Y_stride = 5 * 6;
        std::array<const float*, 3> X_array = {
            X_data, X_data + X_stride, X_data + 2 * X_stride};
        std::array<const float*, 3> W_array = {
            W_data, W_data + W_stride, W_data + 2 * W_stride};
        std::array<float*, 3> Y_array = {
            Y_data, Y_data + Y_stride, Y_data + 2 * Y_stride};
        math::GemmBatched<float, CUDAContext>(
            trans_X_ ? CblasTrans : CblasNoTrans,
            trans_W_ ? CblasTrans : CblasNoTrans,
            3,
            5,
            6,
            10,
            alpha,
            X_array.data(),
            W_array.data(),
            beta,
            Y_array.data(),
            cuda_context_.get());
        */
    }
    
    #[inline] pub fn run_gemm_strided_batched(&mut self, alpha: f32, beta: f32)  {
        
        todo!();
        /*
            const float* X_data = X_->template data<float>();
        const float* W_data = W_->template data<float>();
        float* Y_data = Y_->template mutable_data<float>();
        const int X_stride = 5 * 10;
        const int W_stride = 6 * 10;
        const int Y_stride = 5 * 6;
        math::GemmStridedBatched<float, CUDAContext>(
            trans_X_ ? CblasTrans : CblasNoTrans,
            trans_W_ ? CblasTrans : CblasNoTrans,
            3,
            5,
            6,
            10,
            alpha,
            X_data,
            X_stride,
            W_data,
            W_stride,
            beta,
            Y_data,
            Y_stride,
            cuda_context_.get());
        */
    }
    
    #[inline] pub fn verify_output(&self, value: f32)  {
        
        todo!();
        /*
            Tensor Y_cpu(*Y_, CPU);
        for (int i = 0; i < Y_cpu.numel(); ++i) {
          EXPECT_FLOAT_EQ(value, Y_cpu.template data<float>()[i]);
        }
        */
    }
}


#[test] fn GemmBatchedGPUTest_GemmBatchedGPUFloatTest() {
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


#[test] fn GemmBatchedGPUTest_GemmStridedBatchedGPUFloatTest() {
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

instantiate_test_case_p!{
    /*
    GemmBatchedGPUTrans,
    GemmBatchedGPUTest,
    testing::Combine(testing::Bool(), testing::Bool())
    */
}

pub struct BroadcastGPUTest {
    ws:           Workspace,
    option:       DeviceOption,
    cuda_context: Box<CUDAContext>,
    x:            *mut Tensor, // default = nullptr
    y:            *mut Tensor, // default = nullptr
}

impl BroadcastGPUTest {

    #[inline] fn set_up_data(&mut self, 
        x_dims: &Vec<i32>,
        y_dims: &Vec<i32>,
        x_data: &Vec<f32>)  {
        
        todo!();
        /*
            X_->Resize(X_dims);
        Y_->Resize(Y_dims);
        ASSERT_EQ(X_data.size(), X_->numel());
        cuda_context_->CopyFromCPU<float>(
            X_data.size(), X_data.data(), X_->mutable_data<float>());
        */
    }
    
    #[inline] fn verify_result(&mut self, expected_output: &Vec<f32>)  {
        
        todo!();
        /*
            Blob* blob_y_host = ws_.CreateBlob("Y_host");
        auto* Y_host = BlobGetMutableTensor(blob_y_host, CPU);
        Y_host->CopyFrom(*Y_);
        ASSERT_EQ(expected_output.size(), Y_host->numel());
        for (std::size_t i = 0; i < expected_output.size(); ++i) {
          EXPECT_FLOAT_EQ(expected_output[i], Y_host->data<float>()[i]);
        }
        */
    }
    
    #[inline] fn run_broadcast_test(&mut self, 
        x_dims: &Vec<i32>,
        y_dims: &Vec<i32>,
        x_data: &Vec<f32>,
        y_data: &Vec<f32>)  {

        todo!();
        /*
            SetUpData(X_dims, Y_dims, X_data);
        math::Broadcast<float, CUDAContext>(
            X_dims.size(),
            X_dims.data(),
            Y_dims.size(),
            Y_dims.data(),
            1.0f,
            X_->data<float>(),
            Y_->mutable_data<float>(),
            cuda_context_.get());
        VerifyResult(Y_data);
        */
    }
}

impl TestContext for BroadcastGPUTest {

    #[inline] fn setup() -> Self {
        
        todo!();
        /*
            if (!HasCudaGPU()) {
          return;
        }
        option_.set_device_type(PROTO_CUDA);
        cuda_context_ = make_unique<CUDAContext>(option_);
        Blob* blob_x = ws_.CreateBlob("X");
        Blob* blob_y = ws_.CreateBlob("Y");
        X_ = BlobGetMutableTensor(blob_x, CUDA);
        Y_ = BlobGetMutableTensor(blob_y, CUDA);
        */
    }
    
}

#[test] fn BroadcastGPUTest_BroadcastGPUFloatTest() {
    todo!();
    /*
      if (!HasCudaGPU()) {
        return;
      }
      RunBroadcastTest({2}, {2}, {1.0f, 2.0f}, {1.0f, 2.0f});
      RunBroadcastTest({1}, {2}, {1.0f}, {1.0f, 1.0f});
      RunBroadcastTest({1}, {2, 2}, {1.0f}, {1.0f, 1.0f, 1.0f, 1.0f});
      RunBroadcastTest({2, 1}, {2, 2}, {1.0f, 2.0f}, {1.0f, 1.0f, 2.0f, 2.0f});
      RunBroadcastTest(
          {2, 1},
          {2, 2, 2},
          {1.0f, 2.0f},
          {1.0f, 1.0f, 2.0f, 2.0f, 1.0f, 1.0f, 2.0f, 2.0f});
  */
}
