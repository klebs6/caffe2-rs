crate::ix!();

/**
  | Add the vector as an input to a Workspace
  | depending on the context of the workspace
  |
  */
#[inline] pub fn add_inputCPU<T>(
    shape:  &Vec<i64>,
    values: &Vec<T>,
    name:   &String,
    ws:     *mut Workspace) {
    todo!();
    /*
        Blob* blob = ws->CreateBlob(name);
      auto* tensor = BlobGetMutableTensor(blob, CPU);
      tensor->Resize(shape);
      EigenVectorMap<T> tensor_vec(tensor->mutable_data<T>(), tensor->numel());
      tensor_vec.array() = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>{
          values.data(), static_cast<int>(values.size())};
    */
}

#[inline] pub fn add_inputGPU<T>(
    shape:  &Vec<i64>,
    values: &Vec<T>,
    name:   &String,
    ws:     *mut Workspace) {
    todo!();
    /*
        Tensor tmp(shape, CPU);
      EigenVectorMap<T> tmp_vec(tmp.mutable_data<T>(), tmp.numel());
      tmp_vec.array() = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>{
          values.data(), static_cast<int>(values.size())};

      Blob* blob = ws->CreateBlob(name);
      auto* tensor = BlobGetMutableTensor(blob, CUDA);
      tensor->CopyFrom(tmp);
    */
}


#[inline] pub fn get_device_type<Context>() -> DeviceTypeProto {
    todo!();
    /*
        return PROTO_CPU;
        CUDAContext => PROTO_CUDA
    */

}

/**
  | Create a BatchPermutationOp with the
  | given inputs (actual values are generated
  | sequentially) and run it
  |
  */
#[inline] pub fn create_and_run<Context>(
    out_result: *mut TensorCPU,
    n:          i32,
    shape:      &mut Vec<i64>,
    features:   &mut Vec<f32>,
    indices:    Vec<i32>) 
{
    todo!();
    /*
        Workspace ws;

      AddInput<float, Context>(shape, features, "X", &ws);
      AddInput<int, Context>(vector<int64_t>{N}, indices, "indices", &ws);

      OperatorDef def;
      def.set_name("test");
      def.set_type("BatchPermutation");
      def.add_input("X");
      def.add_input("indices");
      def.add_output("Y");
      def.mutable_device_option()->set_device_type(GetDeviceType<Context>());
      unique_ptr<OperatorStorage> op = CreateOperator(def, &ws);

      EXPECT_NE(nullptr, op.get());
      EXPECT_TRUE(op->Run());

      Blob* Y_blob = ws.GetBlob("Y");
      EXPECT_NE(nullptr, Y_blob);

      auto& Y = Y_blob->Get<Tensor>();
      outResult->CopyFrom(Y);
    */
}

/**
  | Create a BatchPermutationOp with the
  | given inputs (actual values are generated
  | sequentially) and run it
  |
  */
#[inline] pub fn create_and_run_gradient<Context>(
    out_result: *mut TensorCPU,
    n:          i32,
    shape:      &mut Vec<i64>,
    features:   &mut Vec<f32>,
    indices:    Vec<i32>) 
{
    todo!();
    /*
        Workspace ws;

      AddInput<float, Context>(shape, features, "dY", &ws);
      AddInput<int, Context>(vector<int64_t>{N}, indices, "indices", &ws);

      OperatorDef def;
      def.set_name("test");
      def.set_type("BatchPermutationGradient");
      def.add_input("indices");
      def.add_input("dY");
      def.add_output("dX");
      def.mutable_device_option()->set_device_type(GetDeviceType<Context>());
      unique_ptr<OperatorStorage> op = CreateOperator(def, &ws);

      EXPECT_NE(nullptr, op.get());
      EXPECT_TRUE(op->Run());

      Blob* Y_blob = ws.GetBlob("dX");
      EXPECT_NE(nullptr, Y_blob);

      auto& Y = Y_blob->Get<Tensor>();
      outResult->CopyFrom(Y);
    */
}

/**
  | Check that the CPU and GPU implementations
  | provide the exact same results
  |
  */
#[inline] pub fn check_cpugpuequal(
    shape: Vec<i64>,
    indices: Vec<i32>)  
{
    todo!();
    /*
        // Prepare input data
      EXPECT_GT(shape.size(), 1);
      int N = shape[0];
      int input_size = 1;
      for (auto k : shape) {
        input_size *= k;
      }
      int K = N ? input_size / N : 0;
      vector<float> features(input_size);
      std::iota(features.begin(), features.end(), 0);

      // CPU outputs
      Tensor y_cpu{CPU};
      Tensor y_cpu_grad{CPU};

      // CPU BatchPermutation
      CreateAndRun<CPUContext>(&y_cpu, N, shape, features, indices);

      // CPU BatchPermutationGradient
      CreateAndRunGradient<CPUContext>(&y_cpu_grad, N, shape, features, indices);

      // Check CPU output values
      for (auto i = 0; i < indices.size(); ++i) {
        for (auto k = 0; k < K; ++k) {
          EXPECT_NEAR(
              y_cpu.data<float>()[indices[i] * K + k], features[i * K + k], 1e4);
          EXPECT_NEAR(
              y_cpu_grad.data<float>()[i * K + k],
              features[indices[i] * K + k],
              1e4);
        }
      }

      if (!caffe2::HasCudaGPU()) {
        VLOG(2) << "No CudaGPU found. Skip GPU test." << std::endl;
        return;
      }

      // GPU outputs
      Tensor y_gpu{CPU};
      Tensor y_gpu_grad{CPU};

      // GPU BatchPermutation
      CreateAndRun<CPUContext>(&y_gpu, N, shape, features, indices);

      // Compare CPU and GPU BatchPermutation outputs
      EXPECT_EQ(y_cpu.sizes(), y_gpu.sizes());
      ConstEigenVectorMap<float> y_cpu_vec(y_cpu.data<float>(), y_cpu.numel());
      ConstEigenVectorMap<float> y_gpu_vec(y_gpu.data<float>(), y_gpu.numel());
      EXPECT_TRUE(y_cpu_vec.isApprox(y_gpu_vec));

      // GPU BatchPermutationGradient
      CreateAndRunGradient<CUDAContext>(&y_gpu_grad, N, shape, features, indices);

      // Check GPU outputs
      for (auto i = 0; i < indices.size(); ++i) {
        for (auto k = 0; k < K; ++k) {
          EXPECT_NEAR(
              y_gpu.data<float>()[indices[i] * K + k], features[i * K + k], 1e4);
          EXPECT_NEAR(
              y_gpu_grad.data<float>()[i * K + k],
              features[indices[i] * K + k],
              1e4);
        }
      }

      // Compare CPU and GPU BatchPermutationGradient outputs
      EXPECT_EQ(y_cpu_grad.sizes(), y_gpu_grad.sizes());
      ConstEigenVectorMap<float> y_cpu_vec_grad(
          y_cpu_grad.data<float>(), y_cpu_grad.numel());
      ConstEigenVectorMap<float> y_gpu_vec_grad(
          y_gpu_grad.data<float>(), y_gpu_grad.numel());
      EXPECT_TRUE(y_cpu_vec_grad.isApprox(y_gpu_vec_grad));
    */
}

#[test] fn batch_permutation_test_checkcpugpu_equal_generic_dimension() {
    todo!();
    /*
  auto t0 = std::chrono::high_resolution_clock::now();
  int batch_size = 8;
  int max_dimension = 6;
  vector<int64_t> shape = vector<int64_t>{batch_size};

  auto seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);

  for (int i = 2; i < max_dimension; ++i) {
    std::uniform_int_distribution<> dis(1, i);
    shape.push_back(dis(generator));
    CheckCPUGPUEqual(shape, vector<int>{0, 1, 2, 3, 4, 5, 6, 7});
    CheckCPUGPUEqual(shape, vector<int>{7, 6, 5, 4, 3, 2, 1, 0});
    CheckCPUGPUEqual(shape, vector<int>{1, 3, 5, 7, 0, 2, 4, 6});
    CheckCPUGPUEqual(shape, vector<int>{4, 5, 6, 7, 0, 1, 2, 3});
    CheckCPUGPUEqual(shape, vector<int>{3, 1, 5, 7, 6, 2, 4, 0});
  }
  CheckCPUGPUEqual({0, 128}, vector<int>{});
  auto t1 = std::chrono::high_resolution_clock::now();
  double elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  VLOG(2) << "Time elapsed: " << elapsed << " ms" << std::endl;
  return;
  */
}
