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

