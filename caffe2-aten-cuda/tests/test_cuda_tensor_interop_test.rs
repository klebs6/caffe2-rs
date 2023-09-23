crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/test/cuda_tensor_interop_test.cpp]

/**
  | dumbest possible copies
  |
  */
pub fn cuda_get<T>(addr: *mut T) -> T {

    todo!();
        /*
            T result;
      CUDA_ENFORCE(cudaMemcpy(&result, addr, sizeof(T), cudaMemcpyDefault));
      return result;
        */
}

pub fn cuda_set<T>(
    addr:  *mut T,
    value: T)  {

    todo!();
        /*
            CUDA_ENFORCE(cudaMemcpy(addr, &value, sizeof(T), cudaMemcpyDefault));
        */
}

#[test] fn cuda_caffe_2to_pytorch_simple_legacy() {
    todo!();
    /*
    
      if (!is_available()) return;
      Tensor c2_tensor(CUDA);
      c2_tensor.Resize(4, 4);
      auto data = c2_tensor.mutable_data<i64>();
      {
        CUDAContext context;
        math::Set<i64>(16, 777, data, &context);
      }
      Tensor at_tensor(c2_tensor);
      ASSERT_TRUE(at_tensor.is_cuda());

      auto at_cpu = at_tensor.cpu();
      auto it = at_cpu.data_ptr<i64>();
      for (i64 i = 0; i < 16; i++) {
        ASSERT_EQ(it[i], 777);
      }

    */
}

#[test] fn cuda_caffe_2to_pytorch_simple() {
    todo!();
    /*
    
      if (!is_available()) return;
      Tensor c2_tensor =
          empty({4, 4}, dtype<i64>().device(CUDA));
      auto data = c2_tensor.mutable_data<i64>();
      {
        CUDAContext context;
        math::Set<i64>(16, 777, data, &context);
      }
      Tensor at_tensor(c2_tensor);
      ASSERT_TRUE(at_tensor.is_cuda());

      auto at_cpu = at_tensor.cpu();
      auto it = at_cpu.data_ptr<i64>();
      for (i64 i = 0; i < 16; i++) {
        ASSERT_EQ(it[i], 777);
      }

    */
}

#[test] fn cuda_caffe_2to_pytorch_op() {
    todo!();
    /*
    
      if (!is_available()) return;
      Tensor c2_tensor =
          empty({3, 3}, dtype<i64>().device(CUDA));
      auto data = c2_tensor.mutable_data<i64>();
      {
        CUDAContext context;
        math::Set<i64>(9, 111, data, &context);
      }
      Tensor at_tensor(c2_tensor);
      ASSERT_TRUE(at_tensor.is_cuda());

      ASSERT_EQ(sum(at_tensor).item<i64>(), 999);

    */
}

#[test] fn cuda_pytorch_to_caffe2_op() {
    todo!();
    /*
    
      if (!is_available()) return;
      Workspace workspace;
      NetDef net;

      auto at_tensor_a = ones({5, 5}, dtype(kFloat).device(kCUDA));
      auto at_tensor_b = ones({5, 5}, dtype(kFloat).device(kCUDA));
      auto at_tensor_c = ones({5, 5}, dtype(kFloat).device(kCUDA));

      auto* c2_tensor_a = BlobSetTensor(workspace.CreateBlob("a"), Tensor(at_tensor_a));
      auto* c2_tensor_b = BlobSetTensor(workspace.CreateBlob("b"), Tensor(at_tensor_b));

      // Test Alias
      {
        Tensor c2_tensor_from_aten(at_tensor_c);
        BlobSetTensor(workspace.CreateBlob("c"), c2_tensor_from_aten.Alias());
      }

      {
        auto op = net.add_op();
        op->set_type("Sum");
        op->add_input("a");
        op->add_input("b");
        op->add_input("c");
        op->add_output("d");
        op->mutable_device_option()->set_device_type(PROTO_CUDA);
      }

      workspace.RunNetOnce(net);

      const auto& result = workspace.GetBlob("d")->Get<Tensor>();
      ASSERT_EQ(result.GetDeviceType(), CUDA);

      auto data = result.data<float>();
      for (i64 i = 0; i < 25; i++) {
        ASSERT_EQ(cuda_get(data + i), 3.0);
      }
      Tensor at_result(result);
      ASSERT_TRUE(at_result.is_cuda());
      ASSERT_EQ(sum(at_result).item<float>(), 75);

    */
}

#[test] fn cuda_pytorch_to_caffe2_shared_storage_write() {
    todo!();
    /*
    
      if (!is_available()) return;
      auto at_tensor_a = ones({5, 5}, dtype(kFloat).device(kCUDA));
      auto at_tensor_b = at_tensor_a.view({25});

      Tensor c2_tensor_a(at_tensor_a);
      Tensor c2_tensor_b(at_tensor_b);

      // change is visible everywhere
      cuda_set<float>(c2_tensor_a.mutable_data<float>() + 1, 123);
      ASSERT_EQ(cuda_get(c2_tensor_b.mutable_data<float>() + 1), 123);
      ASSERT_EQ(at_tensor_a[0][1].item().to<float>(), 123);
      ASSERT_EQ(at_tensor_b[1].item().to<float>(), 123);

    */
}

#[test] fn cuda_pytorch_to_caffe2_mutual_resizes() {
    todo!();
    /*
    
      if (!is_available()) return;
      auto at_tensor = ones({5, 5}, dtype(kFloat).device(kCUDA));

      Tensor c2_tensor(at_tensor);

      // change is visible
      cuda_set<float>(c2_tensor.mutable_data<float>(), 123);
      ASSERT_EQ(at_tensor[0][0].item().to<float>(), 123);

      // resize PT tensor in smaller direction - storage is preserved
      at_tensor.resize_({4, 4});
      cuda_set<float>(c2_tensor.mutable_data<float>() + 1, 234);
      ASSERT_EQ(at_tensor[0][1].item().to<float>(), 234);

      // resize PT tensor in larger direction - storage is preserved
      at_tensor.resize_({6, 6});
      cuda_set<float>(c2_tensor.mutable_data<float>() + 2, 345);
      ASSERT_EQ(at_tensor[0][2].item().to<float>(), 345);
      ASSERT_EQ(c2_tensor.sizes()[0], 6);
      ASSERT_EQ(c2_tensor.sizes()[1], 6);

      // resize Caffe2 tensor - semantics are to NOT preserve the data, but the
      // TensorImpl is still shared
      c2_tensor.Resize(7, 7);
      cuda_set<float>(c2_tensor.mutable_data<float>() + 3, 456);
      ASSERT_EQ(at_tensor[0][3].item().to<float>(), 456);
      ASSERT_EQ(at_tensor.sizes()[0], 7);
      ASSERT_EQ(at_tensor.sizes()[1], 7);

    */
}
