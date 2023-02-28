crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/test/tensor_interop_test.cpp]

#[test] fn caffe_2to_pytorch_simple_legacy() {
    todo!();
    /*
    
      Tensor c2_tensor(CPU);
      c2_tensor.Resize(4, 4);
      auto data = c2_tensor.mutable_data<i64>();
      for (i64 i = 0; i < 16; i++) {
        data[i] = i;
      }
      Tensor at_tensor(c2_tensor);

      auto it = at_tensor.data_ptr<i64>();
      for (i64 i = 0; i < 16; i++) {
        ASSERT_EQ(it[i], i);
      }

    */
}

#[test] fn caffe_2to_pytorch_simple() {
    todo!();
    /*
    
      Tensor c2_tensor = empty({4, 4}, kLong);
      auto data = c2_tensor.mutable_data<i64>();
      for (i64 i = 0; i < 16; i++) {
        data[i] = i;
      }
      Tensor at_tensor(c2_tensor);

      auto it = at_tensor.data_ptr<i64>();
      for (i64 i = 0; i < 16; i++) {
        ASSERT_EQ(it[i], i);
      }

    */
}

#[test] fn caffe_2to_pytorch_external_data() {
    todo!();
    /*
    
      Tensor c2_tensor = empty({4, 4}, kLong);
      i64 buf[16];
      for (i64 i = 0; i < 16; i++) {
        buf[i] = i;
      }
      c2_tensor.ShareExternalPointer(buf, 16 * sizeof(i64));

      // If the buffer is allocated externally, we can still pass tensor around,
      // but we can't resize its storage using PT APIs
      Tensor at_tensor(c2_tensor);
      at_tensor.permute({1, 0});
      at_tensor.permute({1, 0});
      auto it = at_tensor.data_ptr<i64>();
      for (i64 i = 0; i < 16; i++) {
        ASSERT_EQ(it[i], i);
      }
      ASSERT_FALSE(at_tensor.storage().resizable());
      ASSERT_ANY_THROW(at_tensor.resize_({7,7}));

    */
}

#[test] fn caffe_2to_pytorch_op() {
    todo!();
    /*
    
      Tensor c2_tensor(CPU);
      c2_tensor.Resize(3, 3);
      auto data = c2_tensor.mutable_data<i64>();
      for (i64 i = 0; i < 9; i++) {
        data[i] = i;
      }
      Tensor at_tensor(c2_tensor);

      ASSERT_EQ(sum(at_tensor).item<i64>(), 36);

    */
}

/**
  | Caffe2 doesn't actually have another always-on
  | backend that is not CPU or GPU
  |
  | TEST(Caffe2ToPytorch, UnsupportedDevice) {
  |   Tensor c2_tensor(OPENGL);
  |   c2_tensor.Resize(4, 4);
  |   c2_tensor.mutable_data<float>();
  |   Tensor at_tensor(c2_tensor);
  |   ASSERT_ANY_THROW(sum(at_tensor));
  | }
  */
#[test] fn caffe_2to_pytorch_partially_initialized() {
    todo!();
    /*
    
      // These APIs for partially initialized tensors should go away soon, in the
      // meantime ensure they are caught
      {
        // no dtype, no storage
        Tensor c2_tensor(CPU);
        ASSERT_ANY_THROW(Tensor at_tensor(c2_tensor));
      }
      {
        // storage, no dtype
        Tensor c2_tensor(CPU);
        c2_tensor.Resize(4,4);
        ASSERT_ANY_THROW(Tensor at_tensor(c2_tensor));
      }
      {
        // dtype, no storage
        Tensor c2_tensor(CPU);
        c2_tensor.Resize(4,4);
        c2_tensor.mutable_data<float>();
        c2_tensor.FreeMemory();
        ASSERT_ANY_THROW(Tensor at_tensor(c2_tensor));
      }

    */
}

#[test] fn caffe_2to_pytorch_mutual_resizes() {
    todo!();
    /*
    
      Tensor c2_tensor = empty({5, 5}, kFloat);
      auto data = c2_tensor.mutable_data<float>();
      for (i64 i = 0; i < 25; i++) {
        data[i] = 0;
      }

      Tensor at_tensor(c2_tensor);

      // change is visible
      at_tensor[0][0] = 123;
      ASSERT_EQ(c2_tensor.mutable_data<float>()[0], 123);

      // resize PT tensor in smaller direction - storage is preserved
      at_tensor.resize_({4, 4});
      c2_tensor.mutable_data<float>()[1] = 234;
      ASSERT_EQ(at_tensor[0][1].item().to<float>(), 234);

      // resize PT tensor in larger direction - storage is preserved
      at_tensor.resize_({6, 6});
      c2_tensor.mutable_data<float>()[2] = 345;
      ASSERT_EQ(at_tensor[0][2].item().to<float>(), 345);
      ASSERT_EQ(c2_tensor.sizes()[0], 6);
      ASSERT_EQ(c2_tensor.sizes()[1], 6);

      // resize Caffe2 tensor - semantics are to NOT preserve the data, but the
      // TensorImpl is still shared
      c2_tensor.Resize(7, 7);
      c2_tensor.mutable_data<float>()[3] = 456;
      ASSERT_EQ(at_tensor[0][3].item().to<float>(), 456);
      ASSERT_EQ(at_tensor.sizes()[0], 7);
      ASSERT_EQ(at_tensor.sizes()[1], 7);

    */
}

#[test] fn pytorch_to_caffe2_op() {
    todo!();
    /*
    
      Workspace workspace;
      NetDef net;

      auto at_tensor_a = ones({5, 5}, dtype(kFloat));
      auto at_tensor_b = ones({5, 5}, dtype(kFloat));
      auto at_tensor_c = ones({5, 5}, dtype(kFloat));

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
      }

      workspace.RunNetOnce(net);

      auto result = XBlobGetMutableTensor(workspace.CreateBlob("d"), {5, 5}, kCPU);

      auto it = result.data<float>();
      for (i64 i = 0; i < 25; i++) {
        ASSERT_EQ(it[i], 3.0);
      }
      Tensor at_result(result);
      ASSERT_EQ(sum(at_result).item<float>(), 75);

    */
}

#[test] fn pytorch_to_caffe2_shared_storage_read() {
    todo!();
    /*
    
      Workspace workspace;
      NetDef net;

      auto at_tensor_a = ones({5, 5}, dtype(kFloat));
      auto at_tensor_b = at_tensor_a.view({5, 5});

      auto* c2_tensor_a = BlobSetTensor(workspace.CreateBlob("a"), Tensor(at_tensor_a));
      auto* c2_tensor_b = BlobSetTensor(workspace.CreateBlob("b"), Tensor(at_tensor_b));

      {
        auto op = net.add_op();
        op->set_type("Add");
        op->add_input("a");
        op->add_input("b");
        op->add_output("c");
      }

      workspace.RunNetOnce(net);

      auto result = XBlobGetMutableTensor(workspace.CreateBlob("c"), {5, 5}, kCPU);
      auto it = result.data<float>();
      for (i64 i = 0; i < 25; i++) {
        ASSERT_EQ(it[i], 2.0);
      }
      Tensor at_result(result);
      ASSERT_EQ(sum(at_result).item<float>(), 50);

    */
}

#[test] fn pytorch_to_caffe2_shared_storage_write() {
    todo!();
    /*
    
      auto at_tensor_a = ones({5, 5}, dtype(kFloat));
      auto at_tensor_b = at_tensor_a.view({25});

      Tensor c2_tensor_a(at_tensor_a);
      Tensor c2_tensor_b(at_tensor_b);

      // change is visible everywhere
      c2_tensor_a.mutable_data<float>()[1] = 123;
      ASSERT_EQ(c2_tensor_b.mutable_data<float>()[1], 123);
      ASSERT_EQ(at_tensor_a[0][1].item().to<float>(), 123);
      ASSERT_EQ(at_tensor_b[1].item().to<float>(), 123);

    */
}

#[test] fn pytorch_to_caffe2_mutual_resizes() {
    todo!();
    /*
    
      auto at_tensor = ones({5, 5}, dtype(kFloat));

      Tensor c2_tensor(at_tensor);

      // change is visible
      c2_tensor.mutable_data<float>()[0] = 123;
      ASSERT_EQ(at_tensor[0][0].item().to<float>(), 123);

      // resize PT tensor in smaller direction - storage is preserved
      at_tensor.resize_({4, 4});
      c2_tensor.mutable_data<float>()[1] = 234;
      ASSERT_EQ(at_tensor[0][1].item().to<float>(), 234);

      // resize PT tensor in larger direction - storage is preserved
      at_tensor.resize_({6, 6});
      c2_tensor.mutable_data<float>()[2] = 345;
      ASSERT_EQ(at_tensor[0][2].item().to<float>(), 345);
      ASSERT_EQ(c2_tensor.sizes()[0], 6);
      ASSERT_EQ(c2_tensor.sizes()[1], 6);

      // resize Caffe2 tensor - semantics are to NOT preserve the data, but the
      // TensorImpl is still shared
      c2_tensor.Resize(7, 7);
      c2_tensor.mutable_data<float>()[3] = 456;
      ASSERT_EQ(at_tensor[0][3].item().to<float>(), 456);
      ASSERT_EQ(at_tensor.sizes()[0], 7);
      ASSERT_EQ(at_tensor.sizes()[1], 7);

    */
}

#[test] fn pytorch_to_caffe2_strided() {
    todo!();
    /*
    
      auto at_tensor = ones({5, 5}, dtype(kFloat)).t();
      ASSERT_ANY_THROW(Tensor c2_tensor(at_tensor));
      // but calling contiguous is fine
      Tensor c2_tensor(at_tensor.contiguous());
      for (i64 i = 0; i < 25; i++) {
        ASSERT_EQ(c2_tensor.data<float>()[i], 1.0);
      }

    */
}

#[test] fn pytorch_to_caffe2_inplace_strided() {
    todo!();
    /*
    
      auto at_tensor = zeros({2, 5}, dtype(kFloat));
      Tensor c2_tensor(at_tensor);
      ASSERT_EQ(c2_tensor.sizes()[0], 2);
      ASSERT_EQ(c2_tensor.sizes()[1], 5);

      c2_tensor.mutable_data<float>()[1] = 234;
      ASSERT_EQ(at_tensor[0][1].item().to<float>(), 234);

      at_tensor.t_();
      ASSERT_EQ(c2_tensor.sizes()[0], 5);
      ASSERT_EQ(c2_tensor.sizes()[1], 2);
      // This is BROKEN situation, however checking is_contiguous on every data
      // access is expensive. We rely on user to not do crazy stuff.
      ASSERT_EQ(at_tensor[1][0].item().to<float>(), 234);
      ASSERT_EQ(c2_tensor.data<float>()[1], 234);

    */
}

#[test] fn pytorch_to_caffe2_non_regular_tensor() {
    todo!();
    /*
    
      Tensor at_tensor =
          empty({2, 3}, dtype<float>().layout(kSparse));
      ASSERT_TRUE(at_tensor.is_sparse());
      ASSERT_ANY_THROW(Tensor c2_tensor(at_tensor));

    */
}

#[test] fn caffe_2to_pytorch_non_pod() {
    todo!();
    /*
    
      Tensor c2_tensor = empty({1}, dtype<string>());
      auto data = c2_tensor.mutable_data<string>();
      *data = "test";
      ASSERT_ANY_THROW(Tensor at_tensor(c2_tensor));

    */
}

#[test] fn caffe_2to_pytorch_nullptr() {
    todo!();
    /*
    
      Tensor c2_tensor;
      ASSERT_FALSE(c2_tensor.defined());
      Tensor at_tensor(c2_tensor);
      ASSERT_FALSE(at_tensor.defined());

    */
}

#[test] fn pytorch_to_caffe2_nullptr() {
    todo!();
    /*
    
      Tensor at_tensor;
      ASSERT_FALSE(at_tensor.defined());
      Tensor c2_tensor(at_tensor);
      ASSERT_FALSE(c2_tensor.defined());

    */
}
