crate::ix!();

#[test] fn OperatorFallbackTest_IncrementByOneOp() {
    todo!();
    /*
      OperatorDef op_def = CreateOperatorDef(
          "IncrementByOne", "", vector<string>{"X"},
          vector<string>{"X"});
      Workspace ws;
      Tensor source_tensor(vector<int64_t>{2, 3}, CPU);
      for (int i = 0; i < 6; ++i) {
        source_tensor.mutable_data<float>()[i] = i;
      }
      BlobGetMutableTensor(ws.CreateBlob("X"), CPU)->CopyFrom(source_tensor);
      unique_ptr<OperatorStorage> op(CreateOperator(op_def, &ws));
      EXPECT_TRUE(op.get() != nullptr);
      EXPECT_TRUE(op->Run());
      const TensorCPU& output = ws.GetBlob("X")->Get<TensorCPU>();
      EXPECT_EQ(output.dim(), 2);
      EXPECT_EQ(output.size(0), 2);
      EXPECT_EQ(output.size(1), 3);
      for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(output.data<float>()[i], i + 1);
      }
  */
}

#[test] fn OperatorFallbackTest_GPUIncrementByOneOp() {
    todo!();
    /*
      if (!HasCudaGPU()) return;
      OperatorDef op_def = CreateOperatorDef(
          "IncrementByOne", "", vector<string>{"X"},
          vector<string>{"X"});
      op_def.mutable_device_option()->set_device_type(PROTO_CUDA);
      Workspace ws;
      Tensor source_tensor(vector<int64_t>{2, 3}, CPU);
      for (int i = 0; i < 6; ++i) {
        source_tensor.mutable_data<float>()[i] = i;
      }
      BlobGetMutableTensor(ws.CreateBlob("X"), CUDA)->CopyFrom(source_tensor);
      unique_ptr<OperatorStorage> op(CreateOperator(op_def, &ws));
      EXPECT_TRUE(op.get() != nullptr);
      EXPECT_TRUE(op->Run());
      const TensorCUDA& output = ws.GetBlob("X")->Get<TensorCUDA>();
      Tensor output_cpu(output, CPU);
      EXPECT_EQ(output.dim(), 2);
      EXPECT_EQ(output.size(0), 2);
      EXPECT_EQ(output.size(1), 3);
      for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(output_cpu.data<float>()[i], i + 1);
      }
  */
}
