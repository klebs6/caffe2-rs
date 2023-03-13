crate::ix!();

#[test] fn reshape_op_example() {

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Reshape",
        ["data"],
        ["reshaped", "old_shape"],
        shape=(3,2)
    )

    workspace.FeedBlob("data", (np.random.randint(100, size=(6))))
    print("data:", workspace.FetchBlob("data"))
    workspace.RunOperatorOnce(op)
    print("reshaped:", workspace.FetchBlob("reshaped"))
    print("old_shape:", workspace.FetchBlob("old_shape"))

    data: [86 60 85 96  7 37]
    reshaped: [[86 60]
              [85 96]
              [ 7 37]]
    old_shape: [6]
    */
}

declare_string!{caffe_test_root}

#[inline] pub fn add_const_input(
    shape: &Vec<i64>,
    value: f32,
    name:  &String,
    ws:    *mut Workspace)  {
    
    todo!();
    /*
        DeviceOption option;
      option.set_device_type(PROTO_CUDA);
      CUDAContext context(option);
      Blob* blob = ws->CreateBlob(name);
      auto* tensor = BlobGetMutableTensor(blob, CUDA);
      tensor->Resize(shape);
      math::Set<float, CUDAContext>(
          tensor->numel(), value, tensor->template mutable_data<float>(), &context);
      return;
    */
}

#[test] fn reshape_op_gpu_test_reshape_with_scalar() {
    todo!();
    /*
      if (!HasCudaGPU())
        return;
      Workspace ws;
      OperatorDef def;
      def.set_name("test_reshape");
      def.set_type("Reshape");
      def.add_input("X");
      def.add_output("XNew");
      def.add_output("OldShape");
      def.add_arg()->CopyFrom(MakeArgument("shape", vector<int64_t>{1}));
      def.mutable_device_option()->set_device_type(PROTO_CUDA);
      AddConstInput(vector<int64_t>(), 3.14, "X", &ws);
      // execute the op
      unique_ptr<OperatorStorage> op(CreateOperator(def, &ws));
      EXPECT_TRUE(op->Run());
      Blob* XNew = ws.GetBlob("XNew");
      const Tensor& XNewTensor = XNew->Get<Tensor>();
      EXPECT_EQ(1, XNewTensor.dim());
      EXPECT_EQ(1, XNewTensor.numel());
  */
}
