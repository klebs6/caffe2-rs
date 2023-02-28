crate::ix!();

use crate::{
    Workspace,
};

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

#[test] fn ReshapeOpGPUTest_testReshapeWithScalar() {
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
