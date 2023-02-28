crate::ix!();

use crate::{
    OperatorStorage,
    CPUContext,
};


pub struct IncrementByOneOp {
    storage: OperatorStorage,
    context: CPUContext,
}

impl IncrementByOneOp {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<CPUContext>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& in = Input(0);

        auto* out = Output(0, in.sizes(), at::dtype<float>());
        const float* in_data = in.template data<float>();
        float* out_data = out->template mutable_data<float>();
        for (int i = 0; i < in.numel(); ++i) {
          out_data[i] = in_data[i] + 1.f;
        }
        return true;
        */
    }
}

num_inputs!{IncrementByOne, 1}

num_outputs!{IncrementByOne, 1}

allow_inplace!{IncrementByOne, vec![(0, 0)]}

register_cpu_operator!{IncrementByOne, IncrementByOneOp}

register_cuda_operator!{IncrementByOne, GPUFallbackOp}

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
