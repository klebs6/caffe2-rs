crate::ix!();

pub struct BatchMatMulOpGPUTest {
    option:       DeviceOption,
    cuda_context: Box<CUDAContext>,
    ws:           Workspace,
    def:          OperatorDef,
}

impl BatchMatMulOpGPUTest {

    #[inline] pub fn set_up(&mut self)  {
        
        todo!();
        /*
            if (!HasCudaGPU()) {
          return;
        }
        option_.set_device_type(PROTO_CUDA);
        cuda_context_ = make_unique<CUDAContext>(option_);
        def_.set_name("test");
        def_.set_type("BatchMatMul");
        def_.add_input("A");
        def_.add_input("B");
        def_.add_output("Y");
        def_.mutable_device_option()->set_device_type(PROTO_CUDA);
        */
    }
    
    #[inline] pub fn add_const_input(
        &mut self, 
        dims: &Vec<i64>,
        value: f32,
        name: &String)  
    {
        todo!();
        /*
            Blob* blob = ws_.CreateBlob(name);
        auto* tensor = BlobGetMutableTensor(blob, CUDA);
        tensor->Resize(dims);
        math::Set<float, CUDAContext>(
            tensor->numel(),
            value,
            tensor->template mutable_data<float>(),
            cuda_context_.get());
        */
    }
    
    #[inline] pub fn verify_output(&self, 
        dims: &Vec<i64>,
        value: f32)  
    {
        todo!();
        /*
            const Blob* Y_blob = ws_.GetBlob("Y");
        ASSERT_NE(nullptr, Y_blob);
        const auto& Y = Y_blob->Get<Tensor>();
        Tensor Y_cpu(Y, CPU);
        const auto Y_dims = Y_cpu.sizes();
        ASSERT_EQ(dims.size(), Y_dims.size());
        for (std::size_t i = 0; i < dims.size(); ++i) {
          ASSERT_EQ(dims[i], Y_dims[i]);
        }
        for (int i = 0; i < Y_cpu.numel(); ++i) {
          EXPECT_FLOAT_EQ(value, Y_cpu.data<float>()[i]);
        }
        */
    }
}

#[test] fn BatchMatMulOpGPUTest_BatchMatMulOpGPUNormalTest() {
    todo!();
    /*
  if (!HasCudaGPU()) {
    return;
  }
  AddConstInput(std::vector<int64_t>{3, 5, 10}, 1.0f, "A");
  AddConstInput(std::vector<int64_t>{3, 10, 6}, 1.0f, "B");
  std::unique_ptr<OperatorStorage> op(CreateOperator(def_, &ws_));
  ASSERT_NE(nullptr, op);
  ASSERT_TRUE(op->Run());
  VerifyOutput(std::vector<int64_t>{3, 5, 6}, 10.0f);
  */
}

#[test] fn BatchMatMulOpGPUTest_BatchMatMulOpGPUBroadcastTest() {
    todo!();
    /*
  if (!HasCudaGPU()) {
    return;
  }
  auto* arg = def_.add_arg();
  arg->set_name("broadcast");
  arg->set_i(1);
  AddConstInput(std::vector<int64_t>{3, 5, 10}, 1.0f, "A");
  AddConstInput(std::vector<int64_t>{2, 3, 10, 6}, 1.0f, "B");
  std::unique_ptr<OperatorStorage> op(CreateOperator(def_, &ws_));
  ASSERT_NE(nullptr, op);
  ASSERT_TRUE(op->Run());
  VerifyOutput(std::vector<int64_t>{2, 3, 5, 6}, 10.0f);
  */
}
