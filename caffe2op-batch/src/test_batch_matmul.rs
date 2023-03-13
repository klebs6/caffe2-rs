crate::ix!();

pub struct BatchMatMulOpTest {
    option:      DeviceOption,
    cpu_context: Box<CPUContext>,
    ws:          Workspace,
    def:         OperatorDef,
}

impl BatchMatMulOpTest {

    #[inline] pub fn set_up(&mut self)  {
        
        todo!();
        /*
            cpu_context_ = make_unique<CPUContext>(option_);
        def_.set_name("test");
        def_.set_type("BatchMatMul");
        def_.add_input("A");
        def_.add_input("B");
        def_.add_output("Y");
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
        auto* tensor = BlobGetMutableTensor(blob, CPU);
        tensor->Resize(dims);
        math::Set<float, CPUContext>(
            tensor->numel(),
            value,
            tensor->template mutable_data<float>(),
            cpu_context_.get());
        */
    }
    
    #[inline] pub fn verify_output(
        &self, 
        dims: &Vec<i64>,
        value: f32)  
    {
        todo!();
        /*
            const Blob* Y_blob = ws_.GetBlob("Y");
        ASSERT_NE(nullptr, Y_blob);
        const auto& Y = Y_blob->Get<TensorCPU>();
        const auto Y_dims = Y.sizes();
        ASSERT_EQ(dims.size(), Y_dims.size());
        for (std::size_t i = 0; i < dims.size(); ++i) {
          ASSERT_EQ(dims[i], Y_dims[i]);
        }
        for (int i = 0; i < Y.numel(); ++i) {
          EXPECT_FLOAT_EQ(value, Y.data<float>()[i]);
        }
        */
    }
}

#[test] fn batch_mat_mul_op_test_batch_mat_mul_op_normal_test() {
    todo!();
    /*
  AddConstInput(std::vector<int64_t>{3, 5, 10}, 1.0f, "A");
  AddConstInput(std::vector<int64_t>{3, 10, 6}, 1.0f, "B");
  std::unique_ptr<OperatorStorage> op(CreateOperator(def_, &ws_));
  ASSERT_NE(nullptr, op);
  ASSERT_TRUE(op->Run());
  VerifyOutput(std::vector<int64_t>{3, 5, 6}, 10.0f);
  */
}

#[test] fn batch_mat_mul_op_test_batch_mat_mul_op_broadcast_test() {
    todo!();
    /*
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
