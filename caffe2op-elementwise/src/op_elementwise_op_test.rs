crate::ix!();

pub fn copy_vector_cpu_context_bool(N: i32, x: *const bool, y: *mut bool) {
    todo!();
    /*
       memcpy(y, x, N * sizeof(bool));
       */
}

pub fn copy_vector_cpu_context_i32(N: i32, x: *const i32, y: *mut i32) {
    todo!();
    /*
    memcpy(y, x, N * sizeof(int32_t));
    */
}

#[inline] pub fn fill_tensor<Context, I_Type, O_Type>(
    ws:     *mut Workspace,
    name:   &String,
    shape:  &Vec<i64>,
    values: &Vec<I_Type>) 
{
    todo!();
    /*
        auto* blob = ws->CreateBlob(name);
      auto* tensor = BlobGetMutableTensor(blob, Context::GetDeviceType());
      tensor->Resize(shape);
      auto* mutable_data = tensor->template mutable_data<O_Type>();
      const O_Type* data = reinterpret_cast<const O_Type*>(values.data());
      CopyVector<Context, O_Type>(values.size(), data, mutable_data);
    */
}

#[inline] pub fn create_operator_def<Context>() -> OperatorDef {
    todo!();
    /*
        caffe2::OperatorDef def;
      return def;
    */
}

#[inline] pub fn define_operator<Context>(op_type: &String) -> OperatorDef {
    todo!();
    /*
        caffe2::OperatorDef def = CreateOperatorDef<Context>();
      def.set_name("test");
      def.set_type(op_type);
      def.add_input("X");
      def.add_input("Y");
      def.add_output("Z");
      return def;
    */
}

#[inline] pub fn elementwise_and<Context>() {
    todo!();
    /*
        const int N = 4;
      const int M = 2;
      caffe2::Workspace ws;
      auto def = DefineOperator<Context>("And");
      { // equal size
        FillTensor<Context, uint8_t, bool>(
            &ws, "X", {N}, {true, false, true, false});
        FillTensor<Context, uint8_t, bool>(
            &ws, "Y", {N}, {true, true, false, false});
        std::unique_ptr<caffe2::OperatorStorage> op(caffe2::CreateOperator(def, &ws));
        EXPECT_NE(nullptr, op.get());
        EXPECT_TRUE(op->Run());
        auto* blob = ws.GetBlob("Z");
        EXPECT_NE(nullptr, blob);
        caffe2::Tensor Z(blob->Get<caffe2::Tensor>(), caffe2::CPU);
        EXPECT_EQ(Z.numel(), N);
        std::vector<bool> result{true, false, false, false};
        for (size_t i = 0; i < Z.numel(); ++i) {
          EXPECT_EQ(Z.template data<bool>()[i], result[i]);
        }
      }
      { // broadcast
        auto* arg = def.add_arg();
        arg->set_name("broadcast");
        arg->set_i(1);
        FillTensor<Context, uint8_t, bool>(
            &ws, "X", {M, N}, {true, false, true, false, true, false, true, false});
        FillTensor<Context, uint8_t, bool>(
            &ws, "Y", {N}, {true, true, false, false});
        std::unique_ptr<caffe2::OperatorStorage> op(caffe2::CreateOperator(def, &ws));
        EXPECT_NE(nullptr, op.get());
        EXPECT_TRUE(op->Run());
        auto* blob = ws.GetBlob("Z");
        EXPECT_NE(nullptr, blob);
        caffe2::Tensor Z(blob->Get<caffe2::Tensor>(), caffe2::CPU);
        EXPECT_EQ(Z.numel(), M * N);
        std::vector<bool> result{
            true, false, false, false, true, false, false, false};
        for (size_t i = 0; i < Z.numel(); ++i) {
          EXPECT_EQ(Z.template data<bool>()[i], result[i]);
        }
      }
    */
}

#[inline] pub fn elementwise_or<Context>() {
    todo!();
    /*
        const int N = 4;
      const int M = 2;
      caffe2::Workspace ws;
      auto def = DefineOperator<Context>("Or");
      { // equal size
        FillTensor<Context, uint8_t, bool>(
            &ws, "X", {N}, {true, false, true, false});
        FillTensor<Context, uint8_t, bool>(
            &ws, "Y", {N}, {true, true, false, false});
        std::unique_ptr<caffe2::OperatorStorage> op(caffe2::CreateOperator(def, &ws));
        EXPECT_NE(nullptr, op.get());
        EXPECT_TRUE(op->Run());
        auto* blob = ws.GetBlob("Z");
        EXPECT_NE(nullptr, blob);
        caffe2::Tensor Z(blob->Get<caffe2::Tensor>(), caffe2::CPU);
        EXPECT_EQ(Z.numel(), N);
        std::vector<bool> result{true, true, true, false};
        for (size_t i = 0; i < Z.numel(); ++i) {
          EXPECT_EQ(Z.template data<bool>()[i], result[i]);
        }
      }
      { // broadcast
        auto* arg = def.add_arg();
        arg->set_name("broadcast");
        arg->set_i(1);
        FillTensor<Context, uint8_t, bool>(
            &ws, "X", {M, N}, {true, false, true, false, true, false, true, false});
        FillTensor<Context, uint8_t, bool>(
            &ws, "Y", {N}, {true, true, false, false});
        std::unique_ptr<caffe2::OperatorStorage> op(caffe2::CreateOperator(def, &ws));
        EXPECT_NE(nullptr, op.get());
        EXPECT_TRUE(op->Run());
        auto* blob = ws.GetBlob("Z");
        EXPECT_NE(nullptr, blob);
        caffe2::Tensor Z(blob->Get<caffe2::Tensor>(), caffe2::CPU);
        EXPECT_EQ(Z.numel(), M * N);
        std::vector<bool> result{true, true, true, false, true, true, true, false};
        for (size_t i = 0; i < Z.numel(); ++i) {
          EXPECT_EQ(Z.template data<bool>()[i], result[i]);
        }
      }
    */
}

#[inline] pub fn elementwise_xor<Context>() {
    todo!();
    /*
        const int N = 4;
      const int M = 2;
      caffe2::Workspace ws;
      auto def = DefineOperator<Context>("Xor");
      { // equal size
        FillTensor<Context, uint8_t, bool>(
            &ws, "X", {N}, {true, false, true, false});
        FillTensor<Context, uint8_t, bool>(
            &ws, "Y", {N}, {true, true, false, false});
        std::unique_ptr<caffe2::OperatorStorage> op(caffe2::CreateOperator(def, &ws));
        EXPECT_NE(nullptr, op.get());
        EXPECT_TRUE(op->Run());
        auto* blob = ws.GetBlob("Z");
        EXPECT_NE(nullptr, blob);
        caffe2::Tensor Z(blob->Get<caffe2::Tensor>(), caffe2::CPU);
        EXPECT_EQ(Z.numel(), N);
        std::vector<bool> result{false, true, true, false};
        for (size_t i = 0; i < Z.numel(); ++i) {
          EXPECT_EQ(Z.template data<bool>()[i], result[i]);
        }
      }
      { // broadcast
        auto* arg = def.add_arg();
        arg->set_name("broadcast");
        arg->set_i(1);
        FillTensor<Context, uint8_t, bool>(
            &ws, "X", {M, N}, {true, false, true, false, true, false, true, false});
        FillTensor<Context, uint8_t, bool>(
            &ws, "Y", {N}, {true, true, false, false});
        std::unique_ptr<caffe2::OperatorStorage> op(caffe2::CreateOperator(def, &ws));
        EXPECT_NE(nullptr, op.get());
        EXPECT_TRUE(op->Run());
        auto* blob = ws.GetBlob("Z");
        EXPECT_NE(nullptr, blob);
        caffe2::Tensor Z(blob->Get<caffe2::Tensor>(), caffe2::CPU);
        EXPECT_EQ(Z.numel(), M * N);
        std::vector<bool> result{
            false, true, true, false, false, true, true, false};
        for (size_t i = 0; i < Z.numel(); ++i) {
          EXPECT_EQ(Z.template data<bool>()[i], result[i]);
        }
      }
    */
}

#[inline] pub fn elementwise_not<Context>() {
    todo!();
    /*
        const int N = 2;
      caffe2::Workspace ws;
      caffe2::OperatorDef def = CreateOperatorDef<Context>();
      def.set_name("test");
      def.set_type("Not");
      def.add_input("X");
      def.add_output("Y");
      FillTensor<Context, uint8_t, bool>(&ws, "X", {N}, {true, false});
      std::unique_ptr<caffe2::OperatorStorage> op(caffe2::CreateOperator(def, &ws));
      EXPECT_NE(nullptr, op.get());
      EXPECT_TRUE(op->Run());
      auto* blob = ws.GetBlob("Y");
      EXPECT_NE(nullptr, blob);
      caffe2::Tensor Y(blob->Get<caffe2::Tensor>(), caffe2::CPU);
      EXPECT_EQ(Y.numel(), N);
      std::vector<bool> result{false, true};
      for (size_t i = 0; i < Y.numel(); ++i) {
        EXPECT_EQ(Y.template data<bool>()[i], result[i]);
      }
    */
}

#[inline] pub fn elementwiseEQ<Context>() {
    todo!();
    /*
        const int N = 4;
      const int M = 2;
      caffe2::Workspace ws;
      auto def = DefineOperator<Context>("EQ");
      { // equal size
        FillTensor<Context, int32_t, int32_t>(&ws, "X", {N}, {1, 100, 5, -10});
        FillTensor<Context, int32_t, int32_t>(&ws, "Y", {N}, {0, 100, 4, -10});
        std::unique_ptr<caffe2::OperatorStorage> op(caffe2::CreateOperator(def, &ws));
        EXPECT_NE(nullptr, op.get());
        EXPECT_TRUE(op->Run());
        auto* blob = ws.GetBlob("Z");
        EXPECT_NE(nullptr, blob);
        caffe2::Tensor Z(blob->Get<caffe2::Tensor>(), caffe2::CPU);
        EXPECT_EQ(Z.numel(), N);
        std::vector<bool> result{false, true, false, true};
        for (size_t i = 0; i < Z.numel(); ++i) {
          EXPECT_EQ(Z.template data<bool>()[i], result[i]);
        }
      }
      { // boolean
        FillTensor<Context, uint8_t, bool>(
            &ws, "X", {N}, {true, false, false, true});
        FillTensor<Context, uint8_t, bool>(
            &ws, "Y", {N}, {true, false, true, false});
        std::unique_ptr<caffe2::OperatorStorage> op(caffe2::CreateOperator(def, &ws));
        EXPECT_NE(nullptr, op.get());
        EXPECT_TRUE(op->Run());
        auto* blob = ws.GetBlob("Z");
        EXPECT_NE(nullptr, blob);
        caffe2::Tensor Z(blob->Get<caffe2::Tensor>(), caffe2::CPU);
        EXPECT_EQ(Z.numel(), N);
        std::vector<bool> result{true, true, false, false};
        for (size_t i = 0; i < Z.numel(); ++i) {
          EXPECT_EQ(Z.template data<bool>()[i], result[i]);
        }
      }
      { // broadcast
        auto* arg = def.add_arg();
        arg->set_name("broadcast");
        arg->set_i(1);
        FillTensor<Context, int32_t, int32_t>(
            &ws, "X", {M, N}, {1, 100, 5, -10, 3, 6, -1000, 33});
        FillTensor<Context, int32_t, int32_t>(&ws, "Y", {N}, {1, 6, -1000, -10});
        std::unique_ptr<caffe2::OperatorStorage> op(caffe2::CreateOperator(def, &ws));
        EXPECT_NE(nullptr, op.get());
        EXPECT_TRUE(op->Run());
        auto* blob = ws.GetBlob("Z");
        EXPECT_NE(nullptr, blob);
        caffe2::Tensor Z(blob->Get<caffe2::Tensor>(), caffe2::CPU);
        EXPECT_EQ(Z.numel(), M * N);
        std::vector<bool> result{
            true, false, false, true, false, true, true, false};
        for (size_t i = 0; i < Z.numel(); ++i) {
          EXPECT_EQ(Z.template data<bool>()[i], result[i]);
        }
      }
    */
}

#[test] fn ElementwiseCPUTest_And() {
    todo!();
    /*
      elementwiseAnd<caffe2::CPUContext>();
  */
}


#[test] fn ElementwiseTest_Or() {
    todo!();
    /*
      elementwiseOr<caffe2::CPUContext>();
  */
}


#[test] fn ElementwiseTest_Xor() {
    todo!();
    /*
      elementwiseXor<caffe2::CPUContext>();
  */
}


#[test] fn ElementwiseTest_Not() {
    todo!();
    /*
      elementwiseNot<caffe2::CPUContext>();
  */
}


#[test] fn ElementwiseTest_EQ() {
    todo!();
    /*
      elementwiseEQ<caffe2::CPUContext>();
  */
}

