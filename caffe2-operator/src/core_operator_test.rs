crate::ix!();

/**
  | Since we instantiate this on CPU and
  | GPU (but don't want a
  | 
  | CUDAContext dependency, we use OperatorStorage.
  | In general, you only want to inherit
  | from Operator<Context> in your code.
  |
  */
pub struct JustTest {
    base: OperatorStorage,
}

impl JustTest {

    #[inline] pub fn run(&mut self, stream_id: i32) -> bool {
        
        todo!();
        /*
            return true;
        */
    }
    
    #[inline] pub fn typename(&mut self) -> String {
        
        todo!();
        /*
            return "base";
        */
    }
}

///-----------------
pub struct JustTestAndNeverConstructs {
    base: JustTest,
}
impl JustTestAndNeverConstructs {

    pub fn new(def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : JustTest(def, ws) 

        throw UnsupportedOperatorFeature("I just don't construct.");
        */
    }
    
    #[inline] pub fn run(&mut self, stream_id: i32) -> bool {
        
        todo!();
        /*
            return true;
        */
    }
    
    #[inline] pub fn typename(&mut self) -> String {
        
        todo!();
        /*
            return "FOO";
        */
    }
}

///-----------------
pub struct JustTestAndDoesConstruct {
    base: JustTest,
}
impl JustTestAndDoesConstruct {

    #[inline] pub fn run(&mut self, stream_id: i32) -> bool {
        
        todo!();
        /*
            return true;
        */
    }
    
    #[inline] pub fn typename(&mut self) -> String {
        
        todo!();
        /*
            return "BAR";
        */
    }
}

///----------------
pub struct JustTestWithSomeOutput {
    base: JustTest,
}
impl JustTestWithSomeOutput {

    #[inline] pub fn run(&mut self, stream_id: i32) -> bool {
        
        todo!();
        /*
            *OperatorStorage::Output<int>(0) = 5;
        return true;
        */
    }
    
    #[inline] pub fn typename(&mut self) -> String {
        
        todo!();
        /*
            return "SETTING_SOME_OUTPUT";
        */
    }
}

num_inputs!{JustTest,  {0, 1}}
num_outputs!{JustTest, {0, 1}}

num_inputs!{JustTestCPUOnly,  {0, 1}}
num_outputs!{JustTestCPUOnly, {0, 1}}

register_cpu_operator!{JustTest, JustTest}
register_cpu_operator!{JustTestCPUOnly, JustTest}
register_cpu_operator_with_engine!{JustTest, FOO, JustTestAndNeverConstructs}
register_cpu_operator_with_engine!{JustTest, BAR, JustTestAndDoesConstruct}
register_cpu_operator_with_engine!{JustTest, BAZ, JustTestAndDoesConstruct}
register_cuda_operator!{JustTest, JustTest}
register_cpu_operator!{JustTestWithSomeOutput, JustTestWithSomeOutput}

#[test] fn OperatorTest_DeviceTypeRegistryWorks() {
    todo!();
    /*
      EXPECT_EQ(gDeviceTypeRegistry()->count(CPU), 1);
  */
}


#[test] fn OperatorTest_RegistryWorks() {
    todo!();
    /*
      OperatorDef op_def;
      Workspace ws;
      op_def.set_type("JustTest");
      unique_ptr<OperatorStorage> op = CreateOperator(op_def, &ws);
      EXPECT_NE(nullptr, op.get());
      // After introducing events, CUDA operator creation has to have CUDA compiled
      // as it needs to instantiate an Event object with CUDAContext. Thus we will
      // guard this test below.
      if (HasCudaRuntime()) {
        op_def.mutable_device_option()->set_device_type(PROTO_CUDA);
        op = CreateOperator(op_def, &ws);
        EXPECT_NE(nullptr, op.get());
      }
  */
}


#[test] fn OperatorTest_RegistryWrongDevice() {
    todo!();
    /*
      OperatorDef op_def;
      Workspace ws;
      op_def.set_type("JustTypeCPUOnly");
      op_def.mutable_device_option()->set_device_type(PROTO_CUDA);
      try {
        CreateOperator(op_def, &ws);
        LOG(FATAL) << "No exception was thrown";
      } catch (const std::exception& e) {
        LOG(INFO) << "Exception " << e.what();
      }
  */
}


#[test] fn OperatorTest_ExceptionWorks() {
    todo!();
    /*
      OperatorDef op_def;
      Workspace ws;
      op_def.set_type("ThrowException");
      unique_ptr<OperatorStorage> op = CreateOperator(op_def, &ws);
      // Note: we do not do ASSERT_THROW in order to print out
      // the error message for inspection.
      try {
        op->Run();
        // This should not happen - exception should throw above.
        LOG(FATAL) << "This should not happen.";
      } catch (const EnforceNotMet& err) {
        LOG(INFO) << err.what();
      }
      try {
        op->RunAsync();
        // This should not happen - exception should throw above.
        LOG(FATAL) << "This should not happen.";
      } catch (const EnforceNotMet& err) {
        LOG(INFO) << err.what();
      }
  */
}


#[test] fn OperatorTest_FallbackIfEngineDoesNotBuild() {
    todo!();
    /*
      OperatorDef op_def;
      Workspace ws;
      op_def.set_type("JustTest");
      op_def.set_engine("FOO");
      unique_ptr<OperatorStorage> op = CreateOperator(op_def, &ws);
      EXPECT_NE(nullptr, op.get());
      EXPECT_EQ(static_cast<JustTest*>(op.get())->type(), "base");
  */
}


#[test] fn OperatorTest_MultipleEngineChoices() {
    todo!();
    /*
      OperatorDef op_def;
      Workspace ws;
      op_def.set_type("JustTest");
      op_def.set_engine("FOO,BAR");
      unique_ptr<OperatorStorage> op = CreateOperator(op_def, &ws);
      EXPECT_NE(nullptr, op.get());
      EXPECT_EQ(static_cast<JustTest*>(op.get())->type(), "BAR");
  */
}


#[test] fn OperatorTest_CannotUseUninitializedBlob() {
    todo!();
    /*
      Workspace ws;
      OperatorDef op_def;
      op_def.set_name("JustTest0");
      op_def.set_type("JustTest");
      op_def.add_input("input");
      op_def.add_output("output");
      ASSERT_THROW(CreateOperator(op_def, &ws), EnforceNotMet);
  */
}


#[test] fn OperatorTest_TestParameterAccess() {
    todo!();
    /*
      OperatorDef op_def;
      Workspace ws;
      op_def.set_name("JustTest0");
      op_def.set_type("JustTest");
      op_def.add_input("input");
      op_def.add_output("output");
      AddArgument<float>("arg0", 0.1, &op_def);
      AddArgument<vector<int>>("arg1", vector<int>{1, 2}, &op_def);
      AddArgument<string>("arg2", "argstring", &op_def);
      EXPECT_NE(ws.CreateBlob("input"), nullptr);
      OperatorStorage op(op_def, &ws);
      EXPECT_FLOAT_EQ(op.GetSingleArgument<float>("arg0", 0.0), 0.1);
      vector<int> i = op.GetRepeatedArgument<int>("arg1");
      EXPECT_EQ(i.size(), 2);
      EXPECT_EQ(i[0], 1);
      EXPECT_EQ(i[1], 2);
      EXPECT_EQ(op.GetSingleArgument<string>("arg2", "default"), "argstring");
      auto default1 = op.GetRepeatedArgument<int>("arg3", {2, 3});
      EXPECT_EQ(default1.size(), 2);
      EXPECT_EQ(default1[0], 2);
      EXPECT_EQ(default1[1], 3);
      auto default2 = op.GetRepeatedArgument<int>("arg4");
      EXPECT_EQ(default2.size(), 0);
  */
}


#[test] fn OperatorTest_CannotAccessParameterWithWrongType() {
    todo!();
    /*
      OperatorDef op_def;
      Workspace ws;
      op_def.set_name("JustTest0");
      op_def.set_type("JustTest");
      op_def.add_input("input");
      op_def.add_output("output");
      AddArgument<float>("arg0", 0.1f, &op_def);
      EXPECT_NE(ws.CreateBlob("input"), nullptr);
      OperatorStorage op(op_def, &ws);
      EXPECT_FLOAT_EQ(op.GetSingleArgument<float>("arg0", 0.0), 0.1);
      ASSERT_THROW(op.GetSingleArgument<int>("arg0", 0), EnforceNotMet);
  */
}


#[cfg(gtest_has_death_test)]
#[test] fn OperatorDeathTest_DISABLED_CannotAccessRepeatedParameterWithWrongType() {
    todo!();
    /*
      OperatorDef op_def;
      Workspace ws;
      op_def.set_name("JustTest0");
      op_def.set_type("JustTest");
      op_def.add_input("input");
      op_def.add_output("output");
      AddArgument<vector<float>>("arg0", vector<float>{0.1f}, &op_def);
      EXPECT_NE(ws.CreateBlob("input"), nullptr);
      OperatorStorage op(op_def, &ws);
      auto args = op.GetRepeatedArgument<float>("arg0");
      EXPECT_EQ(args.size(), 1);
      EXPECT_FLOAT_EQ(args[0], 0.1f);
      EXPECT_DEATH(op.GetRepeatedArgument<int>("arg0"),
                   "Argument does not have the right field: expected ints");
  */
}

#[test] fn OperatorTest_TestDefaultValue() {
    todo!();
    /*
      OperatorDef op_def;
      Workspace ws;
      OperatorStorage op(op_def, &ws);
      EXPECT_FLOAT_EQ(op.GetSingleArgument<float>("arg-nonexisting", 0.5f), 0.5f);
  */
}


#[test] fn OperatorTest_TestSetUp() {
    todo!();
    /*
      Workspace ws;
      OperatorDef op_def;
      op_def.set_name("JustTest0");
      op_def.set_type("JustTest");
      op_def.add_input("input");
      op_def.add_output("output");
      EXPECT_NE(nullptr, ws.CreateBlob("input"));
      unique_ptr<OperatorStorage> op(CreateOperator(op_def, &ws));
      EXPECT_NE(nullptr, op.get());
      EXPECT_TRUE(ws.HasBlob("output"));
  */
}


#[test] fn OperatorTest_TestSetUpInputOutputCount() {
    todo!();
    /*
      Workspace ws;
      OperatorDef op_def;
      op_def.set_name("JustTest0");
      op_def.set_type("JustTest");
      op_def.add_input("input");
      op_def.add_input("input2");
      op_def.add_output("output");
      EXPECT_NE(nullptr, ws.CreateBlob("input"));
      EXPECT_NE(nullptr, ws.CreateBlob("input2"));
    #ifndef CAFFE2_NO_OPERATOR_SCHEMA
      // JustTest will only accept one single input.
      ASSERT_ANY_THROW(CreateOperator(op_def, &ws));
    #endif

      op_def.clear_input();
      op_def.add_input("input");
      op_def.add_output("output2");
    #ifndef CAFFE2_NO_OPERATOR_SCHEMA
      // JustTest will only produce one single output.
      ASSERT_ANY_THROW(CreateOperator(op_def, &ws));
    #endif
  */
}


#[test] fn OperatorTest_TestOutputValues() {
    todo!();
    /*
      NetDef net_def;
      net_def.set_name("NetForTest");
      OperatorDef op_def;
      Workspace ws;
      op_def.set_name("JustTest1");
      op_def.set_type("JustTestWithSomeOutput");
      op_def.add_output("output");
      // JustTest will only produce one single output.
      net_def.add_op()->CopyFrom(op_def);
      unique_ptr<NetBase> net(CreateNet(net_def, &ws));
      EXPECT_TRUE(net->Run());
      EXPECT_TRUE(ws.HasBlob("output"));
      EXPECT_EQ(ws.GetBlob("output")->Get<int>(), 5);
  */
}

#[inline] pub fn get_net_def_for_test() -> NetDef {
    
    todo!();
    /*
        NetDef net_def;
      OperatorDef op_def;
      net_def.set_name("NetForTest");
      op_def.set_name("JustTest0");
      op_def.set_type("JustTest");
      op_def.add_input("input");
      op_def.add_output("hidden");
      net_def.add_op()->CopyFrom(op_def);
      op_def.set_name("JustTest1");
      op_def.set_input(0, "hidden");
      op_def.set_output(0, "output");
      net_def.add_op()->CopyFrom(op_def);
      return net_def;
    */
}


#[test] fn NetTest_TestScaffoldingSimpleNet() {
    todo!();
    /*
      NetDef net_def = GetNetDefForTest();
      net_def.set_type("simple");
      Workspace ws;
      EXPECT_NE(nullptr, ws.CreateBlob("input"));
      unique_ptr<NetBase> net(CreateNet(net_def, &ws));
      EXPECT_NE(nullptr, net.get());
      EXPECT_TRUE(ws.HasBlob("input"));
      EXPECT_TRUE(ws.HasBlob("hidden"));
      EXPECT_TRUE(ws.HasBlob("output"));
      EXPECT_TRUE(net->Run());
  */
}

#[test] fn NetTest_TestScaffoldingDAGNet() {
    todo!();
    /*
      NetDef net_def = GetNetDefForTest();
      net_def.set_type("dag");
      net_def.set_num_workers(1);
      Workspace ws;
      EXPECT_NE(nullptr, ws.CreateBlob("input"));
      unique_ptr<NetBase> net(CreateNet(net_def, &ws));
      EXPECT_NE(nullptr, net.get());
      EXPECT_TRUE(ws.HasBlob("input"));
      EXPECT_TRUE(ws.HasBlob("hidden"));
      EXPECT_TRUE(ws.HasBlob("output"));
      EXPECT_TRUE(net->Run());
  */
}

///----------------
pub struct FooGradientOp {
    base: JustTest,
}

impl FooGradientOp
{
    
    #[inline] pub fn typename(&mut self) -> String {
        
        todo!();
        /*
            return "FooGradient";
        */
    }
}

///----------------
pub struct FooGradientDummyEngineOp {
    base: JustTest,
}
impl FooGradientDummyEngineOp {
    
    #[inline] pub fn typename(&mut self) -> String {
        
        todo!();
        /*
            return "FooGradientDummyEngine";
        */
    }
}

///-------------
pub struct GetFooGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetFooGradient<'a> {
    
    #[inline] pub fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return vector<OperatorDef>{
            CreateOperatorDef(
                "FooGradient", "",
                std::vector<string>{GO(0)},
                std::vector<string>{GI(0)})};
        */
    }
}

num_inputs!{FooGradient, 1}

num_outputs!{FooGradient, 1}

register_cpu_gradient_operator!{FooGradient, FooGradientOp}

register_cpu_gradient_operator_with_engine!{
    FooGradient,
    DUMMY_ENGINE,
    FooGradientDummyEngineOp
}

register_gradient!{Foo, GetFooGradient}

#[test] fn OperatorGradientRegistryTest_GradientSimple() {
    todo!();
    /*
      Argument arg = MakeArgument<int>("arg", 1);
      DeviceOption option;
      option.set_device_type(PROTO_CPU);
      OperatorDef def = CreateOperatorDef(
          "Foo", "", std::vector<string>{"in"}, std::vector<string>{"out"},
          std::vector<Argument>{arg}, option, "DUMMY_ENGINE");
      vector<GradientWrapper> g_output(1);
      g_output[0].dense_ = "out_grad";
      GradientOpsMeta meta = GetGradientForOp(def, g_output);
      // Check the names, input and output.
      EXPECT_EQ(meta.ops_.size(), 1);
      const OperatorDef& grad_op_def = meta.ops_[0];
      EXPECT_EQ(grad_op_def.type(), "FooGradient");
      EXPECT_EQ(grad_op_def.name(), "");
      EXPECT_EQ(grad_op_def.input_size(), 1);
      EXPECT_EQ(grad_op_def.output_size(), 1);
      EXPECT_EQ(grad_op_def.input(0), "out_grad");
      EXPECT_EQ(grad_op_def.output(0), "in_grad");
      // Checks the engine, device option and arguments.
      EXPECT_EQ(grad_op_def.engine(), "DUMMY_ENGINE");
      EXPECT_EQ(grad_op_def.device_option().device_type(), PROTO_CPU);
      EXPECT_EQ(grad_op_def.arg_size(), 1);
      EXPECT_EQ(
          grad_op_def.arg(0).SerializeAsString(),
          MakeArgument<int>("arg", 1).SerializeAsString());
      // Checks the gradient name for input.
      EXPECT_EQ(meta.g_input_.size(), 1);
      EXPECT_TRUE(meta.g_input_[0].IsDense());
      EXPECT_EQ(meta.g_input_[0].dense_, "in_grad");

      Workspace ws;
      EXPECT_NE(ws.CreateBlob("out_grad"), nullptr);
      unique_ptr<OperatorStorage> grad_op = CreateOperator(grad_op_def, &ws);
      EXPECT_NE(nullptr, grad_op.get());
      EXPECT_EQ(
          static_cast<JustTest*>(grad_op.get())->type(), "FooGradientDummyEngine");
  */
}


#[test] fn EnginePrefTest_PerOpEnginePref() {
    todo!();
    /*
      OperatorDef op_def;
      Workspace ws;
      op_def.set_type("JustTest");

      SetPerOpEnginePref({{CPU, {{"JustTest", {"BAR"}}}}});
      {
        const auto op = CreateOperator(op_def, &ws);
        EXPECT_NE(nullptr, op.get());
        EXPECT_EQ(static_cast<JustTest*>(op.get())->type(), "BAR");
      }
      // clear
      SetPerOpEnginePref({});

      // Invalid operator type
      ASSERT_THROW(
          SetPerOpEnginePref({{CPU, {{"NO_EXIST", {"BAR"}}}}}), EnforceNotMet);
  */
}


#[test] fn EnginePrefTest_GlobalEnginePref() {
    todo!();
    /*
      OperatorDef op_def;
      Workspace ws;
      op_def.set_type("JustTest");

      SetGlobalEnginePref({{CPU, {"FOO", "BAR"}}});
      {
        const auto op = CreateOperator(op_def, &ws);
        EXPECT_NE(nullptr, op.get());
        EXPECT_EQ(static_cast<JustTest*>(op.get())->type(), "BAR");
      }
      // clear
      SetGlobalEnginePref({});

      SetGlobalEnginePref({{CPU, {"FOO"}}});
      {
        const auto op = CreateOperator(op_def, &ws);
        EXPECT_NE(nullptr, op.get());
        EXPECT_EQ(static_cast<JustTest*>(op.get())->type(), "base");
      }
      // clear
      SetGlobalEnginePref({});

      // Invalid device type
      // This check is no longer necessary with the enum class
      // ASSERT_THROW(SetGlobalEnginePref({{8888, {"FOO"}}}), EnforceNotMet);
  */
}


#[test] fn EnginePrefTest_GlobalEnginePrefAndPerOpEnginePref() {
    todo!();
    /*
      OperatorDef op_def;
      Workspace ws;
      op_def.set_type("JustTest");

      SetPerOpEnginePref({{CPU, {{"JustTest", {"BAR"}}}}});
      SetGlobalEnginePref({{CPU, {"BAZ"}}});
      {
        const auto op = CreateOperator(op_def, &ws);
        EXPECT_NE(nullptr, op.get());
        // per op pref takes precedence
        EXPECT_EQ(static_cast<JustTest*>(op.get())->type(), "BAR");
      }
      // clear
      SetPerOpEnginePref({});
      SetGlobalEnginePref({});
  */
}


#[test] fn EnginePrefTest_GlobalEnginePrefAndPerOpEnginePrefAndOpDef() {
    todo!();
    /*
      OperatorDef op_def;
      Workspace ws;
      op_def.set_type("JustTest");
      op_def.set_engine("BAR");

      SetPerOpEnginePref({{CPU, {{"JustTest", {"BAZ"}}}}});
      SetGlobalEnginePref({{CPU, {"BAZ"}}});
      {
        const auto op = CreateOperator(op_def, &ws);
        EXPECT_NE(nullptr, op.get());
        // operator_def takes precedence
        EXPECT_EQ(static_cast<JustTest*>(op.get())->type(), "BAR");
      }
      // clear
      SetPerOpEnginePref({});
      SetGlobalEnginePref({});
  */
}


#[test] fn EnginePrefTest_SetOpEnginePref() {
    todo!();
    /*
      OperatorDef op_def;
      Workspace ws;
      op_def.set_type("JustTest");

      SetPerOpEnginePref({{CPU, {{"JustTest", {"BAZ"}}}}});
      SetOpEnginePref("JustTest", {{CPU, {"BAR"}}});
      {
        const auto op = CreateOperator(op_def, &ws);
        EXPECT_NE(nullptr, op.get());
        // operator_def takes precedence
        EXPECT_EQ(static_cast<JustTest*>(op.get())->type(), "BAR");
      }
      // clear
      SetPerOpEnginePref({});
      SetGlobalEnginePref({});
  */
}


#[test] fn EnginePrefTest_SetDefaultEngine() {
    todo!();
    /*
      OperatorDef op_def;
      Workspace ws;
      op_def.set_type("JustTest");

      SetPerOpEnginePref({{CPU, {{"JustTest", {"DEFAULT"}}}}});
      SetGlobalEnginePref({{CPU, {"BAR"}}});
      {
        const auto op = CreateOperator(op_def, &ws);
        EXPECT_NE(nullptr, op.get());
        // operator_def takes precedence
        EXPECT_EQ(static_cast<JustTest*>(op.get())->type(), "base");
      }
      // clear
      SetPerOpEnginePref({});
      SetGlobalEnginePref({});
  */
}


///----------------
pub struct JustTestWithRequiredArg {
    base: JustTest,
}
impl JustTestWithRequiredArg {

    #[inline] pub fn run(&mut self, stream_id: i32) -> bool {
        
        todo!();
        /*
            return true;
        */
    }
    
    #[inline] pub fn typename(&mut self) -> String {
        
        todo!();
        /*
            return "JustTestWithRequiredArg";
        */
    }
}

register_cpu_operator!{JustTestWithRequiredArg, JustTestWithRequiredArg}

num_inputs!{JustTestWithRequiredArg, (0,1)}

num_outputs!{JustTestWithRequiredArg, (0,1)}

args!{JustTestWithRequiredArg, 
    0 => ("test_arg", "this arg is required -- true")
}

#[test] fn RequiredArg_Basic() {
    todo!();
    /*
      OperatorDef op_def;
      Workspace ws;
      op_def.set_type("JustTestWithRequiredArg");

      {
        try {
          CreateOperator(op_def, &ws);
          LOG(FATAL) << "No exception was thrown";
        } catch (const std::exception& e) {
          LOG(INFO) << "Exception thrown (expected): " << e.what();
        }
      }

      {
        op_def.add_arg()->CopyFrom(MakeArgument("test_arg", 1));
        const auto op = CreateOperator(op_def, &ws);
        EXPECT_NE(nullptr, op.get());
        EXPECT_EQ(
            static_cast<JustTest*>(op.get())->type(), "JustTestWithRequiredArg");
      }
  */
}

///---------------
pub struct JustTestWithStandardIsTestArg {
    base: JustTest,
}

impl JustTestWithStandardIsTestArg {
    
    #[inline] pub fn run(&mut self, stream_id: i32) -> bool {
        
        todo!();
        /*
            return true;
        */
    }
    
    #[inline] pub fn typename(&mut self) -> String {
        
        todo!();
        /*
            return "JustTestWithStandardIsTestArg";
        */
    }
}

register_cpu_operator!{
    JustTestWithStandardIsTestArg,
    JustTestWithStandardIsTestArg
}

num_inputs!{JustTestWithStandardIsTestArg, (0,1)}

num_outputs!{JustTestWithStandardIsTestArg, (0,1)}

args_are_test!{JustTestWithStandardIsTestArg, 
    0 => ("this is_test arg is required")
}

#[test] fn IsTestArg_standard() {
    todo!();
    /*
      OperatorDef op_def;
      Workspace ws;
      op_def.set_type("JustTestWithStandardIsTestArg");

      {
        try {
          CreateOperator(op_def, &ws);
          LOG(FATAL) << "No exception was thrown";
        } catch (const std::exception& e) {
          LOG(INFO) << "Exception thrown (expected): " << e.what();
        }
      }

      {
        op_def.add_arg()->CopyFrom(MakeArgument(OpSchema::Arg_IsTest, 1));
        const auto op = CreateOperator(op_def, &ws);
        EXPECT_NE(nullptr, op.get());
        EXPECT_EQ(
            static_cast<JustTest*>(op.get())->type(),
            "JustTestWithStandardIsTestArg");
      }
  */
}

///--------------
pub struct JustTestWithNonStandardIsTestArg {
    base: JustTest,
}
impl JustTestWithNonStandardIsTestArg {

    #[inline] pub fn run(&mut self, stream_id: i32) -> bool {
        
        todo!();
        /*
            return true;
        */
    }
    
    #[inline] pub fn typename(&mut self) -> String {
        
        todo!();
        /*
            return "JustTestWithNonStandardIsTestArg";
        */
    }
}

register_cpu_operator!{JustTestWithNonStandardIsTestArg, JustTestWithNonStandardIsTestArg}

num_inputs!{JustTestWithNonStandardIsTestArg, (0,1)}

num_outputs!{JustTestWithNonStandardIsTestArg, (0,1)}

args!{JustTestWithNonStandardIsTestArg, 
    0 => ("OpSchema::Arg_IsTest", "this is_test arg is not required")
}

#[test] fn IsTestArg_non_standard() {
    todo!();
    /*
      OperatorDef op_def;
      Workspace ws;
      op_def.set_type("JustTestWithNonStandardIsTestArg");

      const auto op = CreateOperator(op_def, &ws);
      EXPECT_NE(nullptr, op.get());
      EXPECT_EQ(
          static_cast<JustTest*>(op.get())->type(),
          "JustTestWithNonStandardIsTestArg");
  */
}
