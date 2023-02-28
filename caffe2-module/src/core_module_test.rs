crate::ix!();

use crate::{
    OperatorStorage,
};

/**
  | An explicitly defined module, testing
  | correctness when we statically link
  | a module
  |
  */
caffe2_module![caffe2_module_test_static, "Static module for testing."];

pub struct Caffe2ModuleTestStaticDummyOp {
    storage: OperatorStorage,
}

impl Caffe2ModuleTestStaticDummyOp {

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


register_cpu_operator!{Caffe2ModuleTestStaticDummy, Caffe2ModuleTestStaticDummyOp}

#[test] fn ModuleTest_StaticModule() {
    todo!();
    /*
      const string name = "caffe2_module_test_static";
      const auto& modules = CurrentModules();
      EXPECT_EQ(modules.count(name), 1);
      EXPECT_TRUE(HasModule(name));

      // LoadModule should not raise an error, since the module is already present.
      LoadModule(name);
      // Even a non-existing path should not cause error.
      LoadModule(name, "/does/not/exist.so");
      EXPECT_EQ(modules.count(name), 1);
      EXPECT_TRUE(HasModule(name));

      // The module will then introduce the Caffe2ModuleTestStaticDummyOp.
      OperatorDef op_def;
      Workspace ws;
      op_def.set_type("Caffe2ModuleTestStaticDummy");
      unique_ptr<OperatorStorage> op = CreateOperator(op_def, &ws);
      EXPECT_NE(nullptr, op.get());
  */
}


#[cfg(caffe2_build_shared_libs)]
#[test] fn ModuleTest_DynamicModule() {
    todo!();
    /*
      const string name = "caffe2_module_test_dynamic";
      const auto& modules = CurrentModules();
      EXPECT_EQ(modules.count(name), 0);
      EXPECT_FALSE(HasModule(name));

      // Before loading, we should not be able to create the op.
      OperatorDef op_def;
      Workspace ws;
      op_def.set_type("Caffe2ModuleTestDynamicDummy");
      EXPECT_THROW(
          CreateOperator(op_def, &ws),
          EnforceNotMet);

      // LoadModule should load the proper module.
      LoadModule(name);
      EXPECT_EQ(modules.count(name), 1);
      EXPECT_TRUE(HasModule(name));

      // The module will then introduce the Caffe2ModuleTestDynamicDummyOp.
      unique_ptr<OperatorStorage> op_after_load = CreateOperator(op_def, &ws);
      EXPECT_NE(nullptr, op_after_load.get());
  */
}
