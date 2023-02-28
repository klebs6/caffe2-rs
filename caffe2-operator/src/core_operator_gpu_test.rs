crate::ix!();

///--------------------------
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
            return "BASE";
        */
    }
}

///------------------------
pub struct JustTestCUDA {
    base: JustTest,
}

impl JustTestCUDA {

    #[inline] pub fn run(&mut self, stream_id: i32) -> bool {
        
        todo!();
        /*
            return true;
        */
    }
    
    #[inline] pub fn typename(&mut self) -> String {
        
        todo!();
        /*
            return "CUDA";
        */
    }
}

///-------------------------
pub struct JustTestCUDNN {
    base: JustTest,
}

impl JustTestCUDNN {

    #[inline] pub fn run(&mut self, stream_id: i32) -> bool {
        
        todo!();
        /*
            return true;
        */
    }
    
    #[inline] pub fn typename(&mut self) -> String {
        
        todo!();
        /*
            return "CUDNN";
        */
    }
}

num_inputs!{JustTest, (0,1)}

num_outputs!{JustTest, (0,1)}

register_cuda_operator!{JustTest, JustTestCUDA}

register_cudnn_operator!{JustTest, JustTestCUDNN}

#[test] fn EnginePrefTest_GPUDeviceDefaultPreferredEngines() {
    todo!();
    /*
      if (!HasCudaGPU())
        return;
      OperatorDef op_def;
      Workspace ws;
      op_def.mutable_device_option()->set_device_type(PROTO_CUDA);
      op_def.set_type("JustTest");

      {
        const auto op = CreateOperator(op_def, &ws);
        EXPECT_NE(nullptr, op.get());
        // CUDNN should be taken as it's in the default global preferred engines
        // list
        EXPECT_EQ(static_cast<JustTest*>(op.get())->type(), "CUDNN");
      }
  */
}
