crate::ix!();


lazy_static!{
    static ref counter: AtomicI32 = AtomicI32::new(0);
}

/**
  | A net test dummy op that does nothing but
  | scaffolding.
  |
  | Here, we inherit from OperatorStorage because
  | we instantiate on both CPU and GPU.
  |
  | In general, you want to only inherit from
  | Operator<Context>.
  */
pub struct NetTestDummyOp {
    base: OperatorStorage,
    fail:  bool,
}

impl NetTestDummyOp {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : OperatorStorage(operator_def, ws),
            fail_(OperatorStorage::GetSingleArgument<bool>("fail", false))
        */
    }
    
    #[inline] pub fn run(&mut self, stream_id: i32) -> bool {
        
        todo!();
        /*
            if (fail_) {
          return false;
        }
        counter.fetch_add(1);
        return true;
        */
    }

    /// Simulate CUDA operator behavior
    #[inline] pub fn has_async_part(&self) -> bool {
        
        todo!();
        /*
            return debug_def().device_option().device_type() == PROTO_CUDA;
        */
    }
    
    #[inline] pub fn supports_async_scheduling(&self) -> bool {
        
        todo!();
        /*
            return debug_def().device_option().device_type() == PROTO_CUDA;
        */
    }
}

///------------------------
register_cpu_operator!{NetTestDummy,   NetTestDummyOp}

register_cuda_operator!{NetTestDummy,  NetTestDummyOp}

num_inputs!{NetTestDummy, (0,INT_MAX)}

num_outputs!{NetTestDummy, (0,INT_MAX)}

allow_inplace!{NetTestDummy, vec![(0, 0), (1, 1)]}

///------------------------
register_cpu_operator!{NetTestDummy2,  NetTestDummyOp}

register_cuda_operator!{NetTestDummy2, NetTestDummyOp}

num_inputs!{NetTestDummy2, (0,INT_MAX)}

num_outputs!{NetTestDummy2, (0,INT_MAX)}

allow_inplace!{NetTestDummy2, vec![(1, 0)]}

///------------------------
#[inline] pub fn test_execution(net: &mut Box<NetBase>, num_ops: i32)  {
    
    todo!();
    /*
        // Run 100 times
      for (int i = 0; i < 100; i++) {
        counter.exchange(0);
        net.get()->Run();
        ASSERT_EQ(num_ops, counter.load());
      }
    */
}

#[inline] pub fn check_chaining_and_run(spec: *const u8, expected: &ExecutionChains)  {
    
    todo!();
    /*
        Workspace ws;
      ws.CreateBlob("in");
      NetDef net_def;
      CAFFE_ENFORCE(TextFormat::ParseFromString(spec, &net_def));
      {
        net_def.set_num_workers(4);
        std::unique_ptr<NetBase> net(CreateNet(net_def, &ws));
        auto* dag = dynamic_cast_if_rtti<AsyncNetBase*>(net.get());
        CHECK_NOTNULL(dag);
        const auto& chains = dag->TEST_execution_chains();
        EXPECT_EQ(chains, expected);
        testExecution(net, net_def.op().size());
      }
    */
}


#[test] fn net_test_disabled_chaining_for_different_devices() {
    todo!();
    /*
      const auto spec = R"DOC(
            name: "example"
            type: "dag"
            external_input: "in"
            op {
              input: "in"
              output: "hidden"
              type: "NetTestDummy"
            }
            op {
              input: "hidden"
              output: "out"
              type: "NetTestDummy"
              device_option {
                device_type: 1
              }
            }
            op {
              input: "out"
              output: "out2"
              type: "NetTestDummy"
              device_option {
                device_type: 1
              }
            }
            op {
              input: "out2"
              output: "out3"
              type: "NetTestDummy"
              device_option {
                device_type: 1
                device_id: 1
              }
            }
    )DOC";
      if (HasCudaGPU() && NumCudaDevices() >= 2) {
        checkChainingAndRun(spec, {{0, {0, 1, 2}}, {3, {3}}});
      }
  */
}
