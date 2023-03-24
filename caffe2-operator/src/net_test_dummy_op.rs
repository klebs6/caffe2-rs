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

static counter: AtomicI32 = AtomicI32::new(0);

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
    fail: bool,
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

register_cpu_operator!{NetTestDummy, NetTestDummyOp}
register_cuda_operator!{NetTestDummy, NetTestDummyOp}
register_cpu_operator!{NetTestDummy2, NetTestDummyOp}
register_cuda_operator!{NetTestDummy2, NetTestDummyOp}

///---------------
num_inputs!{NetTestDummy, (0,INT_MAX)}

num_outputs!{NetTestDummy, (0,INT_MAX)}

allow_inplace!{NetTestDummy, vec![(0, 0), (1, 1)]}

///---------------
num_inputs!{NetTestDummy2, (0,INT_MAX)}

num_outputs!{NetTestDummy2, (0,INT_MAX)}

allow_inplace!{NetTestDummy2, vec![(1, 0)]}


