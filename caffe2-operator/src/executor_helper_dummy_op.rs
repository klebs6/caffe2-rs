crate::ix!();

pub const kTestPoolSize: i32 = 4;

pub struct ExecutorHelperDummyOp {
    base: OperatorStorage,
}

impl ExecutorHelperDummyOp {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : OperatorStorage(operator_def, ws)
        */
    }
    
    #[inline] pub fn run(&mut self, stream_id: i32) -> bool {
        
        todo!();
        /*
            auto helper = GetExecutorHelper();
        CAFFE_ENFORCE(helper);
        auto pool = helper->GetPool(device_option());
        CAFFE_ENFORCE(pool);
        auto pool_size = pool->size();
        CAFFE_ENFORCE_EQ(pool_size, kTestPoolSize);
        return true;
        */
    }
}

register_cpu_operator!{ExecutorHelperDummy, ExecutorHelperDummyOp}

