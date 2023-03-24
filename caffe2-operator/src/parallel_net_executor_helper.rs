crate::ix!();

pub struct ParallelNetExecutorHelper {
    base: ExecutorHelper,
    net:  *mut ParallelNet,
}

impl ParallelNetExecutorHelper {

    pub fn new(net: *mut ParallelNet) -> Self {
    
        todo!();
        /*
            : net_(net)
        */
    }
    
    #[inline] pub fn get_pool(&self, option: &DeviceOption) -> *mut dyn TaskThreadPoolBaseInterface {
        
        todo!();
        /*
            return net_->Pool(option);
        */
    }
    
    #[inline] pub fn get_operators(&self) -> Vec<*mut OperatorStorage> {
        
        todo!();
        /*
            return net_->GetOperators();
        */
    }
    
    #[inline] pub fn get_num_workers(&self) -> i32 {
        
        todo!();
        /*
            return net_->num_workers_;
        */
    }
}
