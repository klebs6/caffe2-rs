crate::ix!();

/**
  | first int key - device id, second - pool
  | size, one pool per (device, size)
  |
  */
pub type PoolsMap = HashMap<i32,HashMap<i32,Arc<dyn TaskThreadPoolBaseInterface>>>;

pub struct AsyncNetExecutorHelper {
    base: ExecutorHelper,
    net:  *mut AsyncNetBase,
}

impl AsyncNetExecutorHelper {
    
    pub fn new(net: *mut AsyncNetBase) -> Self {
    
        todo!();
        /*
            : net_(net)
        */
    }
    
    #[inline] pub fn get_pool(&self, option: &DeviceOption) -> Arc<dyn TaskThreadPoolBaseInterface> {
        
        todo!();
        /*
            return net_->pool(option);
        */
    }
}
