crate::ix!();

pub trait GetExecutorHelper {

    #[inline] fn get_executor_helper(&self) -> *mut ExecutorHelper {
        
        todo!();
        /*
            return helper_;
        */
    }
}

pub trait SetExecutorHelper {

    #[inline] fn set_executor_helper(&mut self, helper: *mut ExecutorHelper)  {
        
        todo!();
        /*
            helper_ = helper;
        */
    }
}
