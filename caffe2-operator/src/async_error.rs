crate::ix!();

pub struct AsyncErrorOp {
    storage:      OperatorStorage,
    context:      CPUContext,
    thread:       Box<std::thread::Thread>,
    throw:        bool,
    fail_in_sync: bool,
    sleep_time_s: i32,
    error_msg:    String,
    cancel:       AtomicBool, // default = ATOMIC_FLAG_INIT
}

impl Drop for AsyncErrorOp {

    fn drop(&mut self) {
        todo!();
        /* 
        if (thread_) {
          thread_->join();
        }
       */
    }
}

register_cpu_operator!{AsyncErrorOp, AsyncErrorOp}

impl AsyncErrorOp {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : Operator<CPUContext>(operator_def, ws),
            throw_(OperatorStorage::GetSingleArgument<bool>("throw", false)),
            fail_in_sync_(
                OperatorStorage::GetSingleArgument<bool>("fail_in_sync", false)),
            sleep_time_s_(OperatorStorage::GetSingleArgument<int>("sleep_time", 1)),
            error_msg_(OperatorStorage::GetSingleArgument<std::string>(
                "error_msg",
                "Error"))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            if (fail_in_sync_) {
          if (throw_) {
            throw std::logic_error(error_msg_);
          } else {
            return false;
          }
        } else {
          if (thread_) {
            thread_->join();
          }
          thread_ = std::make_unique<std::thread>([this]() {
            try {
              std::this_thread::sleep_for(std::chrono::seconds(sleep_time_s_));
              if (throw_) {
                throw std::logic_error(error_msg_);
              } else {
                if (!cancel_.test_and_set()) {
                  event().SetFinished(error_msg_.c_str());
                }
              }
            } catch (...) {
              if (!cancel_.test_and_set()) {
                event().SetFinishedWithException(error_msg_.c_str());
              }
            }
          });
          return true;
        }
        */
    }
    
    #[inline] pub fn has_async_part(&self) -> bool {
        
        todo!();
        /*
            return true;
        */
    }
    
    #[inline] pub fn cancel_async_callback(&mut self)  {
        
        todo!();
        /*
            cancel_.test_and_set();
        */
    }
}
