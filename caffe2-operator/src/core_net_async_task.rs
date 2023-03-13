crate::ix!();


/**
  | AsyncTask represents an asynchronous
  | execution of a chain of ops.
  |
  */
pub struct AsyncTask {
    ops:           Vec<*mut OperatorStorage>,
    device_option: DeviceOption,
    future:        AsyncTaskFuture,
}

impl AsyncTask {
    
    pub fn new(ops: &Vec<*mut OperatorStorage>) -> Self {
    
        todo!();
        /*
            : ops_(ops) 

      CAFFE_ENFORCE(!ops_.empty());
      device_option_ = ops_.front()->device_option();
      for (auto& op : ops_) {
        CAFFE_ENFORCE(IsSameDevice(device_option_, op->device_option()));
      }
      Reset();
        */
    }
    
    #[inline] pub fn handle_chain_error(&mut self, 
        op:             *mut OperatorStorage,
        err_str:        *const u8,
        save_exception: Option<bool>)  
    {
        let save_exception = save_exception.unwrap_or(false);
        
        todo!();
        /*
            std::string err_msg = err_str;
      if (op) {
        err_msg += ",  op " + (op->has_debug_def() ? op->type() : " unknown");
      }
      LOG(ERROR) << err_msg;

      // save error message and exception in chain's Event
      auto last_op = ops_.back();
      if (save_exception) {
        last_op->event().SetFinishedWithException(err_msg.c_str());
      } else {
        last_op->event().SetFinished(err_msg.c_str());
      }

      // set future as completed with an error
      // TODO: exceptions in future
      future_.SetCompleted(err_msg.c_str());
        */
    }
    
    #[inline] pub fn run(&mut self, options: &ExecutionOptions) -> bool {
        
        todo!();
        /*
            // TODO: insert CUDA's async stream waits; tracing and counters
      OperatorStorage* op = nullptr;
      try {
        for (auto op_idx = 0U; op_idx < ops_.size(); ++op_idx) {
          op = ops_[op_idx];
          int stream_id = 0; // TODO: thread local stream id
          if (!op->RunAsync(stream_id)) {
            handleChainError(op, "Failed to execute an op");
            return false;
          }
        }

        if (options.finish_chain_) {
          op = ops_.back();
          op->Finish();
        }

        // set the future as successfully completed or, in case of async CPU,
        // use op's callback
        if (IsCPUDeviceType(device_option_.device_type()) &&
            ops_.back()->HasAsyncPart()) {
          auto& event = ops_.back()->event();
          event.SetCallback([this, &event]() {
            CAFFE_ENFORCE(event.IsFinished());
            if (event.Query() == EventStatus::EVENT_SUCCESS) {
              future_.SetCompleted();
            } else {
              // TODO: support for exceptions
              future_.SetCompleted(event.ErrorMessage().c_str());
            }
          });
        } else {
          future_.SetCompleted();
        }
      } catch (const std::exception& e) {
        handleChainError(op, e.what(), /* save_exception */ true);
        return false;
      } catch (...) {
        handleChainError(
            op,
            "Failed to execute task: unknown error",
            /* save_exception */ true);
        return false;
      }

      return true;
        */
    }
    
    #[inline] pub fn reset(&mut self)  {
        
        todo!();
        /*
            for (auto& op : ops_) {
        op->ResetEvent();
      }
      future_.ResetState();
        */
    }
    
    #[inline] pub fn get_device_option(&self) -> DeviceOption {
        
        todo!();
        /*
            return device_option_;
        */
    }
    
    #[inline] pub fn get_future_mut<'a>(&'a mut self) -> &'a mut AsyncTaskFuture {
        
        todo!();
        /*
            return future_;
        */
    }
    
    #[inline] pub fn get_future<'a>(&'a self) -> &'a AsyncTaskFuture {
        
        todo!();
        /*
            return future_;
        */
    }
}
