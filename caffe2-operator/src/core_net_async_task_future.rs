crate::ix!();

pub struct ParentCounter {
    init_parent_count:  i32,
    parent_count:       Atomic<i32>,
    err_mutex:          parking_lot::RawMutex,
    parent_failed:      AtomicBool,
    err_msg:            String,
}

impl ParentCounter {
    
    pub fn new(init_parent_count: i32) -> Self {
    
        todo!();
        /*
            : init_parent_count_(init_parent_count),
            parent_count(init_parent_count),
            parent_failed(false)
        */
    }
    
    #[inline] pub fn reset(&mut self)  {
        
        todo!();
        /*
            std::unique_lock<std::mutex> lock(err_mutex);
            parent_count = init_parent_count_;
            parent_failed = false;
            err_msg = "";
        */
    }
}

/**
  | Represents the state of AsyncTask execution,
  | that can be queried with
  | 
  | IsCompleted/IsFailed. Callbacks
  | are supported through SetCallback
  | and are called upon future's completion.
  |
  */
pub struct AsyncTaskFuture {
    
    mutex:           parking_lot::RawMutex,
    cv_completed:    std::sync::Condvar,
    completed:       AtomicBool,
    failed:          AtomicBool,
    err_msg:         String,
    callbacks:       Vec<fn(_u0: *const AsyncTaskFuture) -> c_void>,
    parent_counter:  Box<ParentCounter>,
}

impl Default for AsyncTaskFuture {
    
    fn default() -> Self {
        todo!();
        /*
            : completed_(false), failed_(false)
        */
    }
}

impl From<&Vec<*mut AsyncTaskFuture>> for AsyncTaskFuture {

    /**
      | Creates a future completed when all
      | given futures are completed
      |
      */
    fn from(futures: &Vec<*mut AsyncTaskFuture>) -> Self {
    
        todo!();
        /*
            : completed_(false), failed_(false) 

      if (futures.size() > 1) {
        parent_counter_ = std::make_unique<ParentCounter>(futures.size());
        for (auto future : futures) {
          future->SetCallback([this](const AsyncTaskFuture* f) {
            if (f->IsFailed()) {
              std::unique_lock<std::mutex> lock(parent_counter_->err_mutex);
              if (parent_counter_->parent_failed) {
                parent_counter_->err_msg += ", " + f->ErrorMessage();
              } else {
                parent_counter_->parent_failed = true;
                parent_counter_->err_msg = f->ErrorMessage();
              }
            }
            int count = --parent_counter_->parent_count;
            if (count == 0) {
              // thread safe to use parent_counter here
              if (!parent_counter_->parent_failed) {
                SetCompleted();
              } else {
                SetCompleted(parent_counter_->err_msg.c_str());
              }
            }
          });
        }
      } else {
        CAFFE_ENFORCE_EQ(futures.size(), (size_t)1);
        auto future = futures.back();
        future->SetCallback([this](const AsyncTaskFuture* f) {
          if (!f->IsFailed()) {
            SetCompleted();
          } else {
            SetCompleted(f->ErrorMessage().c_str());
          }
        });
      }
        */
    }
}

impl AsyncTaskFuture {

    #[inline] pub fn is_completed(&self) -> bool {
        
        todo!();
        /*
            return completed_;
        */
    }
    
    #[inline] pub fn is_failed(&self) -> bool {
        
        todo!();
        /*
            return failed_;
        */
    }
    
    #[inline] pub fn error_message(&self) -> String {
        
        todo!();
        /*
            return err_msg_;
        */
    }
    
    #[inline] pub fn wait(&self)  {
        
        todo!();
        /*
            std::unique_lock<std::mutex> lock(mutex_);
      while (!completed_) {
        cv_completed_.wait(lock);
      }
        */
    }
    
    #[inline] pub fn set_callback(&mut self, callback: fn(_u0: *const AsyncTaskFuture) -> c_void)  {
        
        todo!();
        /*
            std::unique_lock<std::mutex> lock(mutex_);

      callbacks_.push_back(callback);
      if (completed_) {
        callback(this);
      }
        */
    }
    
    #[inline] pub fn set_completed(&mut self, err_msg: Option<&str>)  {
        
        todo!();
        /*
            std::unique_lock<std::mutex> lock(mutex_);

      CAFFE_ENFORCE(!completed_, "Calling SetCompleted on a completed future");
      completed_ = true;

      if (err_msg) {
        failed_ = true;
        err_msg_ = err_msg;
      }

      for (auto& callback : callbacks_) {
        callback(this);
      }

      cv_completed_.notify_all();
        */
    }

    /**
      | ResetState is called on a completed
      | future, does not reset callbacks to
      | keep task graph structure
      |
      */
    #[inline] pub fn reset_state(&mut self)  {
        
        todo!();
        /*
            std::unique_lock<std::mutex> lock(mutex_);
      if (parent_counter_) {
        parent_counter_->Reset();
      }
      completed_ = false;
      failed_ = false;
      err_msg_ = "";
        */
    }
}
