crate::ix!();

/**
  | Use this to implement a Observer using
  | the Observer Pattern template.
  |
  */
pub struct ObserverBase<T> {
    subject: *mut T,
}

impl<T> ObserverBase<T> {
    
    pub fn new(subject: *mut T) -> Self {
    
        todo!();
        /*
            : subject_(subject)
        */
    }
    
    #[inline] pub fn start(&mut self)  {
        
        todo!();
        /*
        
        */
    }
    
    #[inline] pub fn stop(&mut self)  {
        
        todo!();
        /*
        
        */
    }
    
    #[inline] pub fn debug_info(&mut self) -> String {
        
        todo!();
        /*
            return "Not implemented.";
        */
    }
    
    #[inline] pub fn subject(&self) -> *mut T {
        
        todo!();
        /*
            return subject_;
        */
    }
    
    #[inline] pub fn rnn_copy(&self, subject: *mut T, rnn_order: i32) -> Box<ObserverBase<T>> {
        
        todo!();
        /*
            return nullptr;
        */
    }
}

/**
  | Inherit to make your class observable.
  |
  */
pub struct Observable<T> {
    
    /**
      | an on-stack cache for fast iteration;
      |
      | ideally, inside StartAllObservers and
      | StopAllObservers, we should never access
      | observers_list_
      */
    observer_cache: *mut Observer<T>,
    num_observers:  usize, // default = 0
    observers_list: Vec<Box<Observer<T>>>,
}


impl<T> Observable<T> {
    
    #[inline] pub fn num_observers(&mut self) -> usize {
        
        todo!();
        /*
            return num_observers_;
        */
    }
}

pub type Observer<T> = ObserverBase<T>;

impl<T> Observer<T> {
    
    /**
      | Returns a reference to the observer
      | after addition.
      |
      */
    #[inline] pub fn attach_observer(&mut self, observer: Box<Observer<T>>) -> *const Observer<T> {
        
        todo!();
        /*
            CAFFE_ENFORCE(observer, "Couldn't attach a null observer.");
        std::unordered_set<const Observer*> observers;
        for (auto& ob : observers_list_) {
          observers.insert(ob.get());
        }

        const auto* observer_ptr = observer.get();
        if (observers.count(observer_ptr)) {
          return observer_ptr;
        }
        observers_list_.push_back(std::move(observer));
        UpdateCache();

        return observer_ptr;
        */
    }
    
    /**
      | Returns a unique_ptr to the removed
      | observer. If not found, return a nullptr
      |
      */
    #[inline] pub fn detach_observer(&mut self, observer_ptr: *const Observer<T>) -> Box<Observer<T>> {
        
        todo!();
        /*
            for (auto it = observers_list_.begin(); it != observers_list_.end(); ++it) {
          if (it->get() == observer_ptr) {
            auto res = std::move(*it);
            observers_list_.erase(it);
            UpdateCache();
            return res;
          }
        }
        return nullptr;
        */
    }
    
    #[inline] pub fn start_observer(observer: *mut Observer<T>)  {
        
        todo!();
        /*
            try {
          observer->Start();
        } catch (const std::exception& e) {
          LOG(ERROR) << "Exception from observer: " << e.what();
        } catch (...) {
          LOG(ERROR) << "Exception from observer: unknown";
        }
        */
    }
    
    #[inline] pub fn stop_observer(observer: *mut Observer<T>)  {
        
        todo!();
        /*
            try {
          observer->Stop();
        } catch (const std::exception& e) {
          LOG(ERROR) << "Exception from observer: " << e.what();
        } catch (...) {
          LOG(ERROR) << "Exception from observer: unknown";
        }
        */
    }
    
    #[inline] pub fn update_cache(&mut self)  {
        
        todo!();
        /*
            num_observers_ = observers_list_.size();
        if (num_observers_ != 1) {
          // we cannot take advantage of the cache
          return;
        }
        observer_cache_ = observers_list_[0].get();
        */
    }
    
    #[inline] pub fn start_all_observers(&mut self)  {
        
        todo!();
        /*
            // do not access observers_list_ unless necessary
        if (num_observers_ == 0) {
          return;
        } else if (num_observers_ == 1) {
          StartObserver(observer_cache_);
        } else {
          for (auto& observer : observers_list_) {
            StartObserver(observer.get());
          }
        }
        */
    }
    
    #[inline] pub fn stop_all_observers(&mut self)  {
        
        todo!();
        /*
            // do not access observers_list_ unless necessary
        if (num_observers_ == 0) {
          return;
        } else if (num_observers_ == 1) {
          StopObserver(observer_cache_);
        } else {
          for (auto& observer : observers_list_) {
            StopObserver(observer.get());
          }
        }
        */
    }
}
