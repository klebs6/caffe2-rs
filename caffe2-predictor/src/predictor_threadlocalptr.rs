/*!
  | thread_local pointer in C++ is a per
  | thread pointer. However, sometimes
  | we want to have a thread local state that
  | is per thread and also per instance.
  | e.g. we have the following class:
  | 
  | class A { ThreadLocalPtr<int> x; }
  | 
  | We would like to have a copy of x per thread
  | and also per instance of class A
  | 
  | This can be applied to storing per instance
  | thread local state of some class, when
  | we could have multiple instances of
  | the class in the same thread.
  | 
  | We implemented a subset of functions
  | in folly::ThreadLocalPtr that's enough
  | to support BlackBoxPredictor.
  |
  */

crate::ix!();

/**
  | Map of object pointer to instance in
  | each thread to achieve per thread(using
  | thread_local) per object(using the
  | map) thread local pointer
  |
  */
pub type UnsafeThreadLocalMap = HashMap<*mut ThreadLocalPtrImpl,Arc<c_void>>;

pub type UnsafeAllThreadLocalHelperVector = Vec<*mut ThreadLocalHelper>;

/**
  | A thread safe vector of all ThreadLocalHelper,
  | this will be used to encapuslate the
  | locking in the APIs for the changes to
  | the global AllThreadLocalHelperVector instance.
  |
  */
pub struct AllThreadLocalHelperVector {
    vector:  UnsafeAllThreadLocalHelperVector,
    mutex:   parking_lot::RawMutex,
}

/**
  | ThreadLocalHelper is per thread
  |
  */
pub struct ThreadLocalHelper {

    /// mapping of object -> ptr in each thread
    mapping:  UnsafeThreadLocalMap,
    mutex:    parking_lot::RawMutex,
}

/**
  | ThreadLocalPtrImpl is per object
  |
  */
pub struct ThreadLocalPtrImpl { } 

///---------------------------------------
pub struct ThreadLocalPtr<T> {
    impl_:  ThreadLocalPtrImpl,
    phantom: PhantomData<T>,
}

impl<T> ThreadLocalPtr<T> {

    #[inline] pub fn get_mut(&mut self) -> *mut T {
        
        todo!();
        /*
            return impl_.get<T>();
        */
    }
    
    #[inline] pub fn get(&self) -> *mut T {
        
        todo!();
        /*
            return impl_.get<T>();
        */
    }
    
    #[inline] pub fn reset(&mut self, ptr: Option<Box<T>>)  {

        todo!();
        /*
            impl_.reset<T>(ptr.release());
        */
    }
}

impl<T> DerefMut for ThreadLocalPtr<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        todo!();
    }

    /*
  T* operator-> () {
    return get();
  }

  T& operator*() {
    return *get();
  }
    */
}

impl<T> Deref for ThreadLocalPtr<T> {

    type Target = T;

    fn deref(&self) -> &Self::Target {
        todo!();
    }

    /*
  T* operator-> () const {
    return get();
  }

  T& operator*() const {
    return *get();
  }
    */
}

/// meyer's singleton
#[inline] pub fn get_all_thread_local_helper_vector() -> *mut AllThreadLocalHelperVector {
    
    todo!();
    /*
        // leak the pointer to avoid dealing with destruction order issues
      static auto* instance = new AllThreadLocalHelperVector();
      return instance;
    */
}


#[inline] pub fn get_thread_local_helper() -> *mut ThreadLocalHelper {
    
    todo!();
    /*
        static thread_local ThreadLocalHelper instance;
      return &instance;
    */
}

impl AllThreadLocalHelperVector {
    
    /// Add a new ThreadLocalHelper to the vector
    #[inline] pub fn push_back(&mut self, helper: *mut ThreadLocalHelper)  {
        
        todo!();
        /*
            std::lock_guard<std::mutex> lg(mutex_);
      vector_.push_back(helper);
        */
    }
    
    /// Erase a ThreadLocalHelper to the vector
    #[inline] pub fn erase(&mut self, helper: *mut ThreadLocalHelper)  {
        
        todo!();
        /*
            std::lock_guard<std::mutex> lg(mutex_);
      vector_.erase(
          std::remove(vector_.begin(), vector_.end(), helper), vector_.end());
        */
    }

    /**
      | Erase object in all the helpers stored in
      | vector
      |
      | Called during destructor of
      | a ThreadLocalPtrImpl
      */
    #[inline] pub fn erase_tlp(&mut self, ptr: *mut ThreadLocalPtrImpl)  {
        
        todo!();
        /*
            std::lock_guard<std::mutex> lg(mutex_);
      for (auto* ins : vector_) {
        ins->erase(ptr);
      }
        */
    }
}

impl ThreadLocalHelper {
    /// ThreadLocalHelper
    pub fn new() -> Self {
    
        todo!();
        /*
            getAllThreadLocalHelperVector()->push_back(this);
        */
    }
}

impl Drop for ThreadLocalHelper {

    /**
      | When the thread dies, we want to clean
      | up this* in AllThreadLocalHelperVector
      |
      */
    fn drop(&mut self) {
        todo!();
        /* 
          getAllThreadLocalHelperVector()->erase(this);
         */
    }
}

impl ThreadLocalHelper {
    
    /**
      | Insert a (object, ptr) pair into the
      | thread local map
      |
      */
    #[inline] pub fn insert(&mut self, tl_ptr: *mut ThreadLocalPtrImpl, ptr: Arc<c_void>)  {
        
        todo!();
        /*
            std::lock_guard<std::mutex> lg(mutex_);
      mapping_.insert(std::make_pair(tl_ptr, std::move(ptr)));
        */
    }
    
    /// Get the ptr by object
    #[inline] pub fn get(&mut self, key: *mut ThreadLocalPtrImpl) -> *mut c_void {
        
        todo!();
        /*
            /* Grabbing the mutex for the thread local map protecting the case
       * when other object exits(~ThreadLocalPtrImpl()), and removes the
       * element in the map, which will change the iterator returned
       * by find.
       */
      std::lock_guard<std::mutex> lg(mutex_);
      auto it = mapping_.find(key);

      if (it == mapping_.end()) {
        return nullptr;
      } else {
        return it->second.get();
      }
        */
    }
    
    /**
      | Erase the ptr associated with the object
      | in the map
      |
      */
    #[inline] pub fn erase(&mut self, key: *mut ThreadLocalPtrImpl)  {
        
        todo!();
        /*
            std::lock_guard<std::mutex> lg(mutex_);
      mapping_.erase(key);
        */
    }
}

impl Drop for ThreadLocalPtrImpl {

    /**
      | In the case when object dies first, we
      | want to clean up the states in all child
      | threads
      |
      */
    fn drop(&mut self) {
        todo!();
        /* 
      getAllThreadLocalHelperVector()->erase_tlp(this);
 */
    }
}

impl ThreadLocalPtrImpl {
    
    #[inline] pub fn get<T>(&mut self) -> *mut T {
    
        todo!();
        /*
            return static_cast<T*>(getThreadLocalHelper()->get(this));
        */
    }
    
    #[inline] pub fn reset<T>(&mut self, new_ptr: *mut T)  {
    
        todo!();
        /*
            VLOG(2) << "In Reset(" << newPtr << ")";
        auto* wrapper = getThreadLocalHelper();
        // Cleaning up the objects(T) stored in the ThreadLocalPtrImpl in the thread
        wrapper->erase(this);
        if (newPtr != nullptr) {
          std::shared_ptr<void> sharedPtr(newPtr);
          // Deletion of newPtr is handled by shared_ptr
          // as it implements type erasure
          wrapper->insert(this, std::move(sharedPtr));
        }
        */
    }
}
