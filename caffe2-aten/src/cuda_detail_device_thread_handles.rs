/*!
  | Some stateful GPU libraries, such as cuDNN,
  | cuBLAS, use handles to store states. These
  | handles are tied to device, and these libraries
  | requires/recommends not to share handles across
  | host threads.
  |
  | These libraries recommend using one handle per
  | host thread. We may not want to do this because
  | threads are relatively light-weight, but
  | creating and destroying handles is expensive
  | (destroying the handle causes
  | synchronizations). DataParallel, for example,
  | creates new threads for each forward pass.
  |
  | This file implements a handle pool
  | mechanism. The handle pool returns handles on
  | demand as threads request them. If all existing
  | handles in the pool are in use, it creates
  | a new one. As threads terminate, they release
  | handles back into the pool. In this way, the
  | handle pool never creates more handles than the
  | high-water mark of active threads, so it's
  | efficient with DataParallel.
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/cuda/detail/DeviceThreadHandles.h]

pub struct DeviceThreadHandlePoolHandle {

    handle: DeviceThreadHandlePoolHandle,

    /*
    // std::vector.emplace() and push_back() may route through temporaries and call
    // copy/move constructors along the way.  If this is the case, we don't want
    // the destructors of temporaries to call cudnnDestroy on the handle.
    // We can achieve safety (for the narrow case of stashing within std::vectors)
    // by making Handle moveable but not copyable, and transferring handle ownership
    // to the latest constructed object.  This is not a substitute for full-blown
    // reference counting, but reference counting may be overkill here.
    // Another alternative is to wrap the saved Handles in unique_ptrs, i.e.,
    // unordered_map<int, vector<unique_ptr<Handle>>> created_handles;
    DeviceThreadHandlePoolHandle(const DeviceThreadHandlePoolHandle& rhs) = delete;
    */
}

impl Drop for DeviceThreadHandlePoolHandle {

    fn drop(&mut self) {
        todo!();
        /*
            if(handle) Destroy(handle);
        */
    }
}

impl DeviceThreadHandlePoolHandle {
    
    pub fn new(create: bool) -> Self {
        let create: bool = create.unwrap_or(false);
        todo!();
        /*
        : handle(nullptr),

            if(create) Create(&handle);
        */
    }
    
    /// Following https://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom
    pub fn new(rhs: DeviceThreadHandlePoolHandle) -> Self {
    
        todo!();
        /*


            : DeviceThreadHandlePoolHandle() 
            std::swap(handle, rhs.handle);
        */
    }
    
    /// operator= takes argument by value
    pub fn assign_from(&mut self, rhs: DeviceThreadHandlePoolHandle) -> &mut DeviceThreadHandlePoolHandle {
        
        todo!();
        /*
            std::swap(handle, rhs.handle); return *this;
        */
    }
}

/// PoolWindow lazily creates and caches the
/// handles that a particular thread is using, so
/// in the common case handle access doesn't incur
/// either handle creation or a mutex lock.
////
pub struct PoolWindow {

    /**
      | Stores the per-device handles currently
      | owned by this thread
      |
      */
    my_handles:  HashMap<i32,Handle>,
    weak_parent: Weak<DeviceThreadHandlePool>,
}

impl Drop for PoolWindow {

    fn drop(&mut self) {
        todo!();
        /*
            release();
        */
    }
}

impl PoolWindow {

    pub fn new(parent: Arc<DeviceThreadHandlePool>) -> Self {
    
        todo!();
        /*
        : weak_parent(move(parent)),

        
        */
    }
    
    pub fn reserve(&mut self, device: i32) -> Handle {
        
        todo!();
        /*
            // If this thread already has a handle for this device, return it
            if(my_handles.find(device) != my_handles.end())
                return my_handles[device];

            // otherwise, either grab a handle from the pool if one is available,
            // or if not, create a new one.
            auto parent = weak_parent.lock();
            TORCH_CHECK(parent, "Cannot create handle during program termination");
            lock_guard<mutex> guard(parent->mutex);

            if(parent->available_handles[device].size() > 0)
            {
                my_handles[device] = parent->available_handles[device].back();
                parent->available_handles[device].pop_back();
            }
            else
            {
                // In local testing, I do observe that emplace_back sometimes routes through temporaries
                // that incur move-constructor and destructor calls.  See comments in Handle above.
                parent->created_handles[device].emplace_back(true /*create*/);
                my_handles[device] = parent->created_handles[device].back().handle;
            }

            return my_handles[device];
        */
    }
    
    /**
      | Called by the destructor. Releases
      | this thread's handles back into the
      | pool.
      |
      */
    pub fn release(&mut self)  {
        
        todo!();
        /*
            if(my_handles.size() > 0) {
                auto parent = weak_parent.lock();
                if (!parent) {
                    // If this thread exits after atexit handlers have completed, the
                    // cuda context itself may be invalid, so we must leak the handles.
                    return;
                }

                lock_guard<mutex> guard(parent->mutex);
                for(auto d_h : my_handles)
                    parent->available_handles[d_h.first].push_back(d_h.second);
            }
        */
    }
}

//template <typename Handle_t, void Create(Handle_t *), void Destroy(Handle_t)>
//
//: public std::enable_shared_from_this<DeviceThreadHandlePool<Handle_t, Create, Destroy>> {
pub struct DeviceThreadHandlePool {

    mutex: parking_lot::RawMutex,

    /**
      | Handles are lazily created as different
      | threads request them, but are never
      | destroyed until the end of the process.
      |
      | The maximum number of handles this process
      | will create for each device is equal to the
      | high-water mark of the number of
      | concurrently active threads that request
      | handles for that device.
      |
      | When threads terminate, they release their
      | handles back into the pool for reuse.
      |
      | Otherwise, new handles would be created
      | every time new threads were spawned,
      | resulting in poor performance for Python
      | modules that repeatedly or frequently
      | spawned new sets of threads (like
      | DataParallel, which creates a new set of
      | threads for each forward pass).
      |
      | To prevent potential deadlocks, we
      | explicitly choose not to cap the number of
      | handles that are created per device.
      |
      | Example of danger: If we cap the max
      | handles at 4, and 5 threads are sharing
      | a device, only 4 can make forward progress
      | at any time. The other 4 will not release
      | their handles until they exit, so the fifth
      | cannot make progress until then.
      |
      | This is not a problem...UNLESS all
      | 5 threads attempt some sort of
      | synchronization at an intermediate point
      | (ie, before any of them have exited).
      |
      | We have no way to anticipate or enforce
      | that user threads will not attempt such
      | intermediate synchronization.
      |
      | The only way to ensure safety is to avoid
      | imposing a cap on the number of handles.
      */
    created_handles:   HashMap<i32,Vec<Handle>>,
    available_handles: HashMap<i32,Vec<Handle>>,
}

impl DeviceThreadHandlePool {
    
    /**
      | Warning:
      | 
      | If you want to change this function,
      | be aware that this function will be called
      | by multiple threads and there is no mutex
      | guarding the call of this function,
      | so make sure your implementation is
      | thread-safe.
      |
      */
    pub fn new_pool_window(&mut self) -> *mut PoolWindow {
        
        todo!();
        /*
            // The returned pointer will be owned by a thread local variable
            // so that different threads does not share the same PoolWindow.
            return new PoolWindow(this->shared_from_this());
        */
    }
}
