crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/core/Allocator.h]

/**
  | dyn std::alloc::Allocator DataPtr is a unique pointer (with an attached
  | deleter and some context for the deleter) to
  | some memory, which also records what device is
  | for its data.
  |
  | nullptr DataPtrs can still have a nontrivial
  | device; this allows us to treat zero-size
  | allocations uniformly with non-zero
  | allocations.
  |
  */
pub struct DataPtr {
    ptr:    UniqueVoidPtr,
    device: Device,
}

impl Default for DataPtr {
    
    /**
      | Choice of CPU here is arbitrary; if there's
      | an "undefined" device we could use that
      | too
      |
      */
    fn default() -> Self {
        todo!();
        /*
        : ptr(),
        : device(DeviceType::CPU),

        
        */
    }
}

impl Deref for DataPtr {

    type Target = *mut c_void;
    
    #[inline] fn deref(&self) -> &Self::Target {
        todo!();
        /*
            return ptr_.get();
        */
    }
}

impl DataPtr {
    
    pub fn new(
        data:   *mut c_void,
        device: Device) -> Self {
    
        todo!();
        /*
        : ptr(data),
        : device(device),

        
        */
    }
    
    pub fn new_with_deleter<A: std::alloc::Allocator>(
        data:        *mut c_void,
        ctx:         *mut c_void,
        ctx_deleter: A,
        device:      Device) -> Self {
    
        todo!();
        /*
        : ptr(data, ctx, ctx_deleter),
        : device(device),

        
        */
    }
    
    pub fn clear(&mut self)  {
        
        todo!();
        /*
            ptr_.clear();
        */
    }
    
    pub fn get(&self)  {
        
        todo!();
        /*
            return ptr_.get();
        */
    }
    
    pub fn get_context(&self)  {
        
        todo!();
        /*
            return ptr_.get_context();
        */
    }
    
    pub fn release_context(&mut self)  {
        
        todo!();
        /*
            return ptr_.release_context();
        */
    }
    
    pub fn move_context<A: std::alloc::Allocator>(&mut self) -> &mut Box<*mut c_void,A> {
        
        todo!();
        /*
            return ptr_.move_context();
        */
    }
    
    pub fn operator_bool(&self) -> bool {
        
        todo!();
        /*
            return static_cast<bool>(ptr_);
        */
    }
    
    pub fn cast_context<T, A: std::alloc::Allocator>(&self, expected_deleter: A) -> *mut T {
    
        todo!();
        /*
            return ptr_.cast_context<T>(expected_deleter);
        */
    }
    
    pub fn get_deleter<A: std::alloc::Allocator>(&self) -> A {
        
        todo!();
        /*
            return ptr_.get_deleter();
        */
    }

    /**
      | Compare the deleter in a DataPtr to expected_deleter.
      | 
      | If it matches, replace the deleter with
      | new_deleter and return true; otherwise,
      | does nothing and returns false.
      | 
      | In general, it is not safe to unconditionally
      | set the deleter on a DataPtr, because
      | you don't know what the deleter is, and
      | thus will have a hard time properly disposing
      | of the deleter without storing the original
      | deleter (this is difficult to do, because
      | Allocator is not a closure, and because
      | the context on DataPtr is only a single
      | word, you generally don't have enough
      | space to store both the original deleter
      | and its context).
      | 
      | However, in some cases, you know /exactly/
      | what the deleter is, and you have a new
      | deleter that manually wraps the old
      | one. In this case, you can safely swap
      | the deleter after asserting that the
      | deleters line up.
      | 
      | What are the requirements on new_deleter?
      | It must still properly dispose of the
      | void* pointer passed in as its argument,
      | where void* is whatever the context
      | of the original deleter is. So in general,
      | you expect the new deleter to look something
      | like this: [](void* ptr) { some_new_stuff(ptr);
      | get_orig_allocator()->raw_deleter(ptr);
      | }
      | 
      | -----------
      | @note
      | 
      | it won't work to close over the original
      | allocator; you don't have enough space
      | to do that! Also, it's unsafe to assume
      | that the passed in pointer in question
      | is the memory pointer in question; it
      | might not be; be sure to read the source
      | code of the Allocator in question to
      | confirm this.
      |
      */
    pub fn compare_exchange_deleter<A1: std::alloc::Allocator, A2: std::alloc::Allocator>(&mut self, 
        expected_deleter: A1,
        new_deleter:      A2) -> bool {
        
        todo!();
        /*
            return ptr_.compare_exchange_deleter(expected_deleter, new_deleter);
        */
    }
    
    pub fn device(&self) -> Device {
        
        todo!();
        /*
            return device_;
        */
    }

    /**
      | Unsafely mutates the device on a DataPtr.
      | Under normal use, you should never actually
      | need to call this function.
      |
      | We need this for the implementation of the
      | hack detailed in Note [Masquerading as Cuda]
      */
    pub fn unsafe_set_device(&mut self, device: Device)  {
        
        todo!();
        /*
            device_ = device;
        */
    }
}

/**
  | NB: Device is NOT tested for here; a Cuda
  | nullptr is as much a nullptr as a CPU nullptr
  |
  */
lazy_static!{
    /*
    inline bool operator==(const DataPtr& dp, nullptr_t)  {
      return !dp;
    }
    inline bool operator==(nullptr_t, const DataPtr& dp)  {
      return !dp;
    }
    inline bool operator!=(const DataPtr& dp, nullptr_t)  {
      return dp;
    }
    inline bool operator!=(nullptr_t, const DataPtr& dp)  {
      return dp;
    }
    */
}

/**
  | Note [raw_allocate/raw_deallocate and Thrust]
  | ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  | Thrust's support for custom allocators requires
  | us to write something like this:
  |
  |  class ThrustAllocator {
  |    char* allocate(size_t);
  |    void deallocate(char*, size_t);
  |  };
  |
  | This is not good for our unique_ptr based
  | allocator interface, as there is no way to get
  | to the context when we free.
  |
  | However, in some cases the context is exactly
  | the same as the data pointer.  In this case, we
  | can support the "raw" allocate and deallocate
  | interface.  This is what raw_deleter signifies.
  | By default, it returns a nullptr, which means
  | that the raw interface is not implemented.  Be
  | sure to implement it whenever possible, or the
  | raw interface will incorrectly reported as
  | unsupported, when it is actually possible.
  */
pub struct Allocator {

}

impl Allocator {
    
    pub fn raw_allocate(&mut self, n: usize)  {
        
        todo!();
        /*
            auto dptr = allocate(n);
        AT_ASSERT(dptr.get() == dptr.get_context());
        return dptr.release_context();
        */
    }
    
    pub fn raw_deallocate(&mut self, ptr: *mut c_void)  {
        
        todo!();
        /*
            auto d = raw_deleter();
        AT_ASSERT(d);
        d(ptr);
        */
    }
}

/**
  | This context is used to generate DataPtr which
  | have arbitrary function deleters associated
  | with them.
  |
  | In some user facing functions, we give
  | a (user-friendly) interface for constructing
  | tensors from external data which take an
  | arbitrary function deleter.
  |
  | Grep for InefficientStdFunctionContext to find
  | these occurrences.
  |
  | This context is inefficient because we have to
  | do a dynamic allocation
  | InefficientStdFunctionContext, on top of the
  | dynamic allocation which is implied by function
  | itself.
  |
  */
pub struct InefficientStdFunctionContext<A: std::alloc::Allocator> {
    ptr: Box<*mut c_void, A>,
}

impl<A: std::alloc::Allocator> InefficientStdFunctionContext<A> {
    
    pub fn new(ptr: Box<*mut c_void,A>) -> Self {
    
        todo!();
        /*
        : ptr(move(ptr)),
        */
    }
    
    pub fn make_data_ptr(&mut self, 
        ptr:     *mut c_void,
        deleter: &fn(_0: *mut c_void) -> (),
        device:  Device) -> DataPtr {
        
        todo!();
        /*
            return {
          ptr,
          new InefficientStdFunctionContext({ptr, deleter}),
          &deleteInefficientStdFunctionContext,
          device};
        */
    }
}

pub struct AllocatorRegisterer<const t: DeviceType> {

}

impl<const t: DeviceType> AllocatorRegisterer<t> {
    
    pub fn new(alloc: *mut Allocator) -> Self {
    
        todo!();
        /*


            SetAllocator(t, alloc);
        */
    }
}

/// An interface for reporting thread local memory
/// usage per device
/// 
/// Negative alloc_size corresponds to freeing of
/// the memory
///
pub trait MemoryReportingInfoBaseInterface:
fmt::Debug
+ DebugInfoBaseInterface
+ ReportMemoryUsage
+ MemoryProfilingEnabled {}

pub trait ReportMemoryUsage {

    fn report_memory_usage(&mut self, 
        ptr:        *mut c_void,
        alloc_size: i64,
        device:     Device);
}

pub trait MemoryProfilingEnabled {

    fn memory_profiling_enabled(&self) -> bool;
}

//-------------------------------------------[.cpp/pytorch/c10/core/Allocator.cpp]

pub fn delete_inefficient_std_function_context(ptr: *mut c_void)  {
    
    todo!();
        /*
            delete static_cast<InefficientStdFunctionContext*>(ptr);
        */
}

lazy_static!{
    /*
    Allocator* allocator_array[COMPILE_TIME_MAX_DEVICE_TYPES];
     uint8_t allocator_priority[COMPILE_TIME_MAX_DEVICE_TYPES] = {0};
    */
}

/**
  | Set the allocator for DeviceType `t`.
  | The passed in allocator pointer is expected
  | to have static lifetime; this function
  | does NOT take ownership of the raw pointer.
  | (The reason for this is to prevent existing
  | pointers to an allocator of a particular
  | device from being invalidated when
  | 
  | SetAllocator is called.)
  | 
  | Also note that this is not thread-safe,
  | and we assume this function will only
  | be called during initialization.
  | 
  | The 'priority' flag is introduced when
  | we want to overwrite the default allocator,
  | since the allocators are set statically.
  | The default priority is 0, which means
  | the lowest. Only higher or equal priority
  | can overwrite existing ones.
  |
  */
pub fn set_allocator(
    t:        DeviceType,
    alloc:    *mut Allocator,
    priority: Option<u8>)  {

    let priority: u8 = priority.unwrap_or(0);

    todo!();
        /*
            if (priority >= allocator_priority[static_cast<int>(t)]) {
        allocator_array[static_cast<int>(t)] = alloc;
        allocator_priority[static_cast<int>(t)] = priority;
      }
        */
}

pub fn get_allocator(t: &DeviceType) -> *mut Allocator {
    
    todo!();
        /*
            auto* alloc = allocator_array[static_cast<int>(t)];
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(alloc, "Allocator for ", t, " is not set.");
      return alloc;
        */
}

pub fn memory_profiling_enabled() -> bool {
    
    todo!();
        /*
            auto* reporter_ptr = static_cast<MemoryReportingInfoBase*>(
          ThreadLocalDebugInfo::get(DebugInfoKind::PROFILER_STATE));
      return reporter_ptr && reporter_ptr->memoryProfilingEnabled();
        */
}

pub fn report_memory_usage_to_profiler(
    ptr:        *mut c_void,
    alloc_size: i64,
    device:     Device)  {

    todo!();
        /*
            auto* reporter_ptr = static_cast<MemoryReportingInfoBase*>(
          ThreadLocalDebugInfo::get(DebugInfoKind::PROFILER_STATE));
      if (reporter_ptr) {
        reporter_ptr->reportMemoryUsage(ptr, alloc_size, device);
      }
        */
}
