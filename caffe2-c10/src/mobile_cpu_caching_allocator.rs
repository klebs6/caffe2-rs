/*!
 | CPUCachingAllocator:
 |
 | DISCLAIMER:
 |
 | This is subject to change (beta) and only
 | supported on mobile builds.
 |
 | If code snippet such as in 'Usage pattern' is
 | used outside of mobile build you will not
 | observe the intended behavior.
 |
 | See below for more information.
 |
 | Why?
 |
 | It has been observed that some mobile
 | platforms, such as pixel 3, return memory
 | aggressively to the system. This results in
 | page faults in some cases and ends up hurting
 | performance. This caching allocator aims to
 | address that. 
 |
 | Furthermore it also allows users to specify
 | their own allocator by implementing
 | allocate/free virtual interfaces. 
 |
 | What are the cons? There are some cons that were
 | observed where use of caching allocator led to
 | worse performance on some platforms. 
 |
 | Reason being that the caching mechanism used by
 | this allocator left us worse off compared to the
 | corresponding platform's tuned memory
 | allocator. 
 |
 | In that case it seemed better to not use this
 | allocator. 
 |
 | Note there are some ideas to fix this in the works.
 |
 | Usage:
 |
 | Usage pattern:
 |
 | Instantiate and own the caching allocator.
 |
 | unique_ptr<CPUCachingAllocator> caching_allocator =
 |   make_unique<CPUCachingAllocator>();
 | Use caching allocator with a scoped guard at inference time.
 | {
 | WithCPUCachingAllocatorGuard(caching_allocator.get());
 | ... model.forward(...);
 | }
 */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/mobile/CPUCachingAllocator.h]

pub struct CPUCachingAllocator {

    /**
      | Invariants.
      |
      | - 1. If memory is ever allocated via this
      |      allocator then the pointer will exist in
      |      allocation_map_, unless the allocator
      |      returned the memory to OS via free_cached.
      |
      |  - 1.1. Therefore even when the said memory is
      |         "freed" via this allocator (and thus
      |         cached), it will continue to stay in
      |         allocation_map_. Furthermore it will
      |         also exist in available_map_. Thus an
      |         allocated memory pointer can be in both
      |         allocation_map_ and available_map_
      |         simultaneously.
      |
      | - 2. Memory pointer maybe removed from
      |      allocation_map_, when it is freed outside
      |      of the scope of this allocator, but was
      |      allocated by this allocator.
      |
      | - 3. Available map only contains that memory
      |      which was allocated by this allocator and
      |      subsequently freed by this allocator.
      |
      | - As a result of above invariants, allocated
      |   memory ptr cannot be in available_map_ unless
      |   it is in allocation_map_ as well.
      |
      */
    available_map: HashMap<usize,SmallVec<[*mut c_void; 16]>>, // was FlatHashMap
}

pub mod cpu_caching_allocator {

    use super::*;

    lazy_static!{
        /*
        static flat_hash_map<void*, size_t> allocation_map_;
          // Since allocation_map, which is a global instance, is mutated/read via
          // all public APIs we need a global mutex.
          static mutex mutex_;
        */
    }
}

pub trait CPUCachingAllocatorInterface {

    /**
      | Checks the cache to see if allocation of size
      | bytes can be found.
      |
      | If so return cached memory, else allocates
      |   memory, records it for caching and returns.
      |
      */
    fn allocate(&mut self, bytes: usize)  {
        
        todo!();
        /*
        
        */
    }

    /**
      | Checks if the memory being freed is was
      | marked for allocation by an earlier call to
      | allocate.
      |
      | If so cache the allocation.
      |
      | Otherwise free.
      |
      */
    fn free(&mut self, ptr: *mut c_void)  {
        
        todo!();
        /*
        
        */
    }
}

pub struct WithCPUCachingAllocatorGuard {
    prev_caching_allocator_ptr: *mut CPUCachingAllocator, // default = { nullptr }
}

//-------------------------------------------[.cpp/pytorch/c10/mobile/CPUCachingAllocator.cpp]

lazy_static!{
    /*
    thread_local CPUCachingAllocator* caching_allocator_ptr{nullptr};
    */
}

lazy_static!{
    /*
    mutex CPUCachingAllocator::mutex_;
    ska::flat_hash_map<void*, size_t> CPUCachingAllocator::allocation_map_;
    */
}

impl Drop for CPUCachingAllocator {

    fn drop(&mut self) {
        todo!();
        /*
            free_cached();
        */
    }
}

impl CPUCachingAllocator {
    
    /**
      | What it does:
      | 
      | Caches all the allocations carried
      | out by this allocator.
      | 
      | Cache key is the size of the allocation.
      | 
      | If requested size is found in the cache
      | returns the cached pointer.
      | 
      | What it does not do:
      | 
      | No speculative allocation for any future
      | allocations.
      |
      */
    #[inline] pub fn allocate_and_cache(&mut self, bytes: usize)  {
        
        todo!();
        /*
            // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      void* ptr;
      try {
        ptr = alloc_cpu(bytes);
      } catch (Error& e) {
        // If allocation fails, try freeing cached available blocks.
        // For now free all available cached blocks.
        free_cached();
        // Furthermore to consider: If we ever come here running out of memory
        // perhaps it is best to disable caching, since this is likely to happen
        // again.
        // Try again.
        ptr = alloc_cpu(bytes);
      }
      allocation_map_[ptr] = bytes;
      return ptr;
        */
    }
    
    pub fn allocate(&mut self, bytes: usize)  {
        
        todo!();
        /*
            lock_guard<mutex> guard(mutex_);
      const auto& it = available_map_.find(bytes);
      if (it == available_map_.end() || it->second.empty()) {
        return allocate_and_cache(bytes);
      }
      return it->second.pop_back_val();
        */
    }
    
    pub fn free(&mut self, ptr: *mut c_void)  {
        
        todo!();
        /*
            // NB: since we are not really freeing the memory
      // the cases such as quantization code freeing original weights
      // on mobile, will not quite work, as we likely will hold
      // onto that memory.
      // NB: We can also enable max memory cached for better memory
      // management such that free will actually free the memory if
      // we are nearing or above the watermark.
      lock_guard<mutex> guard(mutex_);
      // If this allocation was done before caching allocator was enabled
      // then free regularly
      const auto& it = allocation_map_.find(ptr);
      if (it == allocation_map_.end()) {
        free_cpu(ptr);
        return;
      }
      const size_t alloc_size = it->second;
      available_map_[alloc_size].push_back(ptr);
        */
    }
    
    pub fn record_free(&mut self, ptr: *mut c_void)  {
        
        todo!();
        /*
            // This function captures the case when the allocated memory
      // is being freed outside the scope of this allocator.
      // At the moment only way to capture this is to have the allocator,
      // that uses this CachingAllocator as the backing allocator,
      // call this function explicitly upon freeing memory while
      // outside the scope of caching allocator.
      // If the memory is freed in some other way, then we will likely
      // have undefined behavior or page fault. But this can be
      // the case without caching allocator as well.
      lock_guard<mutex> guard(mutex_);
      const auto& it = allocation_map_.find(ptr);
      if (it != allocation_map_.end()) {
        allocation_map_.erase(it);
      }
        */
    }
    
    pub fn free_cached(&mut self)  {
        
        todo!();
        /*
            for (const auto& it : available_map_) {
        for (const auto ptr : it.second) {
          free_cpu(ptr);
          // When cached memory is return to OS, it must be removed
          // from allocation_map.
          allocation_map_.erase(ptr);
        }
      }
      available_map_.clear();
        */
    }
}

pub fn get_thread_local_caching_allocator() -> *mut CPUCachingAllocator {
    
    todo!();
        /*
            return caching_allocator_ptr;
        */
}

impl WithCPUCachingAllocatorGuard {

    pub fn new(allocator: *mut CPUCachingAllocator) -> Self {
    
        todo!();
        /*


            prev_caching_allocator_ptr_ = GetThreadLocalCachingAllocator();
      caching_allocator_ptr = allocator;
        */
    }
}

impl Drop for WithCPUCachingAllocatorGuard {

    fn drop(&mut self) {
        todo!();
        /*
            caching_allocator_ptr = prev_caching_allocator_ptr_;
        */
    }
}
