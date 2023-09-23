crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/THC/THCCachingHostAllocator.h]
//-------------------------------------------[.cpp/pytorch/aten/src/THC/THCCachingHostAllocator.cpp]

pub struct BlockSize {

    /**
      | allocation size
      |
      */
    size: usize,

    /**
      | host memory pointer
      |
      */
    ptr:  *mut c_void,
}

impl BlockSize {

    pub fn new(
        size: usize,
        ptr:  *mut c_void) -> Self {

        let ptr: *mut c_void = ptr.unwrap_or(NULL);

        todo!();
        /*
        : size(size),
        : ptr(ptr),

        
        */
    }
}

pub struct Block {

    base: BlockSize,

    /**
      | true if the block is currently allocated
      |
      */
    allocated:   bool,

    /**
      | number of outstanding cuda events
      |
      */
    event_count: i32,

    streams:     HashSet<CUDAStream>,
}

impl Block {
    
    pub fn new(
        size:      usize,
        ptr:       *mut c_void,
        allocated: bool) -> Self {
    
        todo!();
        /*
        : block_size(size, ptr),
        : allocated(allocated),
        : event_count(0),
        : streams(),

        
        */
    }
}

pub fn block_comparator(
        a: &BlockSize,
        b: &BlockSize) -> bool {
    
    todo!();
        /*
            // sort by size, break ties with pointer
      if (a.size != b.size) {
        return a.size < b.size;
      }
      return (uintptr_t)a.ptr < (uintptr_t)b.ptr;
        */
}

pub struct HostAllocator {

    /**
      | lock around all operations
      |
      */
    mutex:       Mutex,

    /**
      | blocks by pointer
      |
      */
    blocks:      HashMap<*mut c_void,Block>,

    /**
      | pointers that are ready to be allocated
      | (event_count=0)
      |
      */
    available:   HashSet<BlockSize,Comparison>,

    /**
      | outstanding cuda events
      |
      */
    cuda_events: VecDeque<Pair<cuda::Event,*mut c_void>>,
}

pub mod host_allocator {

    use super::*;

    pub type Comparison = fn(_0: &BlockSize, _1: &BlockSize) -> bool;
}

impl Default for HostAllocator {
    
    fn default() -> Self {
        todo!();
        /*
        : available(BlockComparator),

        
        */
    }
}

impl HostAllocator {

    pub fn malloc(&mut self, 
        ptr:  *mut *mut c_void,
        size: usize) -> CudaError {
        
        todo!();
        /*
            lock_guard<mutex> lock(mutex);

        // process outstanding cuda events which may have occurred
        cudaError_t err = processEvents();
        if (err != cudaSuccess) {
          return err;
        }

        // search for the smallest block which can hold this allocation
        BlockSize search_key(size);
        auto it = available.lower_bound(search_key);
        if (it != available.end()) {
          Block& block = blocks.at(it->ptr);
          THAssert(!block.allocated && block.event_count == 0);
          block.allocated = true;
          *ptr = block.ptr;
          available.erase(it);
          return cudaSuccess;
        }

        // Pinned memory pointers allocated by any device can be directly used by any
        // other device, regardless of the current device at the time of allocation,
        // since we assume unified addressing.
        // So we grab any existing primary context, if available.
        // See pytorch/pytorch#21081.
        OptionalDeviceGuard device_guard;
        auto primary_ctx_device_index = getCUDAHooks().getDevceIndexWithPrimaryContext();
        if (primary_ctx_device_index.has_value()) {
          device_guard.reset_device(Device(DeviceType::Cuda, *primary_ctx_device_index));
        }

        // note that cudaHostAlloc may not touch pointer if size is 0
        *ptr = 0;

        // allocate a new block if no cached allocation is found
        err = cudaHostAlloc(ptr, size, cudaHostAllocDefault);
        if (err != cudaSuccess) {
          return err;
        }

        blocks.insert({*ptr, Block(size, *ptr, true)});
        return cudaSuccess;
        */
    }
    
    pub fn free(&mut self, ptr: *mut c_void) -> CudaError {
        
        todo!();
        /*
            lock_guard<mutex> lock(mutex);

        if (!ptr) {
          return cudaSuccess;
        }

        // process outstanding cuda events which may have occurred
        cudaError_t err = processEvents();
        if (err != cudaSuccess) {
          return err;
        }

        auto it = blocks.find(ptr);
        THAssert(it != blocks.end());

        Block& block = it->second;
        THAssert(block.allocated);

        // free (on valid memory) shouldn't fail, so mark unallocated before
        // we process the streams.
        block.allocated = false;

        // insert CUDA events for each stream on which this block was used. This
        err = insertEvents(block);
        if (err != cudaSuccess) {
          return err;
        }

        if (block.event_count == 0) {
          // the block can be re-used if there are no outstanding cuda events
          available.insert(block);
        }
        return cudaSuccess;
        */
    }
    
    pub fn record_event(&mut self, 
        ptr:    *mut c_void,
        stream: CUDAStream) -> CudaError {
        
        todo!();
        /*
            lock_guard<mutex> lock(mutex);

        auto it = blocks.find(ptr);
        if (it == blocks.end()) {
          // ignore events for untracked pointers
          return cudaSuccess;
        }

        Block& block = it->second;
        THAssert(block.allocated);

        block.streams.insert(stream);
        return cudaSuccess;
        */
    }
    
    pub fn process_events(&mut self) -> CudaError {
        
        todo!();
        /*
            // Process outstanding cudaEvents. Events that are completed are removed
        // from the queue, and the 'event_count' for the corresponding allocation
        // is decremented. Stops at the first event which has not been completed.
        // Since events on different devices or streams may occur out of order,
        // the processing of some events may be delayed.
        while (!cuda_events.empty()) {
          auto& e = cuda_events.front();
          cudaEvent_t event = e.first;

          cudaError_t err = cudaEventQuery(event);
          if (err == cudaErrorNotReady) {
            break;
          } else if (err != cudaSuccess) {
            return err;
          }
          err = cudaEventDestroy(event);
          if (err != cudaSuccess) {
            return err;
          }

          Block& block = blocks.at(e.second);
          block.event_count--;
          if (block.event_count == 0 && !block.allocated) {
            available.insert(block);
          }
          cuda_events.pop_front();
        }
        return cudaSuccess;
        */
    }
    
    pub fn empty_cache(&mut self)  {
        
        todo!();
        /*
            lock_guard<mutex> lock(mutex);

        // remove events for freed blocks
        for (auto it = cuda_events.begin(); it != cuda_events.end(); ++it) {
          cudaEvent_t event = it->first;
          Block& block = blocks.at(it->second);
          if (!block.allocated) {
            THCudaCheckWarn(cudaEventDestroy(event));
            block.event_count--;
          }
        }

        // all cuda_events have been processed
        cuda_events.clear();

        // clear list of available blocks
        available.clear();

        // free and erase non-allocated blocks
        for (auto it = blocks.begin(); it != blocks.end();) {
          Block& block = it->second;
          if (!block.allocated) {
            THCudaCheckWarn(cudaFreeHost(block.ptr));
            it = blocks.erase(it);
          } else {
            ++it;
          }
        }
        */
    }
    
    pub fn insert_events(&mut self, block: &mut Block) -> CudaError {
        
        todo!();
        /*
            cudaError_t err;

        int prev_device;
        err = cudaGetDevice(&prev_device);
        if (err != cudaSuccess) return err;

        unordered_set<CUDAStream> streams(move(block.streams));
        for (auto it = streams.begin(); it != streams.end(); ++it) {
          err = cudaSetDevice(it->device_index());
          if (err != cudaSuccess) break;

          cudaEvent_t event;
          err = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
          if (err != cudaSuccess) break;

          err = cudaEventRecord(event, it->stream());
          if (err != cudaSuccess) break;

          block.event_count++;
          cuda_events.emplace_back(event, block.ptr);
        }

        cudaSetDevice(prev_device);
        return err;
        */
    }
}

lazy_static!{
    /*
    static HostAllocator allocator;
    */
}

/**
  | Records an event in the specified stream. The
  | allocation 'ptr' will not be re-used until the
  | event has occurred.
  |
  */
pub fn thc_caching_host_allocator_record_event(
    ptr:    *mut c_void,
    stream: CUDAStream) -> CudaError {

    todo!();
        /*
            return allocator.recordEvent(ptr, stream);
        */
}

/**
  | Releases cached pinned memory allocations
  | via cudaHostFree
  |
  */
pub fn thc_caching_host_allocator_empty_cache()  {
    
    todo!();
        /*
            allocator.emptyCache();
        */
}

pub fn thc_caching_host_deleter(ptr: *mut c_void)  {
    
    todo!();
        /*
            allocator.free(ptr);
        */
}

pub struct THCCachingHostAllocator {
    base: Allocator,
}

impl THCCachingHostAllocator {
    
    pub fn allocate(&self, size: usize) -> DataPtr {
        
        todo!();
        /*
            THAssert(size >= 0);
        void *ptr;
        THCudaCheck(allocator.malloc(&ptr, size));
        return {ptr, ptr, &THCCachingHostDeleter, DeviceType_CPU};
        */
    }
    
    pub fn raw_deleter(&self) -> DeleterFnPtr {
        
        todo!();
        /*
            return &THCCachingHostDeleter;
        */
    }
}

lazy_static!{
    /*
    static THCCachingHostAllocator thc_caching_host_allocator;
    */
}

/**
  | A caching allocator for CUDA host allocations
  | (pinned memory).
  |
  | This provides a drop-in replacement for
  | THCudaHostAllocator, which re-uses freed pinned
  | (page-locked) memory allocations. This avoids
  | device synchronizations due to cudaFreeHost
  | calls.
  |
  | To ensure correct behavior,
  | THCCachingHostAllocator_recordEvent must be
  | called anytime a pointer from this allocator is
  | used in a cudaMemcpyAsync call between host and
  | device. We implement this for storages and
  | tensors in copy_from_cpu_async_ and
  | copy_to_cpu_async_.
  |
  | Note that this allocator does not split larger
  | allocations into smaller blocks, unlike the
  | caching device allocator.
  |
  */
pub fn get_thc_caching_host_allocator() -> *mut Allocator {
    
    todo!();
        /*
            return &thc_caching_host_allocator;
        */
}
