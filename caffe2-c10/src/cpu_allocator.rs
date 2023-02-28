crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/core/CPUAllocator.h]

/**
  | Use 16-byte alignment on mobile
  | 
  | - ARM NEON AArch32 and AArch64
  | 
  | - x86[-64] < AVX
  |
  */
#[cfg(C10_MOBILE)]
pub const G_ALIGNMENT: usize = 16;

/**
  | Use 64-byte alignment should be enough
  | for computation up to AVX512.
  |
  */
#[cfg(not(C10_MOBILE))]
pub const G_ALIGNMENT: usize = 64;

pub type MemoryDeleter = fn(_0: *mut c_void) -> c_void;

/**
  | A simple struct that is used to report C10's
  | memory allocation and deallocation status to
  | the profiler
  |
  */
pub struct ProfiledCPUMemoryReporter {
    mutex:      RawMutex,
    size_table: HashMap<*mut c_void,usize>,
    allocated:  usize, // default = 0
    log_cnt:    usize, // default = 0
}

//-------------------------------------------[.cpp/pytorch/c10/core/CPUAllocator.cpp]

/**
  | Fill the data memory region of num bytes with
  | a particular garbage pattern.
  |
  | The garbage value is chosen to be NaN if
  | interpreted as floating point value, or a very
  | large integer.
  */
pub fn memset_junk(
        data: *mut c_void,
        num:  usize)  {
    
    todo!();
        /*
            // This garbage pattern is NaN when interpreted as floating point values,
      // or as very large integer values.
      static constexpr int32_t kJunkPattern = 0x7fedbeef;
      static constexpr int64_t kJunkPattern64 =
          static_cast<int64_t>(kJunkPattern) << 32 | kJunkPattern;
      int32_t int64_count = num / sizeof(kJunkPattern64);
      int32_t remaining_bytes = num % sizeof(kJunkPattern64);
      int64_t* data_i64 = reinterpret_cast<int64_t*>(data);
      for (int i = 0; i < int64_count; i++) {
        data_i64[i] = kJunkPattern64;
      }
      if (remaining_bytes > 0) {
        memcpy(data_i64 + int64_count, &kJunkPattern64, remaining_bytes);
      }
        */
}

pub fn alloc_cpu(nbytes: usize)  {
    
    todo!();
        /*
            if (nbytes == 0) {
        return nullptr;
      }
      // We might have clowny upstream code that tries to alloc a negative number
      // of bytes. Let's catch it early.
      CAFFE_ENFORCE(
          ((ptrdiff_t)nbytes) >= 0,
          "alloc_cpu() seems to have been called with negative number: ",
          nbytes);

      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      void* data;
    #ifdef __ANDROID__
      data = memalign(gAlignment, nbytes);
    #elif defined(_MSC_VER)
      data = _aligned_malloc(nbytes, gAlignment);
    #else
      int err = posix_memalign(&data, gAlignment, nbytes);
      if (err != 0) {
        CAFFE_THROW(
            "DefaultCPUAllocator: can't allocate memory: you tried to allocate ",
            nbytes,
            " bytes. Error code ",
            err,
            " (",
            strerror(err),
            ")");
      }
    #endif

      CAFFE_ENFORCE(
          data,
          "DefaultCPUAllocator: not enough memory: you tried to allocate ",
          nbytes,
          " bytes.");

      // move data to a thread's NUMA node
      NUMAMove(data, nbytes, GetCurrentNUMANode());
      CHECK(
          !FLAGS_caffe2_cpu_allocator_do_zero_fill ||
          !FLAGS_caffe2_cpu_allocator_do_junk_fill)
          << "Cannot request both zero-fill and junk-fill at the same time";
      if (FLAGS_caffe2_cpu_allocator_do_zero_fill) {
        memset(data, 0, nbytes);
      } else if (FLAGS_caffe2_cpu_allocator_do_junk_fill) {
        memset_junk(data, nbytes);
      }

      return data;
        */
}

pub fn free_cpu(data: *mut c_void)  {
    
    todo!();
        /*
            #ifdef _MSC_VER
      _aligned_free(data);
    #else
      // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
      free(data);
    #endif
        */
}

pub struct DefaultCPUAllocator { }

unsafe impl std::alloc::Allocator for DefaultCPUAllocator {

    fn allocate(&self, layout: std::alloc::Layout) -> Result<NonNull<[u8]>, std::alloc::AllocError> {

        todo!();
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: std::alloc::Layout) {
        todo!();
    }

    fn allocate_zeroed(&self, layout: std::alloc::Layout) -> Result<NonNull<[u8]>, std::alloc::AllocError> {
        todo!();
    }
}

impl DefaultCPUAllocator {
    
    fn my_allocate(&self, nbytes: usize) -> DataPtr {
        
        todo!();
        /*
            void* data = alloc_cpu(nbytes);
        profiledCPUMemoryReporter().New(data, nbytes);
        return {data, data, &ReportAndDelete, Device(DeviceType::CPU)};
        */
    }
    
    pub fn report_and_delete(ptr: *mut c_void)  {
        
        todo!();
        /*
            if (!ptr) {
          return;
        }
        profiledCPUMemoryReporter().Delete(ptr);
        free_cpu(ptr);
        */
    }
    
    /*
    pub fn raw_deleter(&self) -> DeleterFnPtr {
        
        todo!();
        /*
            return &ReportAndDelete;
        */
    }
    */
}

pub fn profiled_cpu_memory_reporter() -> Arc<ProfiledCPUMemoryReporter> {
    
    todo!();
        /*
            static ProfiledCPUMemoryReporter reporter_;
      return reporter_;
        */
}

/**
  | QNNPACK AND XNNPACK may out-of-bound access the
  | input and / or output tensors. This is
  | by-design, and chosen to make the
  | implementation of micro-kernels both simpler
  | and faster as a result of not having to
  | individually handle the corner cases where the
  | number of processed elements is not a multiple
  | of SIMD register width.
  |
  | This behavior will trigger ASAN though, and may
  | result in a segfault if the accessed memory
  | location just so happens to fall on a page the
  | current process has no read access to.  Here we
  | define a custom allocator that allocates the
  | extra storage required to keep this behavior
  | safe.
  |
  | This allocator could have been restricted to
  | QNNPACK and XNNPACK only, but that would have
  | negative performance ramifications, as input
  | tensors must now be reallocated, and copied
  | over, if the tensor is not allocated with this
  | allocator to begin with.
  |
  | Making this allocator the default on mobile
  | builds minimizes the probability of unnecessary
  | reallocations and copies, and also enables
  | acceleration of operations where the output
  | tensor is allocated outside of the function
  | doing the implementation, wherein the
  | implementation cannot simply re-allocate the
  | output with the guarding allocator.
  |
  | PreGuardBytes: Number of guard bytes to
  | allocate before the allocation.
  |
  | PostGuardBytes: Number of guard bytes to
  | allocate after the allocation.
  */
pub struct DefaultMobileCPUAllocator<const PreGuardBytes: u32,const PostGuardBytes: u32> { }

unsafe impl<const PreGuardBytes: u32,const PostGuardBytes: u32> std::alloc::Allocator 

for DefaultMobileCPUAllocator<PreGuardBytes,PostGuardBytes> {

    fn allocate(&self, layout: std::alloc::Layout) -> Result<NonNull<[u8]>, std::alloc::AllocError> {
        todo!();
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: std::alloc::Layout) {
        todo!();
    }

    fn allocate_zeroed(&self, layout: std::alloc::Layout) -> Result<NonNull<[u8]>, std::alloc::AllocError> {
        todo!();
    }
}

impl<const PreGuardBytes: u32,const PostGuardBytes: u32> DefaultMobileCPUAllocator<PreGuardBytes,PostGuardBytes> {
    
    pub fn deleter(pointer: *mut c_void)  {
        
        todo!();
        /*
            if (C10_UNLIKELY(!pointer)) {
          return;
        }
        // TODO: enable with better TLS support on mobile
        // profiledCPUMemoryReporter().Delete(pointer);
        auto allocator_ptr = GetThreadLocalCachingAllocator();
        auto profiling_allocator_ptr = GetThreadLocalProfilingAllocator();
        if (allocator_ptr != nullptr) {
          allocator_ptr->free(pointer);
        } else if (profiling_allocator_ptr != nullptr) {
          profiling_allocator_ptr->free(pointer);
        } else {
          free_cpu(pointer);
          // This adds extra cost to freeing memory to the default case when
          // caching allocator is not enabled.
          // NOLINTNEXTLINE(clang-analyzer-unix.Malloc)
          CPUCachingAllocator::record_free(pointer);
          auto allocation_planner = GetThreadLocalAllocationPlanner();
          if (allocation_planner != nullptr) {
            allocation_planner->record_free(pointer);
          }
        }
        */
    }
    
    // #[virtual]
    fn my_allocate(&mut self, nbytes: usize) -> DataPtr {
        
        todo!();
        /*
            if (C10_UNLIKELY(0u == nbytes)) {
          return {
              nullptr,
              nullptr,
              &deleter,
              Device(DeviceType::CPU),
          };
        }

        auto alloc_size = PreGuardBytes + nbytes + PostGuardBytes;
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        void* data;
        auto allocator_ptr = GetThreadLocalCachingAllocator();
        auto profiling_allocator_ptr = GetThreadLocalProfilingAllocator();
        if (allocator_ptr != nullptr) {
          data = allocator_ptr->allocate(alloc_size);
        } else if (profiling_allocator_ptr != nullptr) {
          data = profiling_allocator_ptr->allocate(alloc_size);
        } else {
          data = alloc_cpu(alloc_size);
          auto allocation_planner = GetThreadLocalAllocationPlanner();
          if (allocation_planner != nullptr) {
            allocation_planner->record_allocation(alloc_size, data);
          }
        }
        //  profiledCPUMemoryReporter().New(data, alloc_size);
        return {
            reinterpret_cast<uint8_t*>(data) + PreGuardBytes,
            data,
            &deleter,
            Device(DeviceType::CPU),
        };
        */
    }

    /*
    // #[virtual] 
    pub fn raw_deleter(&self) -> DeleterFnPtr {
        
        todo!();
        /*
            return deleter;
        */
    }
    */
}

/**
  | A helper function that is basically
  | doing nothing.
  |
  */
pub fn no_delete(_0: *mut c_void)  {
    
    todo!();
        /*
        
        */
}

/// Get the CPU Allocator.
pub fn get_cpu_allocator() -> *mut Allocator {
    
    todo!();
        /*
            return GetAllocator(DeviceType::CPU);
        */
}

/**
  | Sets the CPU allocator to the given allocator:
  | the caller gives away the ownership of the
  | pointer.
  |
  */
pub fn set_cpu_allocator(alloc: *mut Allocator, priority: Option<u8>) {

    let priority: u8 = priority.unwrap_or(0);
    
    todo!();
        /*
            SetAllocator(DeviceType::CPU, alloc, priority);
        */
}

/**
  | The Mobile CPU allocator must always be present
  | even on non-mobile builds because QNNPACK and
  | XNNPACK are not mobile specific.
  |
  | Pre-guard: 8 bytes for QNNPACK, but set to
  |            gAlignment to ensure SIMD alignment,
  |            not on the allocated memory, but
  |            memory location returned to the
  |            user.
  |
  | Post-guard: 16 bytes for XNNPACK.
  */
lazy_static!{
    /*
    static DefaultMobileCPUAllocator<gAlignment, 16u> g_mobile_cpu_allocator;
    */
}

/// Get the Default Mobile CPU Allocator
pub fn get_default_mobile_cpu_allocator() -> *mut Allocator {
    
    todo!();
        /*
            return &g_mobile_cpu_allocator;
        */
}

/// Get the Default CPU Allocator
#[cfg(C10_MOBILE)]
pub fn get_default_cpu_allocator() -> *mut Allocator {
    
    todo!();
        /*
            return GetDefaultMobileCPUAllocator();
        */
}

/// Get the Default CPU Allocator
#[cfg(not(C10_MOBILE))]
pub fn get_default_cpu_allocator() -> *mut Allocator {
    
    todo!();
        /*
            return &g_cpu_alloc;
        */
}

macro_rules! register_allocator {
    ($t:expr, $f:expr) => {
        /*
        
          namespace {                                     
          static AllocatorRegisterer<t> g_allocator_d(f); 
          }
        */
    }
}

#[cfg(C10_MOBILE)]
register_allocator!(DeviceType::CPU, &g_mobile_cpu_allocator);

/// Global default CPU Allocator
#[cfg(not(C10_MOBILE))]
lazy_static!{
    /*
    static DefaultCPUAllocator g_cpu_alloc;
    */
}

#[cfg(not(C10_MOBILE))]
register_allocator!(DeviceType::CPU, &g_cpu_alloc);

impl ProfiledCPUMemoryReporter {
    
    pub fn new(&mut self, 
        ptr:    *mut c_void,
        nbytes: usize)  {
        
        todo!();
        /*
            if (nbytes == 0) {
        return;
      }
      auto profile_memory = memoryProfilingEnabled();
      size_t allocated = 0;
      if (FLAGS_caffe2_report_cpu_memory_usage || profile_memory) {
        lock_guard<mutex> guard(mutex_);
        size_table_[ptr] = nbytes;
        allocated_ += nbytes;
        allocated = allocated_;
      }
      if (FLAGS_caffe2_report_cpu_memory_usage) {
        LOG(INFO) << "C10 alloc " << nbytes << " bytes, total alloc " << allocated
                  << " bytes.";
      }
      if (profile_memory) {
        reportMemoryUsageToProfiler(ptr, nbytes, Device(DeviceType::CPU));
      }
        */
    }
    
    pub fn delete(&mut self, ptr: *mut c_void)  {
        
        todo!();
        /*
            size_t nbytes = 0;
      auto profile_memory = memoryProfilingEnabled();
      size_t allocated = 0;
      if (FLAGS_caffe2_report_cpu_memory_usage || profile_memory) {
        lock_guard<mutex> guard(mutex_);
        auto it = size_table_.find(ptr);
        if (it != size_table_.end()) {
          allocated_ -= it->second;
          allocated = allocated_;
          nbytes = it->second;
          size_table_.erase(it);
        } else {
          // C10_LOG_EVERY_MS might log every time in some builds,
          // using a simple counter to avoid spammy logs
          if (log_cnt_++ % 1000 == 0) {
            LOG(WARNING) << "Memory block of unknown size was allocated before "
                         << "the profiling started, profiler results will not "
                         << "include the deallocation event";
          }
        }
      }
      if (nbytes == 0) {
        return;
      }
      if (FLAGS_caffe2_report_cpu_memory_usage) {
        LOG(INFO) << "C10 deleted " << nbytes << " bytes, total alloc " << allocated
                  << " bytes.";
      }
      if (profile_memory) {
        reportMemoryUsageToProfiler(
            ptr, -nbytes, Device(DeviceType::CPU));
      }
        */
    }
}

lazy_static!{
    /*
    Allocator* cpu_caching_alloc = nullptr;
    uint8_t cpu_caching_alloc_priority = 0;
    */
}

/**
  | The CPUCachingAllocator is experimental and
  | might disappear in the future.
  |
  | The only place that uses it is in
  | StaticRuntime.
  |
  | Set the CPU Caching Allocator
  |
  */
pub fn set_cpu_caching_allocator(
    alloc:    *mut Allocator,
    priority: Option<u8>)  {

    let priority: u8 = priority.unwrap_or(0);
    
    todo!();
        /*
            if (priority >= cpu_caching_alloc_priority) {
        cpu_caching_alloc = alloc;
        cpu_caching_alloc_priority = priority;
      }
        */
}

/// Get the CPU Caching Allocator
pub fn get_cpu_caching_allocator() -> *mut Allocator {
    
    todo!();
        /*
            if (cpu_caching_alloc == nullptr) {
        VLOG(1)
            << "There is not caching allocator registered for CPU, use the default allocator instead.";
        return GetAllocator(DeviceType::CPU);
      }
      return cpu_caching_alloc;
        */
}
