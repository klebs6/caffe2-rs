crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/TH/THAllocator.h]

pub const TH_ALLOCATOR_MAPPED_SHARED:    usize = 1;
pub const TH_ALLOCATOR_MAPPED_SHAREDMEM: usize = 2;
pub const TH_ALLOCATOR_MAPPED_EXCLUSIVE: usize = 4;
pub const TH_ALLOCATOR_MAPPED_NOCREATE:  usize = 8;
pub const TH_ALLOCATOR_MAPPED_KEEPFD:    usize = 16;
pub const TH_ALLOCATOR_MAPPED_FROMFD:    usize = 32;
pub const TH_ALLOCATOR_MAPPED_UNLINK:    usize = 64;

/**
  | Sentinel value/type to help distinguish the
  | file descriptor constructor from the non-file
  | descriptor constructor
  |
  */
pub enum WithFd { WITH_FD }

pub struct THMapAllocator {

    closed:   bool, // default = false
    filename: String,
    flags:    i32, // default = 0

    /**
      | mapped size
      |
      */
    size:     libc::ptrdiff_t,
    fd:       i32, // default = -1
    base_ptr: *mut c_void, // default = nullptr
}

impl THMapAllocator {
    
    pub fn filename(&self) -> *const u8 {
        
        todo!();
        /*
            return filename_.c_str();
        */
    }
    
    pub fn fd(&self) -> i32 {
        
        todo!();
        /*
            #ifdef _WIN32
        AT_ERROR("THMapAllocator::fd() is unsupported on Windows");
    #else
        return fd_;
    #endif
        */
    }
    
    pub fn size(&self) -> libc::ptrdiff_t {
        
        todo!();
        /*
            return size_;
        */
    }

    /**
      | Return a pointer to the actual data for this
      | allocator (in the case of the refcounted
      | allocator, this is offset from the base
      | pointer.)
      */
    pub fn data(&self)  {
        
        todo!();
        /*
            return base_ptr_;
        */
    }

    /**
      | Closes the data. Helps us avoid destructor
      | shenanigans
      |
      */
    pub fn close(&mut self)  {
        
        todo!();
        /*
        
        */
    }
}

/// Base-from-member idiom
pub struct THRefcountedMapAllocatorArgCheck {

}

pub struct THRefcountedMapAllocator {
    base: THRefcountedMapAllocatorArgCheck,
    base2: THMapAllocator,
}

impl Drop for THRefcountedMapAllocator {

    fn drop(&mut self) {
        todo!();
        /*
            close();
        */
    }
}

//-------------------------------------------[.cpp/pytorch/aten/src/TH/THAllocator.cpp]

#[cfg(ATOMIC_INT_LOCK_FREE_EQ_2)]
pub const TH_ATOMIC_IPC_REFCOUNT: usize = 1;

/* ---------- end of stuff for mapped files  ---------- */

/**
  | default malloc/free allocator. malloc
  | and realloc raise an error (using
  | THError) on allocation failure.
  |
  */
pub fn get_thd_efault_allocator() -> *mut Allocator {
    
    todo!();
        /*
            return GetCPUAllocator();
        */
}

pub const TH_ALLOC_ALIGNMENT: usize = 64;

#[cfg(any(_WIN32,HAVE_MMAP))]
pub struct THMapInfo {
    refcount: Atomic<i32>,
}

#[cfg(any(_WIN32,HAVE_MMAP))]
pub const UNKNOWN_FILENAME: String = "filename not specified";

#[cfg(any(_WIN32,HAVE_MMAP))]
#[cfg(_WIN32)]
pub const UNKNOWN_EVENTNAME: String = "eventname not specified";

#[cfg(any(_WIN32,HAVE_MMAP))]
impl THMapAllocator {
    
    pub fn new(
        _0:       WithFd,
        filename: String,
        fd:       i32,
        flags:    i32,
        size:     usize) -> Self {
    
        todo!();
        /*


            : filename_(filename.empty() ? unknown_filename : move(filename))
      , flags_(0) // to be filled later
      , size_(0) // to be filled later
    #ifdef _WIN32
      , handle_(INVALID_HANDLE_VALUE) // to be filled later
      , event_(INVALID_HANDLE_VALUE) // to be filled later
      , eventname_(filename.empty() ? unknown_eventname : (filename + "_event"))
    #else
      , fd_(fd)
    #endif
      , base_ptr_(nullptr)

      if (!(flags & TH_ALLOCATOR_MAPPED_SHARED) && !(flags & TH_ALLOCATOR_MAPPED_SHAREDMEM)) {
        flags &= ~TH_ALLOCATOR_MAPPED_NOCREATE;
      }
      if ((flags ^ TH_ALLOCATOR_MAPPED_EXCLUSIVE) == 0) {
        AT_ERROR("TH_ALLOCATOR_MAPPED_EXCLUSIVE flag requires opening the file in shared mode");
      }
    #ifdef _WIN32
      if (fd != -1) {
        AT_ERROR("THMapAllocator_newWithFd is unsupported on Windows");
      }
    #endif
      flags_ = flags;

      // OK, now do the allocation

      if (size == 0) {
        return;
      }

    #ifdef _WIN32
      if (flags_ & TH_ALLOCATOR_MAPPED_SHAREDMEM) {
        // Shadowing
        const wchar_t *filename;
        const wchar_t *eventname;
        const wstring wFilename = u8u16(filename_);
        const wstring wEventname = u8u16(eventname_);
        LARGE_INTEGER hfilesz;

        if (filename_[0] == '/') {
          filename = wFilename.c_str() + 1;
          eventname = wEventname.c_str() + 1;
        } else {
          filename = wFilename.c_str();
          eventname = wEventname.c_str();
        }

        hfilesz.QuadPart = size;

        if (flags_ & TH_ALLOCATOR_MAPPED_EXCLUSIVE) {
          event_ = CreateEventW(nullptr, FALSE, FALSE, eventname);
        } else if (flags_ & TH_ALLOCATOR_MAPPED_NOCREATE) {
          event_ = OpenEventW(EVENT_ALL_ACCESS, FALSE, eventname);
        } else {
          AT_ERROR("Expected either TH_ALLOCATOR_MAPPED_EXCLUSIVE or TH_ALLOCATOR_MAPPED_NOCREATE");
        }

        if (event_ == nullptr) {
          AT_ERROR("Couldn't open shared event: <", eventname, ">, error code: <", GetLastError(), ">");
        }

        if (flags_ & TH_ALLOCATOR_MAPPED_EXCLUSIVE) {
          handle_ = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, hfilesz.HighPart, hfilesz.LowPart, filename);
        } else if (flags_ & TH_ALLOCATOR_MAPPED_NOCREATE) {
          handle_ = OpenFileMappingW(FILE_MAP_ALL_ACCESS, FALSE, filename);
        } else {
          AT_ERROR("Expected either TH_ALLOCATOR_MAPPED_EXCLUSIVE or TH_ALLOCATOR_MAPPED_NOCREATE");
        }

        if (handle_ == nullptr) {
          AT_ERROR("Couldn't open shared file mapping: <", filename, ">, error code: <", GetLastError(), ">");
        }

        size_ = size;
        base_ptr_ = MapViewOfFile(handle_, FILE_MAP_ALL_ACCESS, 0, 0, size);
        if (!base_ptr_) {
          AT_ERROR("Couldn't map view of shared file <", filename, ">, error code: <", GetLastError(), ">");
        }
      } else {

        HANDLE hfile;
        HANDLE hmfile;
        LARGE_INTEGER hfilesz;

        if (flags_ & TH_ALLOCATOR_MAPPED_EXCLUSIVE) {
          AT_ERROR("exclusive file mapping is not supported on Windows");
        }
        if (flags_ & TH_ALLOCATOR_MAPPED_NOCREATE) {
          AT_ERROR("file mapping without creation is not supported on Windows");
        }
        if (flags_ & TH_ALLOCATOR_MAPPED_KEEPFD) {
          AT_ERROR("TH_ALLOCATOR_MAPPED_KEEPFD not supported on Windows");
        }
        if (flags_ & TH_ALLOCATOR_MAPPED_FROMFD) {
          AT_ERROR("TH_ALLOCATOR_MAPPED_FROMFD not supported on Windows");
        }

        // Shadowing
        const wchar_t *filename;
        const wstring wFilename = u8u16(filename_);

        filename = wFilename.c_str();

        /* open file */
        /* FILE_FLAG_RANDOM_ACCESS ? */
        if (flags_) {
          hfile = CreateFileW(filename, GENERIC_READ|GENERIC_WRITE, FILE_SHARE_WRITE|FILE_SHARE_READ, 0, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, 0);
          if (hfile == INVALID_HANDLE_VALUE) {
            AT_ERROR("could not open file <", filename_, "> in read-write mode; error code: <", GetLastError(), ">");
          }
        } else {
          hfile = CreateFileW(filename, GENERIC_READ, FILE_SHARE_WRITE|FILE_SHARE_READ, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
          if (hfile == INVALID_HANDLE_VALUE) {
            AT_ERROR("could not open file <", filename_, "> in read-only mode; error code: <", GetLastError(), ">");
          }
        }

        if (GetFileSizeEx(hfile, &hfilesz) == 0) {
          AT_ERROR("could not get file size: <", filename_, ">; error code: <", GetLastError(), ">");
        }

        if (size > 0) {
          if (size > hfilesz.QuadPart) {
            if (flags_) {
              hfilesz.QuadPart = size;
              if (SetFilePointerEx(hfile, hfilesz, NULL, FILE_BEGIN) == 0) {
                CloseHandle(hfile);
                AT_ERROR("unable to stretch file <", filename_, "> to the right size; error code: <", GetLastError(), ">", filename_);
              }
              if (SetEndOfFile(hfile) == 0) {
                CloseHandle(hfile);
                AT_ERROR("unable to write to file <", filename_, ">; error code: <", GetLastError(), ">");
              }
            } else {
              CloseHandle(hfile);
              AT_ERROR("file <", filename_, "> size is smaller than the required mapping size <", size, ">; error code: <", GetLastError(), ">");
            }
          }
        } else {
          size = hfilesz.QuadPart;
        }

        size_ = size; /* if we are here, it must be the right size */

        hfilesz.QuadPart = size_;

        /* get map handle */
        if (flags_) {
          if ( (hmfile = CreateFileMappingW(hfile, NULL, PAGE_READWRITE, hfilesz.HighPart, hfilesz.LowPart, NULL)) == NULL ) {
            AT_ERROR("could not create a map on file <", filename_, ">; error code: <", GetLastError(), ">");
          }
        } else {
          if ( (hmfile = CreateFileMappingW(hfile, NULL, PAGE_WRITECOPY, hfilesz.HighPart, hfilesz.LowPart, NULL)) == NULL ) {
            AT_ERROR("could not create a map on file <", filename_, ">; error code: <", GetLastError(), ">");
          }
        }

        /* map the stuff */
        if(flags_) {
          base_ptr_ = MapViewOfFile(hmfile, FILE_MAP_ALL_ACCESS, 0, 0, 0);
        } else {
          base_ptr_ = MapViewOfFile(hmfile, FILE_MAP_COPY, 0, 0, 0);
        }

        CloseHandle(hfile);
        CloseHandle(hmfile);
      }
    #else /* _WIN32 */
      {
        /* open file */
        int fd;
        int flags; // shadow
        struct stat file_stat;

        if (flags_ & (TH_ALLOCATOR_MAPPED_SHARED | TH_ALLOCATOR_MAPPED_SHAREDMEM)) {
          flags = O_RDWR | O_CREAT;
        } else {
          flags = O_RDONLY;
        }

        if (flags_ & TH_ALLOCATOR_MAPPED_EXCLUSIVE) {
          flags |= O_EXCL;
        }
        if (flags_ & TH_ALLOCATOR_MAPPED_NOCREATE) {
          flags &= ~O_CREAT;
        }

        if (!(flags_ & TH_ALLOCATOR_MAPPED_FROMFD)) {
          if (flags_ & TH_ALLOCATOR_MAPPED_SHARED) {
            if ((fd = open(filename_.c_str(), flags, (mode_t)0600)) == -1) {
              AT_ERROR("unable to open file <", filename_, "> in read-write mode");
            }
          } else if (flags_ & TH_ALLOCATOR_MAPPED_SHAREDMEM) {
    #ifdef HAVE_SHM_OPEN
            if((fd = shm_open(filename_.c_str(), flags, (mode_t)0600)) == -1) {
              AT_ERROR("unable to open shared memory object <", filename_, "> in read-write mode");
            }
    #else
            AT_ERROR("unable to open file <", filename_, "> in sharedmem mode, shm_open unavailable on this platform");
    #endif
          } else {
            if ((fd = open(filename_.c_str(), O_RDONLY)) == -1) {
              AT_ERROR("unable to open file <", filename_, "> in read-only mode");
            }
          }
        } else {
          fd = fd_;
        }

        if (fstat(fd, &file_stat) == -1) {
          if (!(flags_ & TH_ALLOCATOR_MAPPED_FROMFD)) {
            ::close(fd);
          }
          AT_ERROR("unable to stat the file <", filename_, ">");
        }

        if (size > 0) {
          if (size > file_stat.st_size) {
            if (flags_) {
              if (ftruncate(fd, size) == -1) {
                AT_ERROR("unable to resize file <", filename_, "> to the right size");
              }
              if (fstat(fd, &file_stat) == -1 || file_stat.st_size < size) {
                ::close(fd);
                AT_ERROR("unable to stretch file <", filename_, "> to the right size");
              }
    /* on macOS write returns with errno 45 (Opperation not supported) when used
     * with a file descriptor obtained via shm_open
     */
    #ifndef __APPLE__
              if ((write(fd, "", 1)) != 1) /* note that the string "" contains the '\0' byte ... */ {
                ::close(fd);
                AT_ERROR("unable to write to file <", filename_, ">");
              }
    #endif
            } else {
              ::close(fd);
              AT_ERROR("file <", filename_, "> size is smaller than the required mapping size <", size, ">");
            }
          }
        } else {
          size = file_stat.st_size;
        }

        size_ = size; /* if we are here, it must be the right size */

        /* map it */
        if (flags_ & (TH_ALLOCATOR_MAPPED_SHARED | TH_ALLOCATOR_MAPPED_SHAREDMEM)) {
          base_ptr_ = mmap(nullptr, size_, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
        } else {
          base_ptr_ = mmap(nullptr, size_, PROT_READ|PROT_WRITE, MAP_PRIVATE, fd, 0);
        }

        if (base_ptr_ == MAP_FAILED) {
          base_ptr_ = nullptr; /* let's be sure it is NULL */
          AT_ERROR("unable to mmap ", size_, " bytes from file <", filename_, ">: ", strerror(errno), " (", errno, ")");
        }

        if (flags_ & TH_ALLOCATOR_MAPPED_KEEPFD) {
          fd_ = fd;
        } else {
          if (::close(fd) == -1) {
            AT_ERROR("Error closing file <", filename_, ">");
          }
          fd_ = -1;
        }

        if (flags_ & TH_ALLOCATOR_MAPPED_UNLINK) {
          if (flags_ & TH_ALLOCATOR_MAPPED_SHAREDMEM) {
    #ifdef HAVE_SHM_UNLINK
            if (shm_unlink(filename_.c_str()) == -1) {
              AT_ERROR("could not unlink the shared memory file ", filename_);
            }
    #else
            AT_ERROR("could not unlink the shared memory file ", filename_, ", shm_unlink not available on platform");
    #endif
          } else {
            if (unlink(filename_.c_str()) == -1)
              AT_ERROR("could not unlink file %s", filename_);
          }
        }

        if (base_ptr_ == MAP_FAILED) {
          AT_ERROR("$ Torch: unable to mmap memory: you tried to mmap ", size_/1073741824, " GB.");
        }
      }
    #endif
      reportMemoryUsageToProfiler(base_ptr_, size_, Device(DeviceType_CPU));
        */
    }
    
    pub fn new(
        filename: String,
        flags:    i32,
        size:     usize) -> Self {
    
        todo!();
        /*
        : THMapAllocator(WITH_FD, move(filename), -1, flags, size)
        */
    }
}

#[cfg(any(_WIN32,HAVE_MMAP))]
#[cfg(_WIN32)]
pub struct ReleaseContext {
    event:  HANDLE,
    handle: HANDLE,
    wait:   HANDLE,
}

#[cfg(any(_WIN32,HAVE_MMAP))]
#[cfg(_WIN32)]
pub fn wait_for_release_handle(
        lp_param:            PVOID,
        timer_or_wait_fired: bool)  {
    
    todo!();
        /*
            if (lpParam) {
        ReleaseContext *ctx = (ReleaseContext *)lpParam;

        SetEvent(ctx->event);
        CloseHandle(ctx->event);
        CloseHandle(ctx->handle);

        UnregisterWait(ctx->wait);

        THFree(ctx);
      }
        */
}

#[cfg(any(_WIN32,HAVE_MMAP))]
impl THMapAllocator {
    
    pub fn close(&mut self)  {
        
        todo!();
        /*
            if (closed_) {
        return;
      }
      closed_ = true;
      if (base_ptr_ == nullptr) {
        return;
      }
    #ifdef _WIN32
      if ((flags_ & TH_ALLOCATOR_MAPPED_KEEPFD) || (flags_ & TH_ALLOCATOR_MAPPED_SHAREDMEM))
        CloseHandle(handle_);
      if(UnmapViewOfFile(base_ptr_) == 0)
        AT_ERROR("could not unmap the shared memory file");
    #else /* _WIN32 */
      if (flags_ & TH_ALLOCATOR_MAPPED_KEEPFD) {
        if (::close(fd_) == -1) {
          AT_ERROR("could not close file descriptor ", fd_);
        }
      }

      if (munmap(base_ptr_, size_)) {
        AT_ERROR("could not unmap the shared memory file");
      }

      if (!(flags_ & (TH_ALLOCATOR_MAPPED_FROMFD | TH_ALLOCATOR_MAPPED_UNLINK))) {
        if (flags_ & TH_ALLOCATOR_MAPPED_SHAREDMEM) {
    #ifdef HAVE_SHM_UNLINK
          if (shm_unlink(filename_.c_str()) == -1) {
            AT_ERROR("could not unlink the shared memory file ", filename_);
          }
    #else
          AT_ERROR("could not unlink the shared memory file ", filename_, ", shm_unlink not available on platform");
    #endif
        }
      }
    #endif /* _WIN32 */
        */
    }
}

#[cfg(not(any(_WIN32,HAVE_MMAP)))]
impl THMapAllocator {
    
    pub fn new(
        filename: String,
        flags:    i32,
        size:     usize) -> Self {
    
        todo!();
        /*
            AT_ERROR("file mapping not supported on your system");
        */
    }
    
    pub fn new(
        _0:       WithFd,
        filename: String,
        fd:       i32,
        flags:    i32,
        size:     usize) -> Self {
    
        todo!();
        /*
            AT_ERROR("file mapping not supported on your system");
        */
    }
}

#[cfg(all(any(_WIN32,HAVE_MMAP),TH_ATOMIC_IPC_REFCOUNT))]
impl THRefcountedMapAllocatorArgCheck {
    
    pub fn new(flags: i32) -> Self {
    
        todo!();
        /*


            if (flags & TH_ALLOCATOR_MAPPED_FROMFD) {
        AT_ERROR("THRefcountedMapAllocator doesn't support TH_ALLOCATOR_MAPPED_FROMFD flag");
      }
      if (flags & TH_ALLOCATOR_MAPPED_KEEPFD) {
        AT_ERROR("THRefcountedMapAllocator doesn't support TH_ALLOCATOR_MAPPED_KEEPFD flag");
      }
      if (flags & TH_ALLOCATOR_MAPPED_UNLINK) {
        AT_ERROR("THRefcountedMapAllocator doesn't support TH_ALLOCATOR_MAPPED_UNLINK flag");
      }
      if (!(flags & TH_ALLOCATOR_MAPPED_SHAREDMEM)) {
        AT_ERROR("THRefcountedMapAllocator requires TH_ALLOCATOR_MAPPED_SHAREDMEM flag");
      }
        */
    }
}

#[cfg(all(any(_WIN32,HAVE_MMAP),TH_ATOMIC_IPC_REFCOUNT))]
impl THRefcountedMapAllocator {
    
    pub fn new(
        filename: *const u8,
        flags:    i32,
        size:     usize) -> Self {
    
        todo!();
        /*
        : THRefcountedMapAllocatorArgCheck(flags)
      , THMapAllocator(filename, flags, size + TH_ALLOC_ALIGNMENT) 

        initializeAlloc();
        */
    }
    
    pub fn new(
        _0:       WithFd,
        filename: *const u8,
        fd:       i32,
        flags:    i32,
        size:     usize) -> Self {
    
        todo!();
        /*


            : THRefcountedMapAllocatorArgCheck(flags)
      , THMapAllocator(WITH_FD, filename, flags, fd, size + TH_ALLOC_ALIGNMENT) 

        initializeAlloc();
        */
    }
    
    pub fn initialize_alloc(&mut self)  {
        
        todo!();
        /*
            THMapInfo *map_info = (THMapInfo*)base_ptr_;

    #ifdef _WIN32
      ReleaseContext* r_ctx = (ReleaseContext *) THAlloc(sizeof(ReleaseContext));
      r_ctx->handle = handle_;
      r_ctx->event = event_;
      r_ctx->wait = NULL;
      BOOL can_wait = RegisterWaitForSingleObject(&r_ctx->wait, event_, WaitForReleaseHandle, (PVOID)r_ctx, INFINITE, WT_EXECUTEONLYONCE);
      if (!can_wait) {
        AT_ERROR("Couldn't register wait on event, error code: <", GetLastError(), ">");
      }
    #endif

      if (flags_ & TH_ALLOCATOR_MAPPED_EXCLUSIVE) {
        new (&map_info->refcount) atomic<int>(1);
      } else {
        map_info->refcount++;
      }
        */
    }
    
    pub fn close(&mut self)  {
        
        todo!();
        /*
            if (closed_) {
        return;
      }
      closed_ = true;

      void* data = base_ptr_;

    #ifdef _WIN32
      THMapInfo *info = (THMapInfo*)data;
      if (--info->refcount == 0) {
        SetEvent(event_);
      }
      if(UnmapViewOfFile(data) == 0) {
        AT_ERROR("could not unmap the shared memory file");
      }
    #else /* _WIN32 */

      THMapInfo *info = (THMapInfo*)(data);
      if (--info->refcount == 0) {
    #ifdef HAVE_SHM_UNLINK
        if (shm_unlink(filename_.c_str()) == -1) {
          AT_ERROR("could not unlink the shared memory file ", filename_);
        }
    #else
        AT_ERROR("could not unlink the shared memory file ", filename_, ", shm_unlink not available on platform");
    #endif /* HAVE_SHM_UNLINK */
      }
      if (munmap(info, size_)) {
        AT_ERROR("could not unmap the shared memory file ", filename_);
      }
    #endif /* _WIN32 */
        */
    }
    
    pub fn incref(&mut self)  {
        
        todo!();
        /*
            THMapInfo *map_info = static_cast<THMapInfo*>(base_ptr_);
      ++map_info->refcount;
        */
    }
    
    pub fn decref(&mut self) -> i32 {
        
        todo!();
        /*
            THMapInfo *map_info = static_cast<THMapInfo*>(base_ptr_);
      return --map_info->refcount == 0;
        */
    }
}

#[cfg(not(all(any(_WIN32,HAVE_MMAP),TH_ATOMIC_IPC_REFCOUNT)))]
impl THRefcountedMapAllocator {
    
    pub fn new(
        filename: *const u8,
        flags:    i32,
        size:     usize) -> Self {
    
        todo!();
        /*


            : THRefcountedMapAllocatorArgCheck(flags),
        THMapAllocator(filename, flags, size + TH_ALLOC_ALIGNMENT)

      AT_ERROR("refcounted file mapping not supported on your system");
        */
    }
    
    pub fn new(
        _0:       WithFd,
        filename: *const u8,
        fd:       i32,
        flags:    i32,
        size:     usize) -> Self {
    
        todo!();
        /*


            : THRefcountedMapAllocatorArgCheck(flags),
        THMapAllocator(WITH_FD, filename, flags, fd, size + TH_ALLOC_ALIGNMENT)

      AT_ERROR("refcounted file mapping not supported on your system");
        */
    }
}

pub fn delete_thm_ap_allocator(ptr: *mut c_void)  {
    
    todo!();
        /*
            delete static_cast<THMapAllocator*>(ptr);
        */
}

pub fn delete_thr_efcounted_map_allocator(ptr: *mut c_void)  {
    
    todo!();
        /*
            delete static_cast<THRefcountedMapAllocator*>(ptr);
        */
}

impl THMapAllocator {
    
    pub fn from_data_ptr(&mut self, dptr: &DataPtr) -> *mut THMapAllocator {
        
        todo!();
        /*
            return dptr.cast_context<THMapAllocator>(&deleteTHMapAllocator);
        */
    }
}

impl THRefcountedMapAllocator {
    
    pub fn from_data_ptr(&mut self, dptr: &DataPtr) -> *mut THRefcountedMapAllocator {
        
        todo!();
        /*
            return dptr.cast_context<THRefcountedMapAllocator>(&deleteTHRefcountedMapAllocator);
        */
    }
}

impl THMapAllocator {
    
    pub fn make_data_ptr(&mut self, 
        filename:        String,
        flags:           i32,
        size:            usize,
        actual_size_out: *mut usize) -> DataPtr {
        
        todo!();
        /*
            auto* context = new THMapAllocator(move(filename), flags, size);
      if (actual_size_out) *actual_size_out = context->size();
      return {context->data(), context, &deleteTHMapAllocator, DeviceType_CPU};
        */
    }
    
    pub fn make_data_ptr(&mut self, 
        _0:              WithFd,
        filename:        *const u8,
        fd:              i32,
        flags:           i32,
        size:            usize,
        actual_size_out: *mut usize) -> DataPtr {
        
        todo!();
        /*
            auto* context = new THMapAllocator(WITH_FD, filename, fd, flags, size);
      if (actual_size_out) *actual_size_out = context->size();
      return {context->data(), context, &deleteTHMapAllocator, DeviceType_CPU};
        */
    }
}

impl THRefcountedMapAllocator {
    
    pub fn make_data_ptr(&mut self, 
        filename:        *const u8,
        flags:           i32,
        size:            usize,
        actual_size_out: *mut usize) -> DataPtr {
        
        todo!();
        /*
            auto* context = new THRefcountedMapAllocator(filename, flags, size);
      if (actual_size_out) *actual_size_out = context->size() - TH_ALLOC_ALIGNMENT;
      return {context->data(), context, &deleteTHRefcountedMapAllocator, DeviceType_CPU};
        */
    }
    
    pub fn make_data_ptr(&mut self, 
        _0:              WithFd,
        filename:        *const u8,
        fd:              i32,
        flags:           i32,
        size:            usize,
        actual_size_out: *mut usize) -> DataPtr {
        
        todo!();
        /*
            auto* context = new THRefcountedMapAllocator(WITH_FD, filename, fd, flags, size);
      if (actual_size_out) *actual_size_out = context->size() - TH_ALLOC_ALIGNMENT;
      return {context->data(), context, &deleteTHRefcountedMapAllocator, DeviceType_CPU};
        */
    }
    
    pub fn data(&self)  {
        
        todo!();
        /*
            return static_cast<void*>(static_cast<char*>(base_ptr_) + TH_ALLOC_ALIGNMENT);
        */
    }
}

impl Drop for THMapAllocator {

    fn drop(&mut self) {
        todo!();
        /*
      close();
      reportMemoryUsageToProfiler(base_ptr_, -size_, Device(DeviceType_CPU));
        */
    }
}
