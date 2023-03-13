crate::ix!();


/**
  | A work-stealing threadpool loosely
  | based off of pthreadpool
  |
  */
pub const kCacheLineSize: usize = 64;

/**
  | A threadpool with the given number of
  | threads.
  | 
  | -----------
  | @note
  | 
  | the kCacheLineSize alignment is present
  | only for cache performance, and is not
  | strictly enforced (for example, when
  | the object is created on the heap). Thus,
  | in order to avoid misaligned intrinsics,
  | no SSE instructions shall be involved
  | in the ThreadPool implementation.
  | ----------
  | @note
  | 
  | alignas is disabled because some compilers
  | do not deal with TORCH_API and alignas
  | annotations at the same time.
  |
  */
#[repr(align(64))] //kCacheLineSize
pub struct ThreadPool {

    execution_mutex:  parking_lot::RawMutex,
    min_work_size:    usize,
    num_threads:      Atomic<usize>,
    workers_pool:     Arc<WorkersPool>,
    tasks:            Vec<Arc<dyn Task>>,
}

unsafe impl Send for ThreadPool {}
unsafe impl Sync for ThreadPool {}

lazy_static!{
    static ref default_thread_pool: Box<ThreadPool> = Box::new(ThreadPool::default());
}

static default_num_threads: AtomicUsize = AtomicUsize::new(0);

#[inline] pub fn get_default_num_threads() -> usize {
    
    todo!();
    /*
        CAFFE_ENFORCE(cpuinfo_initialize(), "cpuinfo initialization failed");
      int numThreads = cpuinfo_get_processors_count();

      bool applyCap = false;
    #if defined(C10_ANDROID)
      applyCap = FLAGS_caffe2_threadpool_android_cap;
    #elif defined(C10_IOS)
      applyCap = FLAGS_caffe2_threadpool_ios_cap;
    #endif

      if (applyCap) {
        switch (numThreads) {
    #if defined(C10_ANDROID) && (CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64)
          case 4:
            switch (cpuinfo_get_core(0)->midr & UINT32_C(0xFF00FFF0)) {
              case UINT32_C(0x51002110): /* Snapdragon 820 Kryo Silver */
              case UINT32_C(0x51002010): /* Snapdragon 821 Kryo Silver */
              case UINT32_C(0x51002050): /* Snapdragon 820/821 Kryo Gold */
                /* Kryo: 2+2 big.LITTLE */
                numThreads = 2;
                break;
              default:
                /* Anything else: assume homogeneous architecture */
                numThreads = 4;
                break;
            }
            break;
    #endif
          case 5:
            /* 4+1 big.LITTLE */
            numThreads = 4;
            break;
          case 6:
            /* 2+4 big.LITTLE */
            numThreads = 2;
            break;
          case 8:
            /* 4+4 big.LITTLE */
            numThreads = 4;
            break;
          case 10:
            /* 4+4+2 Min.Med.Max, running on Med cores */
            numThreads = 4;
            break;
          default:
            if (numThreads > 4) {
              numThreads = numThreads / 2;
            }
            break;
        }
      }

      if (FLAGS_pthreadpool_size) {
        // Always give precedence to explicit setting.
        numThreads = FLAGS_pthreadpool_size;
      }
      return numThreads;
    */
}

/**
  | Default smallest amount of work that
  | will be partitioned between multiple
  | threads; the runtime value is configurable
  |
  */
pub const kDefaultMinWorkSize: usize = 1;

impl Default for ThreadPool {
    fn default() -> Self {
        //low default 4 threads per pool
        Self::new(4)
    }
}

impl ThreadPool {
    
    #[inline] pub fn get_min_work_size(&self) -> usize {
        
        todo!();
        /*
            return minWorkSize_;
        */
    }
    
    #[inline] pub fn default_thread_pool(&mut self) -> Box<ThreadPool> {
        
        todo!();
        /*
            defaultNumThreads_ = getDefaultNumThreads();
      LOG(INFO) << "Constructing thread pool with " << defaultNumThreads_
                << " threads";
      return std::make_unique<ThreadPool>(defaultNumThreads_);
        */
    }
    
    pub fn new(num_threads: i32) -> Self {
        todo!();
        /*
            : minWorkSize_(kDefaultMinWorkSize),
          numThreads_(numThreads),
          workersPool_(std::make_shared<WorkersPool>())
        */
    }
    
    /**
      | Returns the number of threads currently
      | in use
      |
      */
    #[inline] pub fn get_num_threads(&self) -> i32 {
        
        todo!();
        /*
            return numThreads_;
        */
    }

    /**
      | Sets the number of threads
      |
      | # of threads should not be bigger than the
      | number of big cores
      */
    #[inline] pub fn set_num_threads(&mut self, num_threads: usize)  {
        
        todo!();
        /*
            if (defaultNumThreads_ == 0) {
        defaultNumThreads_ = getDefaultNumThreads();
      }
      numThreads_ = std::min(numThreads, defaultNumThreads_);
        */
    }

    /**
      | Sets the minimum work size (range) for
      | which to invoke the threadpool; work sizes
      | smaller than this will just be run on the
      | main (calling) thread
      */
    #[inline] pub fn set_min_work_size(&mut self, size: usize)  {
        
        todo!();
        /*
            std::lock_guard<std::mutex> guard(executionMutex_);
      minWorkSize_ = size;
        */
    }

    pub fn run(f: fn(a: i32, b: usize) -> (), range: usize) {
        todo!();
        /*
          const auto numThreads = numThreads_.load(std::memory_order_relaxed);

          std::lock_guard<std::mutex> guard(executionMutex_);
          // If there are no worker threads, or if the range is too small (too
          // little work), just run locally
          const bool runLocally = range < minWorkSize_ ||
              FLAGS_caffe2_threadpool_force_inline || (numThreads == 0);
          if (runLocally) {
            // Work is small enough to just run locally; multithread overhead
            // is too high
            for (size_t i = 0; i < range; ++i) {
              fn(0, i);
            }
            return;
          }

          struct FnTask : public Task {
            FnTask(){};
            ~FnTask() override{};
            const std::function<void(int, size_t)>* fn_;
            int idx_;
            size_t start_;
            size_t end_;
            void Run() override {
              for (auto i = start_; i < end_; ++i) {
                (*fn_)(idx_, i);
              }
            }
          };

          CAFFE_ENFORCE_GE(numThreads_, 1);
          const size_t unitsPerTask = (range + numThreads - 1) / numThreads;
          tasks_.resize(numThreads);
          for (size_t i = 0; i < numThreads; ++i) {
            if (!tasks_[i]) {
              tasks_[i].reset(new FnTask());
            }
            auto* task = (FnTask*)tasks_[i].get();
            task->fn_ = &fn;
            task->idx_ = i;
            task->start_ = std::min<size_t>(range, i * unitsPerTask);
            task->end_ = std::min<size_t>(range, (i + 1) * unitsPerTask);
            if (task->start_ >= task->end_) {
              tasks_.resize(i);
              break;
            }
            CAFFE_ENFORCE_LE(task->start_, range);
            CAFFE_ENFORCE_LE(task->end_, range);
          }
          CAFFE_ENFORCE_LE(tasks_.size(), numThreads);
          CAFFE_ENFORCE_GE(tasks_.size(), 1);
          workersPool_->Execute(tasks_);
        */
    }

    /**
      | Run an arbitrary function in a thread-safe
      | manner accessing the Workers Pool
      |
      */
    pub fn with_pool(f: fn(x: *mut WorkersPool) -> ()) {
        todo!();
        /*
          std::lock_guard<std::mutex> guard(executionMutex_);
          f(workersPool_.get());
        */
    }
}
