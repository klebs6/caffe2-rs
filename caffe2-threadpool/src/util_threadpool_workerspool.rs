/**
  | Uses code derived from gemmlowp, 
  | https://github.com/google/gemmlowp/blob/6c91e1ed0c2eff1182d804310b92911fe9c18019/internal/multi_thread_gemm.h
  | 
  | Changes:
  | 
  | - allocation-free execute()
  | 
  | - Use RAII where possible.
  | 
  | - Run the first task on the main thread
  | (since that is the largest task).
  | 
  | - removed custom allocator.
  | 
  | - Removed some ifdef's
  | 
  | - cache-line align Worker.
  | 
  | - use std::atomic instead of volatile
  | and custom barriers.
  | 
  | - use std::mutex/std::condition_variable
  | instead of raw pthreads.
  |
  */

crate::ix!();

use crate::State;

pub const kGEMMLOWPCacheLineSize: usize = 64;

pub struct AllocAligned<T> {
    phantom: PhantomData<T>,
}

impl<T> AllocAligned<T> {

    /**
      | Allocate a T aligned at an `align` byte
      | address
      |
      */
    #[inline] pub fn alloc<Args>(args: Args) -> *mut T {
        todo!();
        /*
            void* p = nullptr;

        #if defined(__ANDROID__)
            p = memalign(kGEMMLOWPCacheLineSize, sizeof(T));
        #elif defined(_MSC_VER)
            p = _aligned_malloc(sizeof(T), kGEMMLOWPCacheLineSize);
        #else
            posix_memalign((void**)&p, kGEMMLOWPCacheLineSize, sizeof(T));
        #endif

            if (p) {
              return new (p) T(std::forward<Args>(args)...);
            }

            return nullptr;
        */
    }

    /**
      | Free a T previously allocated via
      | AllocAligned<T>::alloc()
      |
      */
    #[inline] pub fn release(p: *mut T)  {
        
        todo!();
        /*
            if (p) {
              p->~T();
        #if defined(_MSC_VER)
              _aligned_free((void*)p);
        #else
              free((void*)p);
        #endif
            }
        */
    }
}

/**
  | Deleter object for unique_ptr for an
  | aligned object
  |
  */
pub struct AlignedDeleter<T> { 
    phantom: PhantomData<T>,
}

impl<T> AlignedDeleter<T> {

    #[inline] pub fn invoke(&self, p: *mut T)  {
        
        todo!();
        /*
            AllocAligned<T>::release(p);
        */
    }
}

pub const kMaxBusyWaitNOPs: i32 = 32 * 1000 * 1000;

pub const GEMMLOWP_NOP:   &'static str = "nop\n";

#[macro_export] macro_rules! GEMMLOWP_NOP4  { () => { GEMMLOWP_NOP.repeat(4); } }
#[macro_export] macro_rules! GEMMLOWP_NOP16 { () => { GEMMLOWP_NOP4![].repeat(4) } }
#[macro_export] macro_rules! GEMMLOWP_NOP64 { () => { GEMMLOWP_NOP16![].repeat(4) } }

#[inline] pub fn do_256nops() -> i32 {
    
    todo!();
    /*
        #if defined(_MSC_VER)
      GEMMLOWP_NOP64;
    #else
      asm volatile(GEMMLOWP_NOP64);
    #endif
      return 64;
    */
}

/**
  | Waits until *var != initial_value.
  | 
  | Returns the new value of *var. The guarantee
  | here is that the return value is different
  | from initial_value, and that that new
  | value has been taken by *var at some point
  | during the execution of this function.
  | There is no guarantee that this is still
  | the value of *var when this function
  | returns, since *var is not assumed to
  | be guarded by any lock.
  | 
  | First does some busy-waiting for a fixed
  | number of no-op cycles, then falls back
  | to passive waiting for the given condvar,
  | guarded by the given mutex.
  | 
  | The idea of doing some initial busy-waiting
  | is to help get better and more consistent
  | multithreading benefits for small
  | GEMM sizes.
  | 
  | Busy-waiting help ensuring that if
  | we need to wake up soon after having started
  | waiting, then we can wake up quickly
  | (as opposed to, say, having to wait to
  | be scheduled again by the OS). On the
  | other hand, we must still eventually
  | revert to passive waiting for longer
  | waits (e.g. worker threads having finished
  | a GEMM and waiting until the next GEMM)
  | so as to avoid permanently spinning.
  |
  */
#[inline] pub fn wait_for_variable_change<T>(
    var:           *mut Atomic<T>,
    initial_value: T,
    cond:          *mut std::sync::Condvar,
    mutex:         *mut parking_lot::RawMutex) -> T 
{
    todo!();
    /*
        // If we are on a platform that supports it, spin for some time.
      {
        int nops = 0;
        // First, trivial case where the variable already changed value.
        T new_value = var->load(std::memory_order_relaxed);
        if (new_value != initial_value) {
          std::atomic_thread_fence(std::memory_order_acquire);
          return new_value;
        }
        // Then try busy-waiting.
        while (nops < kMaxBusyWaitNOPs) {
          nops += Do256NOPs();
          new_value = var->load(std::memory_order_relaxed);
          if (new_value != initial_value) {
            std::atomic_thread_fence(std::memory_order_acquire);
            return new_value;
          }
        }
      }

      // Finally, do real passive waiting.
      {
        std::unique_lock<std::mutex> g(*mutex);
        T new_value = var->load(std::memory_order_relaxed);
        // Handle spurious wakeups.
        cond->wait(g, [&]() {
          new_value = var->load(std::memory_order_relaxed);
          return new_value != initial_value;
        });
        DCHECK_NE(static_cast<size_t>(new_value), static_cast<size_t>(initial_value));
        return new_value;
      }
    */
}

/**
  | A BlockingCounter lets one thread to
  | wait for N events to occur.
  | 
  | This is how the master thread waits for
  | all the worker threads to have finished
  | working.
  |
  */
pub struct BlockingCounter {

    cond:  std::sync::Condvar,
    mutex: parking_lot::RawMutex,
    count: Atomic<usize>, // default = 0
}

impl BlockingCounter {

    /**
      | Sets/resets the counter; initial_count
      | is the number of decrementing events
      | that the Wait() call will be waiting for.
      |
      */
    #[inline] pub fn reset(&mut self, initial_count: usize)  {
        
        todo!();
        /*
            std::lock_guard<std::mutex> g(mutex_);
        DCHECK_EQ(count_, 0);
        count_ = initial_count;
        */
    }

    /**
      | Decrements the counter; if the counter
      | hits zero, signals the thread that was
      | waiting for that, and returns true.
      |
      | Otherwise (if the decremented count is
      | still nonzero), returns false.
      */
    #[inline] pub fn decrement_count(&mut self) -> bool {
        
        todo!();
        /*
            const auto count_value = count_.fetch_sub(1, std::memory_order_relaxed) - 1;
        DCHECK_GE(count_value, 0);
        if (count_value == 0) {
          std::lock_guard<std::mutex> g(mutex_);
          cond_.notify_one();
        }
        bool retval = count_value == 0;
        return retval;
        */
    }

    /**
      | Waits for the N other threads (N having
      | been set by Reset()) to hit the
      | BlockingCounter.
      |
      */
    #[inline] pub fn wait(&mut self)  {
        
        todo!();
        /*
            while (size_t count_value = count_.load(std::memory_order_relaxed)) {
          WaitForVariableChange(&count_, count_value, &cond_, &mutex_);
        }
        */
    }
}

/// A workload for a worker.
pub trait Task {
    fn run() where Self: Sized;
}

#[repr(u8)]
pub enum WorkerState {

    /**
      | The initial state before the thread
      | main loop runs.
      |
      */
    ThreadStartup, 

    /**
      | Is not working, has not yet received
      | new work to do.
      |
      */
    Ready, 

    /// Has work to do.
    HasWork, 

    /**
      | Should exit at earliest convenience.
      |
      */
    ExitAsSoonAsPossible 
}

/// A worker thread.
#[repr(align(64))] //kGEMMLOWPCacheLineSize
pub struct Worker {

    /// The underlying thread.
    thread:      Box<std::thread::Thread>,

    /// The task to be worked on.
    task:        AtomicPtr<*mut dyn Task>,

    /**
      | The condition variable and mutex guarding
      | state changes.
      |
      */
    state_cond:  std::sync::Condvar,
    state_mutex: parking_lot::RawMutex,

    /**
      | The state enum tells if we're currently
      | working, waiting for work, etc.
      |
      */
    state:       Atomic<State>,

    /**
      | pointer to the master's thread
      | BlockingCounter object, to notify the
      | master thread of when this worker switches
      | to the 'Ready' state.
      */
    counter_to_decrement_when_ready: *const BlockingCounter,
}

impl Drop for Worker {
    fn drop(&mut self) {
        todo!();
        /* 
        ChangeState(State::ExitAsSoonAsPossible);
        thread_->join();
       */
    }
}

impl Worker {
    
    pub fn new(counter_to_decrement_when_ready: *mut BlockingCounter) -> Self {
        todo!();
        /*
            : task_(nullptr),
            state_(State::ThreadStartup),
            counter_to_decrement_when_ready_(counter_to_decrement_when_ready) 

        thread_ = std::make_unique<std::thread>([this]() { this->ThreadFunc(); });
        */
    }

    /**
      | Changes State; may be called from either
      | the worker thread or the master thread;
      | however, not all state transitions are
      | legal, which is guarded by assertions.
      */
    #[inline] pub fn change_state(&mut self, new_state: WorkerState)  {
        
        todo!();
        /*
            std::lock_guard<std::mutex> g(state_mutex_);
        DCHECK(new_state != state_.load(std::memory_order_relaxed));
        switch (state_.load(std::memory_order_relaxed)) {
        case State::ThreadStartup:
          DCHECK(new_state == State::Ready);
          break;
        case State::Ready:
          DCHECK(new_state == State::HasWork || new_state == State::ExitAsSoonAsPossible);
          break;
        case State::HasWork:
          DCHECK(new_state == State::Ready || new_state == State::ExitAsSoonAsPossible);
          break;
        default:
          abort();
        }
        state_.store(new_state, std::memory_order_relaxed);
        state_cond_.notify_one();
        if (new_state == State::Ready) {
          counter_to_decrement_when_ready_->DecrementCount();
        }
        */
    }

    /// Thread entry point.
    #[inline] pub fn thread_func(&mut self)  {
        
        todo!();
        /*
            c10::setThreadName("CaffeWorkersPool");
        ChangeState(State::Ready);

        // Thread main loop
        while (true) {
          // Get a state to act on
          // In the 'Ready' state, we have nothing to do but to wait until
          // we switch to another state.
          State state_to_act_upon =
              WaitForVariableChange(&state_, State::Ready, &state_cond_, &state_mutex_);

          // We now have a state to act on, so act.
          switch (state_to_act_upon) {
          case State::HasWork:
            // Got work to do! So do it, and then revert to 'Ready' state.
            DCHECK(task_.load());
            (*task_).Run();
            task_ = nullptr;
            ChangeState(State::Ready);
            break;
          case State::ExitAsSoonAsPossible:
            return;
          default:
            abort();
          }
        }
        */
    }
    
    #[inline] pub fn thread_func_with_arg(arg: *mut c_void)  {
        
        todo!();
        /*
            static_cast<Worker*>(arg)->ThreadFunc();
        return nullptr;
        */
    }

    /**
      | Called by the master thread to give this
      | worker work to do.
      |
      | It is only legal to call this if the
      | worker
      */
    #[inline] pub fn start_work<T: Task>(&mut self, task: *mut T)  {
        
        todo!();
        /*
            DCHECK(!task_.load());
        task_ = task;
        DCHECK(state_.load(std::memory_order_acquire) == State::Ready);
        ChangeState(State::HasWork);
        */
    }
}

///-------------------------------------

pub struct WorkersPool { 

    workers: Vec<Box<Worker>>,

    /// The BlockingCounter used to wait for the workers.
    counter_to_decrement_when_ready: BlockingCounter,
}

impl WorkersPool {
    
    #[inline] pub fn execute<T: Task>(&mut self, tasks: &Vec<Arc<T>>)  {
        
        todo!();
        /*
            CAFFE_ENFORCE_GE(tasks.size(), 1);
        // One of the tasks will be run on the current thread.
        int workers_count = tasks.size() - 1;
        CreateWorkers(workers_count);
        DCHECK_LE(workers_count, (int)workers_.size());
        counter_to_decrement_when_ready_.Reset(workers_count);
        for (size_t task = 1; task < tasks.size(); ++task) {
          workers_[task - 1]->StartWork(tasks[task].get());
        }
        // Execute the remaining workload immediately on the current thread.
        auto& task = tasks.front();
        task->Run();
        // Wait for the workers submitted above to finish.
        counter_to_decrement_when_ready_.Wait();
        */
    }

    /**
      | Ensures that the pool has at least the
      | given count of workers.
      |
      | If any new worker has to be created, this
      | function waits for it to be ready.
      */
    #[inline] pub fn create_workers(&mut self, workers_count: usize)  {
        
        todo!();
        /*

        /*
        /// make_unique that guarantees alignment
        template <typename T>
        struct MakeAligned {
          template <typename... Args>
          static std::unique_ptr<T, AlignedDeleter<T>> make(Args&&... args) {
            return std::unique_ptr<T, AlignedDeleter<T>>(
                AllocAligned<T>::alloc(std::forward<Args>(args)...));
          }
        };
        */
            if (workers_.size() >= workers_count) {
          return;
        }
        counter_to_decrement_when_ready_.Reset(workers_count - workers_.size());
        while (workers_.size() < workers_count) {
          workers_.push_back(MakeAligned<Worker>::make(&counter_to_decrement_when_ready_));
        }
        counter_to_decrement_when_ready_.Wait();
        */
    }
}
