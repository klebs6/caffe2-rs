crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/core/thread_pool.h]

pub trait TaskThreadPoolBaseInterface:
Run
+ Size
+ NumAvailable
+ InThreadPool {
    
    fn default_num_threads(&self) -> usize {
        
        todo!();
        /*
            auto num_threads = thread::hardware_concurrency();
    #if defined(_M_X64) || defined(__x86_64__)
        num_threads /= 2;
    #endif
        return num_threads;
        */
    }
}

pub trait Run {

    fn run(&mut self, func: fn() -> ());
}

pub trait Size {

    fn size(&self) -> usize;
}

pub trait NumAvailable {

    /**
     | The number of available (i.e. idle)
     | threads in this thread pool.
     |
     */
    fn num_available(&self) -> usize;
}

pub trait InThreadPool {

    /**
     | Check if the current thread is from the
     | thread pool.
     |
     */
    fn in_thread_pool(&self) -> bool;
}

pub struct ThreadPoolTaskElement {
    run_with_id: bool,
    no_id:       fn() -> (),
    with_id:     fn(_0: usize) -> (),
}

impl ThreadPoolTaskElement {
    
    pub fn new(f: fn() -> ()) -> Self {
    
        todo!();
        /*
        : run_with_id(false),
        : no_id(move(f)),
        : with_id(nullptr),

        
        */
    }
    
    pub fn new_a(f: fn(_0: usize) -> ()) -> Self {
    
        todo!();
        /*
        : run_with_id(true),
        : no_id(nullptr),
        : with_id(move(f)),

        
        */
    }
}

pub struct C10ThreadPool {
    tasks:        SegQueue<ThreadPoolTaskElement>,
    threads:      Vec<Thread>,
    mutex:        RefCell<RawMutex>,
    condition:    Condvar,
    completed:    Condvar,
    running:      AtomicBool,
    complete:     bool,
    available:    usize,
    total:        usize,
    numa_node_id: i32,
}

impl TaskThreadPoolBaseInterface for C10ThreadPool { }

impl Run for C10ThreadPool {

    fn run(&mut self, func: fn() -> ())  {
        
        todo!();
        /*
            if (threads_.size() == 0) {
        throw runtime_error("No threads to run a task");
      }
      unique_lock<mutex> lock(mutex_);

      // Set task and signal condition variable so that a worker thread will
      // wake up and use the task.
      tasks_.emplace(move(func));
      complete_ = false;
      condition_.notify_one();
        */
    }
}

impl Size for C10ThreadPool {

    fn size(&self) -> usize {
        
        todo!();
        /*
            return threads_.size();
        */
    }
}

impl NumAvailable for C10ThreadPool {

    fn num_available(&self) -> usize {
        
        todo!();
        /*
            unique_lock<mutex> lock(mutex_);
      return available_;
        */
    }
}

impl InThreadPool for C10ThreadPool {

    fn in_thread_pool(&self) -> bool {
        
        todo!();
        /*
            for (auto& thread : threads_) {
        if (thread.get_id() == this_thread::get_id()) {
          return true;
        }
      }
      return false;
        */
    }
}

impl C10ThreadPool {
    
    pub fn run_task_withid<Task>(&mut self, task: Task)  {
    
        todo!();
        /*
            unique_lock<mutex> lock(mutex_);

        // Set task and signal condition variable so that a worker thread will
        // wake up and use the task.
        tasks_.emplace(static_cast<function<void(size_t)>>(task));
        complete_ = false;
        condition_.notify_one();
        */
    }

    /// @brief Wait for queue to be empty
    pub fn wait_work_complete(&mut self)  {
        
        todo!();
        /*
        
        */
    }
 
    /// @brief Entry point for pool threads.
    pub fn main_loop(&mut self, index: usize)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn new(
        pool_size:    i32,
        numa_node_id: Option<i32>,
        init_thread:  fn() -> ()) -> Self {

        let numa_node_id: i32 = numa_node_id.unwrap_or(-1);
    
        todo!();
        /*
            : threads_(pool_size < 0 ? defaultNumThreads() : pool_size),
          running_(true),
          complete_(true),
          available_(threads_.size()),
          total_(threads_.size()),
          numa_node_id_(numa_node_id) 

      for (size_t i = 0; i < threads_.size(); ++i) {
        threads_[i] = thread([this, i, init_thread]() {
          if (init_thread) {
            init_thread();
          }
          this->main_loop(i);
        });
      }
        */
    }
    
    pub fn wait_work_complete_a(&mut self)  {
        
        todo!();
        /*
            unique_lock<mutex> lock(mutex_);
      while (!complete_) {
        completed_.wait(lock);
      }
        */
    }

    pub fn main_loop_a(&mut self, index: usize)  {
        
        todo!();
        /*
            unique_lock<mutex> lock(mutex_);
      while (running_) {
        // Wait on condition variable while the task is empty and
        // the pool is still running.
        while (tasks_.empty() && running_) {
          condition_.wait(lock);
        }
        // If pool is no longer running, break out of loop.
        if (!running_) {
          break;
        }

        // Copy task locally and remove from the queue.  This is
        // done within its own scope so that the task object is
        // destructed immediately after running the task.  This is
        // useful in the event that the function contains
        // shared_ptr arguments bound via bind.
        {
          task_element_t tasks = move(tasks_.front());
          tasks_.pop();
          // Decrement count, indicating thread is no longer available.
          --available_;

          lock.unlock();

          // Run the task.
          try {
            if (tasks.run_with_id) {
              tasks.with_id(index);
            } else {
              tasks.no_id();
            }
          } catch (const exception& e) {
            LOG(ERROR) << "Exception in thread pool task: " << e.what();
          } catch (...) {
            LOG(ERROR) << "Exception in thread pool task: unknown";
          }

          // Destruct tasks before taking the lock.  As tasks
          // are user provided function, they can run
          // arbitrary code during destruction, including code
          // that can reentrantly call into C10ThreadPool (which would
          // cause a deadlock if we were holding the lock).
        }

        // Update status of empty, maybe
        // Need to recover the lock first
        lock.lock();

        // Increment count, indicating thread is available.
        ++available_;
        if (tasks_.empty() && available_ == total_) {
          complete_ = true;
          completed_.notify_one();
        }

        // Deliberately hold the lock on the backedge, so this thread has an
        // opportunity to acquire a new task before another thread acquires
        // the lock.
      } // while running_
        */
    }
}


pub struct TaskThreadPool {
    base: C10ThreadPool,
}

impl TaskThreadPool {
    
    pub fn new(
        pool_size:    usize,
        numa_node_id: Option<i32>) -> Self {

        let numa_node_id: i32 = numa_node_id.unwrap_or(-1);

        todo!();
        /*


            : C10ThreadPool(pool_size, numa_node_id, [numa_node_id]() {
              setThreadName("CaffeTaskThread");
              NUMABind(numa_node_id);
            })
        */
    }
}

c10_declare_shared_registry!{
    ThreadPoolRegistry,
    TaskThreadPoolBase,
    int,
    int,
    bool
}

//-------------------------------------------[.cpp/pytorch/c10/core/thread_pool.cpp]

impl Drop for C10ThreadPool {

    fn drop(&mut self) {
        todo!();
        /*
            // Set running flag to false then notify all threads.
      {
        unique_lock<mutex> lock(mutex_);
        running_ = false;
        condition_.notify_all();
      }

      for (auto& t : threads_) {
        try {
          t.join();
        } catch (const exception&) {
        }
      }
        */
    }
}

c10_define_shared_registry!{
    ThreadPoolRegistry,
    TaskThreadPoolBase,
    int,
    int,
    bool
}
