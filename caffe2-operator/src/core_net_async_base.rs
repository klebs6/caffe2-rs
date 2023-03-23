crate::ix!();

declare_int!{caffe2_streams_per_gpu}
declare_int!{caffe2_net_async_max_gpus}
declare_int!{caffe2_net_async_max_numa_nodes}
declare_int!{caffe2_net_async_thread_pool_size}
declare_bool!{caffe2_net_async_check_stream_status}
declare_bool!{caffe2_net_async_use_single_pool}
declare_bool!{caffe2_net_async_use_per_net_pools}
declare_bool!{caffe2_net_async_run_root_tasks_inline}
declare_bool!{caffe2_net_async_profile_operators}

///--------------------------------------

struct AsyncNetCancelled { }

impl AsyncNetCancelled {
    
    #[inline] pub fn what(&self) -> *const u8 {
        
        todo!();
        /*
            return "Cancelled";
        */
    }
}

/**
  | first int key - device id, second - pool
  | size, one pool per (device, size)
  |
  */
pub type PoolsMap = HashMap<i32,HashMap<i32,Arc<dyn TaskThreadPoolBaseInterface>>>;


///------------------------------------------
pub struct AsyncNetExecutorHelper {
    base: ExecutorHelper,
    net:  *mut AsyncNetBase,
}

impl AsyncNetExecutorHelper {
    
    pub fn new(net: *mut AsyncNetBase) -> Self {
    
        todo!();
        /*
            : net_(net)
        */
    }
    
    #[inline] pub fn get_pool(&self, option: &DeviceOption) -> Arc<dyn TaskThreadPoolBaseInterface> {
        
        todo!();
        /*
            return net_->pool(option);
        */
    }
}

#[inline] pub fn get_async_net_thread_pool<TaskThreadPoolImpl, const device_type: i32>(
    device_id:  i32,
    pool_size:  i32,
    create_new: bool) -> Arc<dyn TaskThreadPoolBaseInterface> {

    todo!();
    /*
        static std::unordered_map<
          int,
          std::unordered_map<int, std::weak_ptr<TaskThreadPoolBase>>>
          pools;
      static std::mutex pool_mutex;

      const auto& device_type_name = DeviceTypeName(device_type);

      if (pool_size <= 0) {
        if (FLAGS_caffe2_net_async_thread_pool_size > 0) {
          pool_size = FLAGS_caffe2_net_async_thread_pool_size;
          LOG(INFO) << "Using default " << device_type_name
                    << " pool size: " << pool_size << "; device id: " << device_id;
        } else {
          auto num_cores = std::thread::hardware_concurrency();
          CAFFE_ENFORCE(num_cores > 0, "Failed to get number of CPU cores");
          LOG(INFO) << "Using estimated " << device_type_name
                    << " pool size: " << num_cores << "; device id: " << device_id;
          pool_size = num_cores;
        }
      } else {
        LOG(INFO) << "Using specified " << device_type_name
                  << " pool size: " << pool_size << "; device id: " << device_id;
      }

      if (create_new) {
        LOG(INFO) << "Created new " << device_type_name
                  << " pool, size: " << pool_size << "; device id: " << device_id;
        return std::make_shared<TaskThreadPoolImpl>(pool_size, device_id);
      } else {
        std::lock_guard<std::mutex> lock(pool_mutex);

        auto shared_pool = pools[device_id][pool_size].lock();
        if (!shared_pool) {
          LOG(INFO) << "Created shared " << device_type_name
                    << " pool, size: " << pool_size << "; device id: " << device_id;
          shared_pool = std::make_shared<TaskThreadPoolImpl>(pool_size, device_id);
          pools[device_id][pool_size] = shared_pool;
        }
        return shared_pool;
      }
    */
}

/**
  | experimental support for multiple
  | streams per worker per GPU
  |
  */
define_int!{caffe2_streams_per_gpu,
    1,
    "Number of streams per worker per GPU to use in GPU thread pool (experimental)"
}

define_bool!{caffe2_net_async_inference_mode,
    false,
    "If set, use one single chain containing all ops"}

define_bool!{caffe2_net_async_profile_operators,
    false,
    "If set, profile operators of the net regardless of net being prof_dag."}

define_int!{caffe2_net_async_max_gpus,
    16,
    "Max number of GPUs allowed in net async executor"}

define_int!{caffe2_net_async_max_numa_nodes,
    8,
    "Max number of NUMA nodes allowed in net async executor"}

define_int!{caffe2_net_async_thread_pool_size,
    0,
    "Number of threads in device thread pool by default"}

define_bool!{caffe2_net_async_check_stream_status,
    false,
    "Select next non-busy stream"}

define_bool!{caffe2_net_async_use_single_pool,
    false,
    "Use single thread pool for all devices"}

define_bool!{caffe2_net_async_use_per_net_pools,
    false,
    "Use per net thread pools"}

define_bool!{caffe2_net_async_run_root_tasks_inline,
    false,
    "Run root tasks in current thread instread of scheduling to threadpool"}


register_creator!{
    /*
    ThreadPoolRegistry,
    CPU,
    caffe2::GetAsyncNetThreadPool<TaskThreadPool, caffe2::PROTO_CPU>
    */
}

register_creator!{
    /*
    ThreadPoolRegistry,
    CUDA,
    caffe2::GetAsyncNetThreadPool<TaskThreadPool, caffe2::PROTO_CUDA>
    */
}

register_creator!{
    /*
    ThreadPoolRegistry,
    HIP,
    caffe2::GetAsyncNetThreadPool<TaskThreadPool, caffe2::PROTO_HIP>
    */
}
