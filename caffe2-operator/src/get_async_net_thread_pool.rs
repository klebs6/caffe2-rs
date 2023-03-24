crate::ix!();

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
