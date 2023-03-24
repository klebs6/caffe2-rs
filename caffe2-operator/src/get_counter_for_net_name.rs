crate::ix!();

#[inline] pub fn get_counter_for_net_name(net_name: &String) -> i32 {
    
    todo!();
    /*
        // Append a unique number suffix because there could be multiple instances
      // of the same net and we want to uniquely associate each instance with
      // a profiling trace.
      static std::unordered_map<std::string, int> net_name_to_counter;
      static std::mutex map_mutex;
      std::unique_lock<std::mutex> map_lock(map_mutex);
      int counter = net_name_to_counter[net_name] + 1;
      net_name_to_counter[net_name] = counter;
      return counter;
    */
}
