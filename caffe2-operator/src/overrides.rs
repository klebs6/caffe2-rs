crate::ix!();

#[inline] pub fn default_overrides<'a>() -> &'a HashMap<String, String> {
    
    todo!();
    /*
        // redirecting legacy net types to async_scheduling (except for 'simple');
      // async_scheduling checks net type for backward compatibility
      static const std::unordered_map<std::string, std::string> overrides = {
          {"dag", "async_scheduling"},
          {"prof_dag", "async_scheduling"},
          {"async_dag", "async_scheduling"},
          {"async_polling", "async_scheduling"},
          {"async_simple", "simple"}, // "async_simple" impl has been removed.
          {"rnn", "simple"}, // "rnn" impl has been removed.
      };
      return overrides;
    */
}

#[inline] pub fn apply_potential_executor_override(net_type: *mut String)  {
    
    todo!();
    /*
        auto executors = caffe2::split(',', FLAGS_caffe2_override_executor);
      CAFFE_ENFORCE(
          executors.size() % 2 == 0, "Invalid override executors flag value");
      std::unordered_map<std::string, std::string> overrides;
      for (const auto& kv : defaultOverrides()) {
        overrides[kv.first] = kv.second;
      }
      for (size_t idx = 0; idx < executors.size(); idx += 2) {
        overrides[executors[idx]] = executors[idx + 1];
      }
      if (overrides.count(*net_type)) {
        VLOG(1) << "Overrode net type '" << *net_type << "' with '"
                << overrides[*net_type] << "'";
        *net_type = overrides[*net_type];
      }
    */
}
