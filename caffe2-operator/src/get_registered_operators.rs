crate::ix!();

/// Get a set of registered operator names
#[inline] pub fn get_registered_operators() -> HashSet<String> {
    
    todo!();
    /*
        std::set<std::string> all_keys;

      // CPU operators
      for (const auto& name : CPUOperatorRegistry()->Keys()) {
        all_keys.emplace(name);
      }
      // CUDA operators
      for (const auto& name : CUDAOperatorRegistry()->Keys()) {
        all_keys.emplace(name);
      }

      // HIP operators
      for (const auto& name : HIPOperatorRegistry()->Keys()) {
        all_keys.emplace(name);
      }

      return all_keys;
    */
}
