crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/util/numa.h]

//-------------------------------------------[.cpp/pytorch/c10/util/numa.cpp]

#[cfg(all(__linux__,C10_USE_NUMA,not(C10_MOBILE)))]
pub const C10_ENABLE_NUMA: bool = true;

// This code used to have a lot of VLOGs. However,
// because allocation might be triggered during
// static initialization, it's unsafe to invoke
// VLOG here


/**
  | Check whether NUMA is enabled
  |
  */
#[cfg(C10_ENABLE_NUMA)]
pub fn is_numa_enabled() -> bool {
    
    todo!();
        /*
            return FLAGS_caffe2_cpu_numa_enabled && numa_available() >= 0;
        */
}

#[cfg(not(C10_ENABLE_NUMA))]
pub fn is_numa_enabled() -> bool {
    
    todo!();
        /*
            return false;
        */
}

#[cfg(not(C10_ENABLE_NUMA))]
pub fn numa_bind(numa_node_id: i32)  { }

/**
  | Bind to a given NUMA node
  |
  */
#[cfg(C10_ENABLE_NUMA)]
pub fn numa_bind(numa_node_id: i32)  {
    
    todo!();
        /*
            if (numa_node_id < 0) {
        return;
      }
      if (!IsNUMAEnabled()) {
        return;
      }

      TORCH_CHECK(
          numa_node_id <= numa_max_node(),
          "NUMA node id ",
          numa_node_id,
          " is unavailable");

      auto bm = numa_allocate_nodemask();
      numa_bitmask_setbit(bm, numa_node_id);
      numa_bind(bm);
      numa_bitmask_free(bm);
        */
}

#[cfg(not(C10_ENABLE_NUMA))]
pub fn get_numa_node(ptr: *const c_void) -> i32 {
    -1
}

/**
  | Get the NUMA id for a given pointer `ptr`
  |
  */
#[cfg(C10_ENABLE_NUMA)]
pub fn get_numa_node(ptr: *const c_void) -> i32 {
    
    todo!();
        /*
            if (!IsNUMAEnabled()) {
        return -1;
      }
      AT_ASSERT(ptr);

      int numa_node = -1;
      TORCH_CHECK(
          get_mempolicy(
              &numa_node,
              NULL,
              0,
              const_cast<void*>(ptr),
              MPOL_F_NODE | MPOL_F_ADDR) == 0,
          "Unable to get memory policy, errno:",
          errno);
      return numa_node;
        */
}

#[cfg(not(C10_ENABLE_NUMA))]
pub fn get_num_numa_nodes() -> i32 {
    -1
}

/**
  | Get number of NUMA nodes
  |
  */
#[cfg(C10_ENABLE_NUMA)]
pub fn get_num_numa_nodes() -> i32 {
    
    todo!();
        /*
            if (!IsNUMAEnabled()) {
        return -1;
      }

      return numa_num_configured_nodes();
        */
}

#[cfg(not(C10_ENABLE_NUMA))]
pub fn numa_move(
        ptr:          *mut c_void,
        size:         usize,
        numa_node_id: i32)  { }
/**
  | Move the memory pointed to by `ptr` of
  | a given size to another NUMA node
  |
  */
#[cfg(C10_ENABLE_NUMA)]
pub fn numa_move(
        ptr:          *mut c_void,
        size:         usize,
        numa_node_id: i32)  {
    
    todo!();
        /*
            if (numa_node_id < 0) {
        return;
      }
      if (!IsNUMAEnabled()) {
        return;
      }
      AT_ASSERT(ptr);

      uintptr_t page_start_ptr =
          ((reinterpret_cast<uintptr_t>(ptr)) & ~(getpagesize() - 1));
      ptrdiff_t offset = reinterpret_cast<uintptr_t>(ptr) - page_start_ptr;
      // Avoid extra dynamic allocation and NUMA api calls
      AT_ASSERT(
          numa_node_id >= 0 &&
          static_cast<unsigned>(numa_node_id) < sizeof(unsigned long) * 8);
      unsigned long mask = 1UL << numa_node_id;
      TORCH_CHECK(
          mbind(
              reinterpret_cast<void*>(page_start_ptr),
              size + offset,
              MPOL_BIND,
              &mask,
              sizeof(mask) * 8,
              MPOL_MF_MOVE | MPOL_MF_STRICT) == 0,
          "Could not move memory to a NUMA node");
        */
}

/**
  | Get the current NUMA node id
  |
  */
#[cfg(C10_ENABLE_NUMA)]
pub fn get_current_numa_node() -> i32 {
    
    todo!();
        /*
            if (!IsNUMAEnabled()) {
        return -1;
      }

      auto n = numa_node_of_cpu(sched_getcpu());
      return n;
        */
}

#[cfg(not(C10_ENABLE_NUMA))]
pub fn get_current_numa_node() -> i32 {
    -1
}
