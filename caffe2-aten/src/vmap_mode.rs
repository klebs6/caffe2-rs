crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/VmapMode.h]

/**
  | VmapMode contains a thread local count of how
  | many nested vmaps we are currently inside. That
  | number is known as the `vmap level`.
  |
  | VmapMode is used in the implementation of the
  | Python `torch.vmap` API.
  |
  | NOTE: this is NOT the c++ api for
  | torch.vmap. That doesn't exist yet.
  */
pub struct VmapMode {

}

impl VmapMode {

    /**
      | Returns the vmap level, aka the count
      | of how many nested vmaps we're in.
      |
      */
    pub fn current_vmap_level() -> i64 {
        
        todo!();
        /*
        
        */
    }

    /**
      | Increment the count of nested vmaps. If this
      | causes the vmap level to be greater than 0,
      | then it enables DispatchKey::VmapMode on all
      | tensors.
      |
      */
    pub fn increment_nesting() -> i64 {
        
        todo!();
        /*
        
        */
    }

    /**
      | Decrements the count of nested vmaps. If this
      | causes the vmap level to be equal to 0, then
      | it disables DispatchKey::VmapMode on all
      | tensors.
      |
      */
    pub fn decrement_nesting() -> i64 {
        
        todo!();
        /*
        
        */
    }
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/VmapMode.cpp]

lazy_static!{
    /*
    thread_local i64 VmapMode_current_vmap_level = 0;
    */
}

impl VmapMode {
    
    pub fn current_vmap_level(&mut self) -> i64 {
        
        todo!();
        /*
            return VmapMode_current_vmap_level;
        */
    }
    
    pub fn increment_nesting(&mut self) -> i64 {
        
        todo!();
        /*
            VmapMode_current_vmap_level++;
      if (VmapMode_current_vmap_level == 1) {
        tls_set_dispatch_key_included(DispatchKey::VmapMode, true);
      }
      return VmapMode_current_vmap_level;
        */
    }
    
    pub fn decrement_nesting(&mut self) -> i64 {
        
        todo!();
        /*
            VmapMode_current_vmap_level--;
      if (VmapMode_current_vmap_level == 0) {
        tls_set_dispatch_key_included(DispatchKey::VmapMode, false);
      }
      return VmapMode_current_vmap_level;
        */
    }
}
