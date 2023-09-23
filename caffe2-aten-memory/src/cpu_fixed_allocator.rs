crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/CPUFixedAllocator.h]

/**
  | This file creates a fake allocator that just
  | throws exceptions if it is actually used.
  |
  | state passed to the allocator is the
  | std::function<void(void*)> called when the blob
  | is release by ATen
  */
pub fn cpu_fixed_malloc(
        _0: *mut c_void,
        _1: libc::ptrdiff_t)  {
    
    todo!();
        /*
            AT_ERROR("attempting to resize a tensor view of an external blob");
        */
}

pub fn cpu_fixed_realloc(
    _0: *mut c_void,
    _1: *mut c_void,
    _2: libc::ptrdiff_t

) {
    
    todo!();
        /*
            AT_ERROR("attempting to resize a tensor view of an external blob");
        */
}

pub fn cpu_fixed_free(
    state:      *mut c_void,
    allocation: *mut c_void
) {
    
    todo!();
        /*
            auto on_release = static_cast<std::function<void(void*)>*>(state);
        (*on_release)(allocation);
        delete on_release;
        */
}

lazy_static!{
    /*
    static Allocator CPU_fixed_allocator =
      { cpu_fixed_malloc, cpu_fixed_realloc, cpu_fixed_free };
    */
}
