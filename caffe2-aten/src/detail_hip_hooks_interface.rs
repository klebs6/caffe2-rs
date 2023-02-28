crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/detail/HIPHooksInterface.h]

/**
  | The HIPHooksInterface is an omnibus interface
  | for any HIP functionality which we may want to
  | call into from CPU code (and thus must be
  | dynamically dispatched, to allow for separate
  | compilation of HIP code).
  |
  | See CUDAHooksInterface for more detailed
  | motivation.
  */
pub trait HIPHooksInterface {
    
    /// Initialize THHState and, transitively, the
    /// HIP state
    ///
    fn init_hip(&self) -> Box<THHState> {
        
        todo!();
        /*
        AT_ERROR("Cannot initialize HIP without ATen_hip library.");
        */
    }
    
    fn init_hip_generator(&self, _0: *mut Context) -> Box<GeneratorImpl> {
        
        todo!();
        /*
            AT_ERROR("Cannot initialize HIP generator without ATen_hip library.");
        */
    }
    
    fn has_hip(&self) -> bool {
        
        todo!();
        /*
            return false;
        */
    }
    
    fn current_device(&self) -> i64 {
        
        todo!();
        /*
            return -1;
        */
    }
    
    fn get_pinned_memory_allocator(&self) -> *mut Allocator {
        
        todo!();
        /*
            AT_ERROR("Pinned memory requires HIP.");
        */
    }
    
    fn register_hip_types(&self, _0: *mut Context)  {
        
        todo!();
        /*
            AT_ERROR("Cannot registerHIPTypes() without ATen_hip library.");
        */
    }
    
    fn get_num_gpu_s(&self) -> i32 {
        
        todo!();
        /*
            return 0;
        */
    }
}

/**
  | NB: dummy argument to suppress "ISO
  | C++11 requires at least one argument
  | for the "..." in a variadic macro"
  |
  */
pub struct  HIPHooksArgs {}

c10_declare_registry!{
    HIPHooksRegistry, 
    HIPHooksInterface, 
    HIPHooksArgs
}

#[macro_export] macro_rules! register_hip_hooks {
    ($clsname:ident) => {
        /*
        
          C10_REGISTER_CLASS(HIPHooksRegistry, clsname, clsname)
        */
    }
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/detail/HIPHooksInterface.cpp]

/**
  | See getCUDAHooks for some more commentary
  |
  */
pub fn get_hip_hooks() -> &HIPHooksInterface {
    
    todo!();
        /*
            static unique_ptr<HIPHooksInterface> hip_hooks;
    #if !defined C10_MOBILE
      static once_flag once;
      call_once(once, [] {
        hip_hooks = HIPHooksRegistry()->Create("HIPHooks", HIPHooksArgs{});
        if (!hip_hooks) {
          hip_hooks =
              unique_ptr<HIPHooksInterface>(new HIPHooksInterface());
        }
      });
    #else
      if (hip_hooks == nullptr) {
        hip_hooks =
            unique_ptr<HIPHooksInterface>(new HIPHooksInterface());
      }
    #endif
      return *hip_hooks;
        */
}

c10_define_registry!{
    HIPHooksRegistry, 
    HIPHooksInterface, 
    HIPHooksArgs
}
