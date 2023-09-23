crate::ix!();

/**
  | Compiler will complain if you put things like
  | std::tuple<Tensor, Tensor> in the `fn` argument
  | of DECLARE_DISPATCH. Some possible workarounds,
  | e.g., adding parentheses and using helper
  | struct to get rid of the parentheses, do not
  | work with MSVC.
  |
  | So do a `using`-declaration if you need to pass
  | in such `fn`, e.g.,
  | grid_sampler_2d_backward_cpu_kernel in
  | GridSampleKernel.h.
  */
#[macro_export] macro_rules! declare_dispatch {
    ($fn:ty, $name:expr) => {
        /*
        
          struct name : DispatchStub<fn, name> {   
            name() = default;                      
            name(const name&) = delete;            
            name& operator=(const name&) = delete; 
          };                                       
          extern TORCH_API struct name name
        */
    }
}

#[macro_export] macro_rules! define_dispatch {
    ($name:expr) => {
        /*
                struct name name
        */
    }
}

#[macro_export] macro_rules! register_no_cpu_dispatch {
    ($name:expr, $fn_type:expr) => {
        /*
        
          REGISTER_ARCH_DISPATCH(name, DEFAULT, static_cast<fn_type>(nullptr))         
          REGISTER_AVX_DISPATCH(name, static_cast<fn_type>(nullptr))                   
          REGISTER_AVX2_DISPATCH(name, static_cast<fn_type>(nullptr))          
          REGISTER_VSX_DISPATCH(name, static_cast<fn_type>(nullptr))
        */
    }
}
