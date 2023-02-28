crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/mkldnn/IDeepRegistration.cpp]

#[cfg(feature = "mkldnn")]
lazy_static!{
    /*
    RegisterEngineAllocator cpu_alloc(
      engine::cpu_engine(),
      [](usize size) {
        return c10::GetAllocator(c10::DeviceType::CPU)->raw_allocate(size);
      },
      [](void* p) {
        c10::GetAllocator(c10::DeviceType::CPU)->raw_deallocate(p);
      }
    );
    */
}

#[cfg(feature = "mkldnn")]
pub mod mkldnn {

    use super::*;

    pub fn clear_computation_cache()  {
        
        todo!();
            /*
                // Reset computation_cache for forward convolutions
          // As it also caches max number of OpenMP workers
          ideep::convolution_forward::t_store().clear();
            */
    }
}
