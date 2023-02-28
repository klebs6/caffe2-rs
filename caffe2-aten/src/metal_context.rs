crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/metal/Context.h]

pub trait MetalInterface:
IsMetalAvailable
+ MetalCopy {}

pub trait IsMetalAvailable {
    
    fn is_metal_available(&self) -> bool;
}

pub trait MetalCopy {

    fn metal_copy(&self, 
        self_: &mut Tensor,
        src:   &Tensor) -> &mut Tensor;
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/metal/Context.cpp]

lazy_static!{
    /*
    atomic<const MetalInterface*> g_metal_impl_registry;
    */
}

pub struct MetalImplRegistrar {

}

impl MetalImplRegistrar {
    
    pub fn new(impl_: *mut MetalInterface) -> Self {
    
        todo!();
        /*


            g_metal_impl_registry.store(impl);
        */
    }
}

pub fn metal_copy(
        self_: &mut Tensor,
        src:   &Tensor) -> &mut Tensor {
    
    todo!();
        /*
            auto p = metal::g_metal_impl_registry.load();
      if (p) {
        return p->metal_copy_(self, src);
      }
      AT_ERROR("Metal backend was not linked to the build");
        */
}

pub fn is_metal_available() -> bool {
    
    todo!();
        /*
            auto p = metal::g_metal_impl_registry.load();
      return p ? p->is_metal_available() : false;
        */
}
