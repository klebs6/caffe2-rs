crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/vulkan/Context.h]

pub trait VulkanInterface:
IsVulkanAvailable
+ VulkanCopy {}

pub trait IsVulkanAvailable {

    fn is_vulkan_available(&self) -> bool;
}

pub trait VulkanCopy {

    fn vulkan_copy(&self, 
        self_: &mut Tensor,
        src:   &Tensor) -> &mut Tensor;
}

lazy_static!{
    /*
    extern atomic<const VulkanImplInterface*> g_vulkan_impl_registry;
    */
}

pub struct VulkanImplRegistrar {

}

impl VulkanImplRegistrar {

    pub fn new(_0: *mut VulkanImplInterface) -> Self {
    
        todo!();
        /*
        
        */
    }
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/vulkan/Context.cpp]

lazy_static!{
    /*
    atomic<const VulkanImplInterface*> g_vulkan_impl_registry;
    */
}

impl VulkanImplRegistrar {
    
    pub fn new(impl_: *mut VulkanImplInterface) -> Self {
    
        todo!();
        /*

            g_vulkan_impl_registry.store(impl);
        */
    }
}

pub fn vulkan_copy(
        self_: &mut Tensor,
        src:   &Tensor) -> &mut Tensor {
    
    todo!();
        /*
            auto p = vulkan::g_vulkan_impl_registry.load();
      if (p) {
        return p->vulkan_copy_(self, src);
      }
      AT_ERROR("Vulkan backend was not linked to the build");
        */
}

pub fn is_vulkan_available() -> bool {
    
    todo!();
        /*
            #ifdef USE_VULKAN_API
      return native::vulkan::api::available();
    #else
      auto p = vulkan::g_vulkan_impl_registry.load();
      return p ? p->is_vulkan_available() : false;
    #endif
        */
}
