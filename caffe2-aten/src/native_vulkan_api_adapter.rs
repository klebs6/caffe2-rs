crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/api/Adapter.h]

/**
  | A Vulkan Adapter represents a physical device
  | and its properties.
  |
  | Adapters are enumerated through the Runtime and
  | are used in creation of Contexts.
  |
  | Each tensor in PyTorch is associated with
  | a Context to make the device <-> tensor
  | affinity explicit.
  |
  */
#[cfg(USE_VULKAN_API)]
pub struct VulkanAdapter {
    runtime:                    *mut Runtime,
    handle:                     VkPhysicalDevice,
    properties:                 VkPhysicalDeviceProperties,
    memory_properties:          VkPhysicalDeviceMemoryProperties,
    compute_queue_family_index: u32,
}

#[cfg(USE_VULKAN_API)]
impl VulkanAdapter {

    #[inline] pub fn has_unified_memory(&self) -> bool {
        
        todo!();
        /*
            // Ideally iterate over all memory types to see if there is a pool that
        // is both host-visible, and device-local.  This should be a good proxy
        // for now.
        return VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU == properties.deviceType;
        */
    }
    
    #[inline] pub fn local_work_group_size(&self) -> Shader_WorkGroup {
        
        todo!();
        /*
            return { 4u, 4u, 4u, };
        */
    }
}
