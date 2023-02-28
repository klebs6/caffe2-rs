crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/api/Common.h]

#[cfg(USE_VULKAN_SHADERC_RUNTIME)]
#[macro_export] macro_rules! vk_kernel {
    ($name:ident) => {
        /*
        
          ::native::vulkan::api::Shader::Descriptor{ 
            name##_glsl,                                 
          }
        */
    }
}

#[cfg(not(USE_VULKAN_SHADERC_RUNTIME))]
#[macro_export] macro_rules! vk_kernel {
    ($name:ident) => {
        /*
        
          ::native::vulkan::api::Shader::Descriptor{ 
            name##_spv,                                  
            name##_spv_len,                              
          }
        */
    }
}

macro_rules! vk_check {
    ($function:ident) => {
        /*
        
          do {                                                      
            const VkResult result = (function);                     
            TORCH_CHECK(VK_SUCCESS == result, "VkResult:", result); 
          } while (false)
        */
    }
}

#[macro_export] macro_rules! vk_check_relaxed {
    ($function:ident) => {
        /*
        
          do {                                                      
            const VkResult result = (function);                     
            TORCH_CHECK(VK_SUCCESS <= result, "VkResult:", result); 
          } while (false)
        */
    }
}

#[macro_export] macro_rules! vk_deleter {
    ($Handle:ident) => {
        /*
        
            native::vulkan::api::destroy_##Handle
        */
    }
}

#[macro_export] macro_rules! vk_deleter_dispatchable_declare {
    ($Handle:ident) => {
        /*
        
            void destroy_##Handle(const Vk##Handle handle)
        */
    }
}

#[macro_export] macro_rules! vk_deleter_non_dispatchable_declare {
    ($Handle:ident) => {
        /*
        
          class destroy_##Handle final {                      
                                                       
            explicit destroy_##Handle(const VkDevice device); 
            void operator()(const Vk##Handle handle) const;   
                                                      
            VkDevice device_;                                 
          };
        */
    }
}

pub struct GPU {
    adapter: *const Adapter,
    device:  VkDevice,
    queue:   VkQueue,
}

vk_deleter_dispatchable_declare!{Instance}
vk_deleter_dispatchable_declare!{Device}
vk_deleter_non_dispatchable_declare!{Semaphore}
vk_deleter_non_dispatchable_declare!{Fence}
vk_deleter_non_dispatchable_declare!{Buffer}
vk_deleter_non_dispatchable_declare!{Image}
vk_deleter_non_dispatchable_declare!{Event}
vk_deleter_non_dispatchable_declare!{BufferView}
vk_deleter_non_dispatchable_declare!{ImageView}
vk_deleter_non_dispatchable_declare!{ShaderModule}
vk_deleter_non_dispatchable_declare!{PipelineCache}
vk_deleter_non_dispatchable_declare!{PipelineLayout}
vk_deleter_non_dispatchable_declare!{Pipeline}
vk_deleter_non_dispatchable_declare!{DescriptorSetLayout}
vk_deleter_non_dispatchable_declare!{Sampler}
vk_deleter_non_dispatchable_declare!{DescriptorPool}
vk_deleter_non_dispatchable_declare!{CommandPool}

/**
  | Vulkan objects are referenced via handles.
  |
  | The spec defines Vulkan handles under two
  | categories: dispatchable and non-dispatchable.
  |
  | Dispatchable handles are required to be
  | strongly typed as a result of being pointers to
  | unique opaque types.
  |
  | Since dispatchable handles are pointers at the
  | heart, unique_ptr can be used to manage their
  | lifetime with a custom
  | deleter. Non-dispatchable handles on the other
  | hand, are not required to have strong types,
  | and even though they default to the same
  | implementation as dispatchable handles on some
  | platforms - making the use of unique_ptr
  | possible - they are only required by the spec
  | to weakly aliases 64-bit integers which is the
  | implementation some platforms default to.
  |
  | This makes the use of unique_ptr difficult
  | since semantically unique_ptrs store pointers
  | to their payload which is also what is passed
  | onto the custom deleters.
  */
pub struct Handle<Type,Deleter> {
    payload: Type,
    deleter: Deleter,
}

impl Drop for Handle {

    #[inline] fn drop(&mut self) {
        todo!();
        /*
            reset();
        */
    }
}

impl<Type, Deleter> Handle<Type, Deleter> {

    pub const NULL: Type = {};
    
    pub fn new(
        payload: Type,
        deleter: Deleter) -> Self {
    
        todo!();
        /*
        
        */
    }
    
    pub fn new(_0: Handle) -> Self {
    
        todo!();
        /*
        
        */
    }
    
    pub fn assign_from(&mut self, _0: Handle) -> &mut Handle {
        
        todo!();
        /*
        
        */
    }
    
    pub fn operator_bool(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn get(&self) -> Type {
        
        todo!();
        /*
        
        */
    }
    
    pub fn release(&mut self) -> Type {
        
        todo!();
        /*
        
        */
    }
    
    pub fn reset(&mut self, payload: Type)  {
        let payload: Type = payload.unwrap_or(kNull);

        todo!();
        /*
        
        */
    }
    
    pub fn new<Type, Deleter>(
        payload: Type,
        deleter: Deleter) -> Self {
    
        todo!();
        /*


            : payload_(payload),
        deleter_(move(deleter))
        */
    }
    
    pub fn new<Type, Deleter>(handle: Handle) -> Self {
    
        todo!();
        /*


            : payload_(handle.release()),
        deleter_(move(handle.deleter_))
        */
    }
    
    #[inline] pub fn assign_from(&mut self, handle: Handle) -> &mut Handle<Type,Deleter> {
    
        todo!();
        /*
            reset(handle.release());
      deleter_ = move(handle.deleter_);
      return *this;
        */
    }
    
    #[inline] pub fn operator_bool(&self) -> bool {
        
        todo!();
        /*
            return get();
        */
    }
    
    #[inline] pub fn get(&self) -> Type {
        
        todo!();
        /*
            return payload_;
        */
    }
    
    #[inline] pub fn release(&mut self) -> Type {
        
        todo!();
        /*
            const Type payload = payload_;
      payload_ = kNull;

      return payload;
        */
    }
    
    #[inline] pub fn reset(&mut self, payload: Type)  {
        
        todo!();
        /*
            using swap;
      swap(payload_, payload);

      if (kNull != payload) {
        deleter_(payload);
      }
        */
    }
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/api/Common.cpp]

#[macro_export] macro_rules! vk_deleter_dispatchable_define {
    ($Handle:ident) => {
        /*
        
          VK_DELETER_DISPATCHABLE_DECLARE(Handle) {     
            if (C10_LIKELY(VK_NULL_HANDLE != handle)) { 
              vkDestroy##Handle(handle, nullptr);       
            }                                           
          }
        */
    }
}

#[macro_export] macro_rules! vk_deleter_non_dispatchable_define {
    ($Handle:ident) => {
        /*
        
          destroy_##Handle::destroy_##Handle(const VkDevice device)           
            : device_(device) {                                               
          }                                                                   
                                                                              
          void destroy_##Handle::operator()(const Vk##Handle handle) const {  
            if (C10_LIKELY(VK_NULL_HANDLE != handle)) {                       
              vkDestroy##Handle(device_, handle, nullptr);                    
            }                                                                 
          }
        */
    }
}

vk_deleter_dispatchable_define!{Instance}
vk_deleter_dispatchable_define!{Device}
vk_deleter_non_dispatchable_define!{Semaphore}
vk_deleter_non_dispatchable_define!{Fence}
vk_deleter_non_dispatchable_define!{Buffer}
vk_deleter_non_dispatchable_define!{Image}
vk_deleter_non_dispatchable_define!{Event}
vk_deleter_non_dispatchable_define!{BufferView}
vk_deleter_non_dispatchable_define!{ImageView}
vk_deleter_non_dispatchable_define!{ShaderModule}
vk_deleter_non_dispatchable_define!{PipelineCache}
vk_deleter_non_dispatchable_define!{PipelineLayout}
vk_deleter_non_dispatchable_define!{Pipeline}
vk_deleter_non_dispatchable_define!{DescriptorSetLayout}
vk_deleter_non_dispatchable_define!{Sampler}
vk_deleter_non_dispatchable_define!{DescriptorPool}
vk_deleter_non_dispatchable_define!{CommandPool}
