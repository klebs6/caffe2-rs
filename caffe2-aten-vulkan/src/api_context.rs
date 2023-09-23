crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/api/Context.h]

/**
  | Vulkan Context holds onto all relevant Vulkan
  | state as it pertains to our use of Vulkan in
  | PyTorch.
  |
  | A Context is associated with one, and only one,
  | Adapter as a precursor to multi-GPU support.
  |
  | All Vulkan tensors in PyTorch are associated
  | with a Context to make tensor <-> device
  | affinity explicit. The context is currently
  | a global object, but technically it does not
  | need to be if we were to make it explicit to
  | the user.
  |
  */
pub struct Context {

    /**
      | Construction and destruction order
      | matters. Do not move members around.
      |
      */
    adapter:    Adapter,
    device:     Handle<VkDevice, decltype(&VK_DELETER(Device))>,
    queue:      VkQueue,
    command:    Command,
    shader:     Shader,
    pipeline:   Pipeline,
    descriptor: Descriptor,
    resource:   Resource,
}

impl Context {
    
    #[inline] pub fn gpu(&mut self) -> Gpu {
        
        todo!();
        /*
      // A GPU is simply a (physical device, logical device, device queue) trio.
      return {
        &adapter_,
        device(),
        queue(),
      };
        */
    }
    
    #[inline] pub fn command(&mut self) -> &mut Command {
        
        todo!();
        /*
            return command_;
        */
    }
    
    #[inline] pub fn shader(&mut self) -> &mut Shader {
        
        todo!();
        /*
            return shader_;
        */
    }
    
    #[inline] pub fn pipeline(&mut self) -> &mut Pipeline {
        
        todo!();
        /*
            return pipeline_;
        */
    }
    
    #[inline] pub fn descriptor(&mut self) -> &mut Descriptor {
        
        todo!();
        /*
            return descriptor_;
        */
    }
    
    #[inline] pub fn resource(&mut self) -> &mut Resource {
        
        todo!();
        /*
            return resource_;
        */
    }
    
    #[inline] pub fn device(&mut self) -> VkDevice {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(device_);
      return device_.get();
        */
    }
    
    #[inline] pub fn queue(&mut self) -> VkQueue {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(queue_);
      return queue_;
        */
    }
}

#[inline] pub fn bind<const Indices: usize, Arguments>(
        descriptor_set: &mut DescriptorSet,
        _1:             IndexSequence<Indices>,
        arguments:      Arguments)  {

    todo!();
        /*
            const int _[]{
        0,
        (descriptor_set.bind(Indices, forward<Arguments>(arguments)), 0)...,
      };
        */
}

impl Context {
    
    #[inline] pub fn dispatch<Arguments>(&mut self, 
        command_buffer:          &mut CommandBuffer,
        shader_layout_signature: &ShaderLayoutSignature,
        shader_descriptor:       &ShaderDescriptor,
        global_work_group:       &ShaderWorkGroup,
        local_work_group_size:   &ShaderWorkGroup,
        arguments:               Arguments)  {
    
        todo!();
        /*
            // Forward declaration
      Descriptor::Set dispatch_prologue(
          Command::Buffer&,
          const Shader::Layout::Signature&,
          const Shader::Descriptor&,
          const Shader::WorkGroup&);

      // Factor out template parameter independent code to minimize code bloat.
      Descriptor::Set descriptor_set = dispatch_prologue(
          command_buffer,
          shader_layout_signature,
          shader_descriptor,
          local_work_group_size);

      bind(
          descriptor_set,
          index_sequence_for<Arguments...>{},
          forward<Arguments>(arguments)...);

      // Forward declaration
      void dispatch_epilogue(
          Command::Buffer&,
          const Descriptor::Set&,
          const Shader::WorkGroup&);

      // Factor out template parameter independent code to minimize code bloat.
      dispatch_epilogue(
          command_buffer,
          descriptor_set,
          global_work_group);
        */
    }
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/api/Context.cpp]

pub fn create_device(
    physical_device:            VkPhysicalDevice,
    compute_queue_family_index: u32) -> VkDevice {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          physical_device,
          "Invalid Vulkan physical device!");

      const float queue_priorities = 1.0f;
      const VkDeviceQueueCreateInfo device_queue_create_info{
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        nullptr,
        0u,
        compute_queue_family_index,
        1u,
        &queue_priorities,
      };

      u32 device_extension_properties_count = 0;
      VK_CHECK(vkEnumerateDeviceExtensionProperties(
          physical_device,
          nullptr,
          &device_extension_properties_count,
          nullptr));

      vector<VkExtensionProperties> device_extension_properties(
          device_extension_properties_count);

      VK_CHECK(vkEnumerateDeviceExtensionProperties(
          physical_device,
          nullptr,
          &device_extension_properties_count,
          device_extension_properties.data()));

      constexpr const char* const requested_device_extensions[]{
      #ifdef VK_KHR_portability_subset
        // https://vulkan.lunarg.com/doc/view/1.2.162.0/mac/1.2-extensions/vkspec.html#VUID-VkDeviceCreateInfo-pProperties-04451
        VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME,
      #endif
      };

      vector<const char*> enabled_device_extensions;

      for (const auto& requested_device_extension : requested_device_extensions) {
        for (const auto& extension : device_extension_properties) {
          if (strcmp(requested_device_extension, extension.extensionName) == 0) {
            enabled_device_extensions.push_back(requested_device_extension);
            break;
          }
        }
      }

      const VkDeviceCreateInfo device_create_info{
        VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        nullptr,
        0u,
        1u,
        &device_queue_create_info,
        0u,
        nullptr,
        static_cast<u32>(enabled_device_extensions.size()),
        enabled_device_extensions.data(),
        nullptr,
      };

      VkDevice device{};
      VK_CHECK(vkCreateDevice(physical_device, &device_create_info, nullptr, &device));
      TORCH_CHECK(device, "Invalid Vulkan device!");

      return device;
        */
}

pub fn acquire_queue(
        device:                     VkDevice,
        compute_queue_family_index: u32) -> VkQueue {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          device,
          "Invalid Vulkan device!");

      VkQueue queue{};
      vkGetDeviceQueue(device, compute_queue_family_index, 0, &queue);
      TORCH_CHECK(queue, "Invalid Vulkan queue!");

      return queue;
        */
}

impl Drop for Context {

    fn drop(&mut self) {
        todo!();
        /*
            try {
        flush();
      }
      catch (const exception& e) {
        TORCH_WARN(
            "Vulkan: Context destructor raised an exception! Error: ",
            e.what());
      }
      catch (...) {
        TORCH_WARN(
            "Vulkan: Context destructor raised an exception! "
            "Error: Unknown");
      }
        */
    }
}

impl Context {
    
    pub fn new(adapter: &Adapter) -> Self {
    
        todo!();
        /*
            : adapter_(adapter),
          device_(
              create_device(
                  adapter.handle,
                  adapter.compute_queue_family_index),
              &VK_DELETER(Device)),
          queue_(acquire_queue(device(), adapter.compute_queue_family_index)),
          command_(gpu()),
          shader_(gpu()),
          pipeline_(gpu()),
          descriptor_(gpu()),
          resource_(gpu()) 
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          device_,
          "Invalid Vulkan device!");
        */
    }
    
    /**
      | This function is expensive and its use
      | consequential for performance. Only use this
      | function for debugging or as a short term
      | hack on way to a more performant solution.
      */
    pub fn flush(&mut self)  {
        
        todo!();
        /*
            VK_CHECK(vkQueueWaitIdle(queue()));

      resource().pool.purge();
      descriptor().pool.purge();
      command().pool.purge();
        */
    }
}

pub fn available() -> bool {
    
    todo!();
        /*
            return context();
        */
}

pub fn context() -> *mut Context {
    
    todo!();
        /*
            static const unique_ptr<Context> context([]() -> Context* {
        try {
          const Adapter adapter = runtime()->select([](const Adapter& adapter) {
            // Select the first adapter.
            return true;
          });

          return new Context(adapter);
        }
        catch (const exception& e) {
          TORCH_WARN("Vulkan: Failed to initialize context! Error: ", e.what());
        }
        catch (...) {
          TORCH_WARN("Vulkan: Failed to initialize context! Error: Unknown");
        }

        return nullptr;
      }());

      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          context,
          "Invalid Vulkan context!");

      return context.get();
        */
}

pub struct VulkanImpl {
    base: VulkanImplInterface,
}

impl VulkanImpl {
    
    pub fn is_vulkan_available(&self) -> bool {
        
        todo!();
        /*
            return available();
        */
    }
    
    pub fn vulkan_copy<'a>(&self, 
        self_: &mut Tensor,
        src:   &Tensor) -> &'a mut Tensor {
        
        todo!();
        /*
            return vulkan::ops::copy_(self, src);
        */
    }
}

lazy_static!{
    /*
    static VulkanImplRegistrar g_vulkan_impl(new VulkanImpl());
    */
}

pub fn dispatch_prologue(
        command_buffer:          &mut CommandBuffer,
        shader_layout_signature: &ShaderLayoutSignature,
        shader_descriptor:       &ShaderDescriptor,
        local_work_group_size:   &ShaderWorkGroup) -> DescriptorSet {
    
    todo!();
        /*
            Context* const context = api::context();
      const GPU gpu = context->gpu();
      Descriptor& descriptor = context->descriptor();
      Pipeline& pipeline = context->pipeline();
      Shader& shader = context->shader();

      const Shader::Layout::Object shader_layout =
          shader.layout.cache.retrieve({
            shader_layout_signature,
          });

      command_buffer.bind(
          pipeline.cache.retrieve({
            pipeline.layout.cache.retrieve({
              shader_layout.handle,
            }),
            shader.cache.retrieve(shader_descriptor),
            local_work_group_size,
          }));

      return descriptor.pool.allocate(shader_layout);
        */
}

pub fn dispatch_epilogue(
        command_buffer:    &mut CommandBuffer,
        descriptor_set:    &DescriptorSet,
        global_work_group: &ShaderWorkGroup)  {
    
    todo!();
        /*
            command_buffer.bind(descriptor_set);
      command_buffer.dispatch(global_work_group);
        */
}
