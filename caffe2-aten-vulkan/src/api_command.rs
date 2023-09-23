crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/api/Command.h]


pub struct CommandBufferBound {
    pipeline:       PipelineObject,
    descriptor_set: VkDescriptorSet,
}

impl CommandBufferBound {
    
    pub fn reset(&mut self)  {
        
        todo!();
        /*
        
        */
    }
}

pub struct CommandBufferBarrierStage {
    src: VkPipelineStageFlags,
    dst: VkPipelineStageFlags,
}

impl CommandBufferBarrierStage {
    
    pub fn operator_bool(&self)  {
        
        todo!();
        /*
        
        */
    }
}

pub struct CommandBufferBarrier {
    stage:   CommandBufferBarrierStage,
    buffers: SmallVec<[ResourceBufferBarrier;4]>,
    images:  SmallVec<[ResourceImageBarrier;4]>,
}

impl CommandBufferBarrier {
    
    pub fn reset(&mut self)  {
        
        todo!();
        /*
        
        */
    }
}

pub struct CommandBuffer {
    command_buffer: VkCommandBuffer,
    bound:          CommandBufferBound,
    barriers:       CommandBufferBarrier,
}

impl CommandBuffer {
    
    pub fn new(command_buffer: VkCommandBuffer) -> Self {
        let command_buffer: VkCommandBuffer = command_buffer.unwrap_or(VK_NULL_HANDLE);
        todo!();
        /*


        
        */
    }
    
    pub fn new(_0: Buffer) -> Self {
    
        todo!();
        /*


        
        */
    }
    
    pub fn assign_from(&mut self, _0: Buffer) -> &mut Buffer {
        
        todo!();
        /*
        
        */
    }
    
    pub fn operator_bool(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn handle(&self) -> VkCommandBuffer {
        
        todo!();
        /*
        
        */
    }
    
    pub fn begin(&mut self)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn end(&mut self)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn barrier(&mut self, barrier: &PipelineBarrier)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn bind(&mut self, pipeline: &PipelineObject)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn bind(&mut self, set: &DescriptorSet)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn copy_(&mut self, 
        source:      ResourceBufferObject,
        destination: ResourceBufferObject)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn dispatch(&mut self, global_work_group: &ShaderWorkGroup)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn barrier(&mut self)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn invalidate(&mut self)  {
        
        todo!();
        /*
        
        */
    }
}

pub mod command_pool_configuration {
    pub const QUANTUM: u32 = 4;
    pub const RESERVE: u32 = 16;
    pub const SUBMIT:  u32 = 3;
}

pub struct CommandPoolBuffer {
    pool:   Vec<VkCommandBuffer>,
    in_use: usize,
}

pub struct CommandPoolStream {
    buffer:  Buffer,
    counter: u32,
}

pub struct CommandPool {
    device:       VkDevice,
    command_pool: Handle<VkCommandPool, VK_DELETER(CommandPool)>,
    buffer:       CommandPoolBuffer,
    stream:       CommandPoolStream,
}

impl CommandPool {
    
    pub fn new(gpu: &Gpu) -> Self {
    
        todo!();
        /*

        */
    }
    
    pub fn new(_0: Pool) -> Self {
    
        todo!();
        /*
        
        */
    }
    
    pub fn assign_from(&mut self, _0: Pool) -> &mut Pool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn allocate(&mut self) -> Buffer {
        
        todo!();
        /*
        
        */
    }
    
    pub fn stream(&mut self) -> &mut Buffer {
        
        todo!();
        /*
        
        */
    }
    
    pub fn purge(&mut self)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn submit(&mut self, 
        queue:   VkQueue,
        buffers: &[Buffer],
        fence:   ResourceFence)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn invalidate(&mut self)  {
        
        todo!();
        /*
        
        */
    }
}

pub struct Command {

    /** [thread_count] */
    pool: CommandPool,
}

impl Command {

    pub fn new(gpu: &Gpu) -> Self {
    
        todo!();
        /*


            : pool(gpu)
        */
    }
}

impl CommandBuffer {
    
    #[inline] pub fn operator_bool(&self) -> bool {
        
        todo!();
        /*
            return VK_NULL_HANDLE != command_buffer_;
        */
    }
    
    #[inline] pub fn handle(&self) -> VkCommandBuffer {
        
        todo!();
        /*
            return command_buffer_;
        */
    }
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/api/Command.cpp]

pub fn create_command_pool(
    device:             VkDevice,
    queue_family_index: u32) -> VkCommandPool {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          device,
          "Invalid Vulkan device!");

      const VkCommandPoolCreateInfo command_pool_create_info{
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        nullptr,
        VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        queue_family_index,
      };

      VkCommandPool command_pool{};
      VK_CHECK(vkCreateCommandPool(
          device,
          &command_pool_create_info,
          nullptr,
          &command_pool));

      TORCH_CHECK(
          command_pool,
          "Invalid Vulkan command pool!");

      return command_pool;
        */
}

pub fn allocate_command_buffers(
        device:          VkDevice,
        command_pool:    VkCommandPool,
        command_buffers: *mut VkCommandBuffer,
        count:           u32)  {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          device,
          "Invalid Vulkan device!");

      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          command_pool,
          "Invalid Vulkan command pool!");

      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          command_buffers && (count > 0u),
          "Invalid usage!");

      const VkCommandBufferAllocateInfo command_buffer_allocate_info{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        nullptr,
        command_pool,
        VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        count,
      };

      VK_CHECK(vkAllocateCommandBuffers(
          device,
          &command_buffer_allocate_info,
          command_buffers));
        */
}

impl CommandBuffer {
    
    pub fn new(command_buffer: VkCommandBuffer) -> Self {
    
        todo!();
        /*


            : command_buffer_(command_buffer)
        */
    }
    
    pub fn new(buffer: Buffer) -> Self {
    
        todo!();
        /*


            : command_buffer_(move(buffer.command_buffer_)),
        bound_(move(buffer.bound_)),
        barriers_(move(buffer.barriers_)) 
      buffer.invalidate();
        */
    }
    
    pub fn assign_from(&mut self, buffer: Buffer) -> &mut CommandBuffer {
        
        todo!();
        /*
            if (&buffer != this) {
        command_buffer_ = move(buffer.command_buffer_);
        bound_ = move(buffer.bound_);
        barriers_ = move(buffer.barriers_);

        buffer.invalidate();
      };

      return *this;
        */
    }
    
    pub fn begin(&mut self)  {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          command_buffer_,
          "This command buffer is in an invalid state! "
          "Potential reason: This command buffer is moved from.");

      const VkCommandBufferBeginInfo command_buffer_begin_info{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        nullptr,
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        nullptr,
      };

      VK_CHECK(vkBeginCommandBuffer(
          command_buffer_,
          &command_buffer_begin_info));

      // Reset
      bound_.reset();
      barriers_.reset();
        */
    }
    
    pub fn end(&mut self)  {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          command_buffer_,
          "This command buffer is in an invalid state! "
          "Potential reason: This command buffer is moved from.");

      VK_CHECK(vkEndCommandBuffer(command_buffer_));
        */
    }
    
    pub fn barrier(&mut self, barrier: &PipelineBarrier)  {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          command_buffer_,
          "This command buffer is in an invalid state! "
          "Potential reason: This command buffer is moved from.");

      barriers_.stage.src |= barrier.stage.src;
      barriers_.stage.dst |= barrier.stage.dst;

      barriers_.buffers.insert(
          barriers_.buffers.end(),
          barrier.buffers.begin(),
          barrier.buffers.end());

      barriers_.images.insert(
          barriers_.images.end(),
          barrier.images.begin(),
          barrier.images.end());
        */
    }
    
    pub fn bind(&mut self, pipeline: &PipelineObject)  {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          command_buffer_,
          "This command buffer is in an invalid state! "
          "Potential reason: This command buffer is moved from.");

      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          pipeline,
          "Invalid Vulkan pipeline!");

      if (pipeline.handle != bound_.pipeline.handle) {
        vkCmdBindPipeline(
            command_buffer_,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline.handle);

        bound_.pipeline = pipeline;
      }
        */
    }
    
    pub fn bind(&mut self, set: &DescriptorSet)  {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          command_buffer_,
          "This command buffer is in an invalid state! "
          "Potential reason: This command buffer is moved from.");

      const VkDescriptorSet descriptor_set = set.handle();

      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          descriptor_set,
          "Invalid Vulkan descriptor set!");

      if (descriptor_set != bound_.descriptor_set) {
        vkCmdBindDescriptorSets(
            command_buffer_,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            bound_.pipeline.layout,
            0u,
            1u,
            &descriptor_set,
            0u,
            nullptr);

        bound_.descriptor_set = descriptor_set;
      }
        */
    }
    
    pub fn copy_(&mut self, 
        source:      ResourceBufferObject,
        destination: ResourceBufferObject)  {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          command_buffer_,
          "This command buffer is in an invalid state! "
          "Potential reason: This command buffer is moved from.");

      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          source,
          "Invalid Vulkan source buffer!");

      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          destination,
          "Invalid Vulkan destination buffer!");

      barrier();

      const VkBufferCopy buffer_copy{
        0u,
        0u,
        min(source.range, destination.range),
      };

      vkCmdCopyBuffer(
          command_buffer_,
          source.handle,
          destination.handle,
          1u,
          &buffer_copy);
        */
    }
    
    pub fn dispatch(&mut self, global_work_group: &ShaderWorkGroup)  {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          command_buffer_,
          "This command buffer is in an invalid state! "
          "Potential reason: This command buffer is moved from.");

      barrier();

      vkCmdDispatch(
          command_buffer_,
          utils::div_up(
              global_work_group.data[0u],
              bound_.pipeline.local_work_group.data[0u]),
          utils::div_up(
              global_work_group.data[1u],
              bound_.pipeline.local_work_group.data[1u]),
          utils::div_up(
              global_work_group.data[2u],
              bound_.pipeline.local_work_group.data[2u]));
        */
    }
    
    pub fn barrier(&mut self)  {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          command_buffer_,
          "This command buffer is in an invalid state! "
          "Potential reason: This command buffer is moved from.");

      if (barriers_.stage) {
        SmallVector<VkBufferMemoryBarrier, 4u> buffer_memory_barriers;

        for (const Resource::Buffer::Barrier& barrier : barriers_.buffers) {
          buffer_memory_barriers.push_back({
                VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                nullptr,
                barrier.memory.src,
                barrier.memory.dst,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                barrier.object.handle,
                barrier.object.offset,
                barrier.object.range,
              });
        }

        SmallVector<VkImageMemoryBarrier, 4u> image_memory_barriers;

        for (const Resource::Image::Barrier& barrier : barriers_.images) {
          image_memory_barriers.push_back({
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                barrier.memory.src,
                barrier.memory.dst,
                barrier.layout.src,
                barrier.layout.dst,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                barrier.object.handle,
                {
                  VK_IMAGE_ASPECT_COLOR_BIT,
                  0u,
                  VK_REMAINING_MIP_LEVELS,
                  0u,
                  VK_REMAINING_ARRAY_LAYERS,
                },
              });
        }

        vkCmdPipelineBarrier(
            command_buffer_,
            barriers_.stage.src,
            barriers_.stage.dst,
            0u,
            0u,
            nullptr,
            buffer_memory_barriers.size(),
            buffer_memory_barriers.data(),
            image_memory_barriers.size(),
            image_memory_barriers.data());
      }

      // Reset
      barriers_.reset();
        */
    }
    
    pub fn invalidate(&mut self)  {
        
        todo!();
        /*
            command_buffer_ = VK_NULL_HANDLE;
        */
    }
}

impl CommandBufferBound {
    
    #[inline] pub fn reset(&mut self)  {
        
        todo!();
        /*
            pipeline = {};
      descriptor_set = VK_NULL_HANDLE;
        */
    }
}

impl CommandBufferBarrierStage {
    
    #[inline] pub fn operator_bool(&self) -> bool {
        
        todo!();
        /*
            return (0u != src) || (0u != dst);
        */
    }
}

impl CommandBufferBarrier {
    
    #[inline] pub fn reset(&mut self)  {
        
        todo!();
        /*
            stage = {};
      buffers.clear();
      images.clear();
        */
    }
}

impl CommandPool {
    
    pub fn new(gpu: &Gpu) -> Self {
    
        todo!();
        /*


            : device_(gpu.device),
        command_pool_(
            create_command_pool(gpu.device, gpu.adapter->compute_queue_family_index),
            VK_DELETER(CommandPool)(device_)),
        buffer_{} 
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          device_,
          "Invalid Vulkan device!");

      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          command_pool_,
          "Invalid Vulkan command pool!");

      buffer_.pool.reserve(Configuration::kReserve);
        */
    }
    
    pub fn new(pool: Pool) -> Self {
    
        todo!();
        /*


            : device_(move(pool.device_)),
        command_pool_(move(pool.command_pool_)),
        buffer_(move(pool.buffer_)),
        stream_(move(pool.stream_)) 
      pool.invalidate();
        */
    }
    
    pub fn assign_from(&mut self, pool: Pool) -> &mut CommandPool {
        
        todo!();
        /*
            if (&pool != this) {
        device_ = move(pool.device_);
        command_pool_ = move(pool.command_pool_);
        buffer_ = move(pool.buffer_);
        stream_ = move(pool.stream_);

        pool.invalidate();
      };

      return *this;
        */
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        todo!();
        /*
            try {
        if (device_ && command_pool_) {
          purge();
        }
      }
      catch (const exception& e) {
        TORCH_WARN(
            "Vulkan: Command pool destructor raised an exception! Error: ",
            e.what());
      }
      catch (...) {
        TORCH_WARN(
            "Vulkan: Command pool destructor raised an exception! "
            "Error: Unknown");
      }
        */
    }
}

impl CommandPool {
    
    pub fn allocate(&mut self) -> CommandBuffer {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          device_ && command_pool_,
          "This command pool is in an invalid state! "
          "Potential reason: This command pool is moved from.");

      if (buffer_.pool.size() == buffer_.in_use) {
        buffer_.pool.resize(
            buffer_.pool.size() +
            Configuration::kQuantum);

        allocate_command_buffers(
            device_,
            command_pool_.get(),
            buffer_.pool.data() + buffer_.in_use,
            Configuration::kQuantum);
      }

      return Buffer(buffer_.pool[buffer_.in_use++]);
        */
    }
    
    pub fn stream(&mut self) -> &mut CommandBuffer {
        
        todo!();
        /*
            if (!stream_.buffer) {
        stream_.buffer = allocate();
        stream_.buffer.begin();
        stream_.counter = 0u;
      }

      return stream_.buffer;
        */
    }
    
    pub fn purge(&mut self)  {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          device_ && command_pool_,
          "This command pool is in an invalid state! "
          "Potential reason: This command pool is moved from.");

      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          !stream_.buffer,
          "Pending command buffer detected.  Make sure all command buffers are "
          "submitted to the queue for execution prior to reclaiming pool memory.");

      buffer_.in_use = 0u;
      VK_CHECK(vkResetCommandPool(device_, command_pool_.get(), 0u));
        */
    }
    
    pub fn submit(&mut self, 
        queue:   VkQueue,
        buffers: &[Buffer],
        fence:   ResourceFence)  {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          device_ && command_pool_,
          "This command pool is in an invalid state! "
          "Potential reason: This command pool is moved from.");

      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          queue,
          "Invalid Vulkan queue!");

      SmallVector<VkCommandBuffer, Configuration::kReserve> command_buffers;
      command_buffers.reserve(buffers.size());

      for (const Buffer& buffer : buffers) {
        VkCommandBuffer command_buffer = buffer.handle();

        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
            command_buffer,
            "Invalid Vulkan command buffer!");

        // Are we submitting our one and only command stream, or a regular command
        // buffer whose scope is manually maintained by the user?  Automatically
        // maintain state and submission rate if the former.

        if (stream_.buffer.handle() == command_buffer) {
          // Hand the stream off to the driver if:
          // - The user has implictly signaled interest in the results via a fence.
          // - We are over the submission cutoff.  We don't want to starve the GPU.

          if (fence || (stream_.counter++ > Configuration::kSubmit)) {
            stream_.buffer.end();
            stream_.buffer.invalidate();
          }
          // Skip - Accumulate more calls prior to submission.
          else {
            command_buffer = VK_NULL_HANDLE;
          }
        }

        if (command_buffer) {
          command_buffers.push_back(command_buffer);
        }
      }

      if (!command_buffers.empty()) {
        const VkSubmitInfo submit_info{
          VK_STRUCTURE_TYPE_SUBMIT_INFO,
          nullptr,
          0u,
          nullptr,
          nullptr,
          utils::safe_downcast<u32>(command_buffers.size()),
          command_buffers.data(),
          0u,
          nullptr,
        };

        VK_CHECK(vkQueueSubmit(queue, 1u, &submit_info, fence.handle()));
      }
        */
    }
    
    pub fn invalidate(&mut self)  {
        
        todo!();
        /*
            device_ = VK_NULL_HANDLE;
      command_pool_.reset();
        */
    }
}
