crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/api/Descriptor.h]
pub union DescriptorSetItemInfo {
    buffer: VkDescriptorBufferInfo,
    image:  VkDescriptorImageInfo,
}

pub struct DescriptorSetItem {
    binding: u32,
    ty:      VkDescriptorType,
    info:    DescriptorSetItemInfo,
}

pub struct DescriptorSetBindings {
    items: SmallVector<Item,6>,
    dirty: RefCell<bool>,
}

pub struct DescriptorSet {
    device:                  VkDevice,
    descriptor_set:          VkDescriptorSet,
    shader_layout_signature: ShaderLayoutSignature,
    bindings:                DescriptorSetBindings,
}

impl DescriptorSet {
    
    pub fn new(
        device:                  VkDevice,
        descriptor_set:          VkDescriptorSet,
        shader_layout_signature: &ShaderLayoutSignature) -> Self {
    
        todo!();
        /*
        
        */
    }
    
    pub fn bind(&mut self, 
        binding: u32,
        buffer:  &ResourceBufferObject) -> &mut DescriptorSet {
        
        todo!();
        /*
        
        */
    }
    
    pub fn bind(&mut self, 
        binding: u32,
        image:   &ResourceImageObject) -> &mut DescriptorSet {
        
        todo!();
        /*
        
        */
    }
    
    pub fn handle(&self) -> VkDescriptorSet {
        
        todo!();
        /*
        
        */
    }
    
    pub fn invalidate(&mut self)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn update(&mut self, item: &Item)  {
        
        todo!();
        /*
        
        */
    }
}

pub mod descriptor_pool_configuration  {

    pub const QUANTUM: u32 = 16;
    pub const RESERVE: u32 = 64;
}

pub struct DescriptorPoolSetLayout {
    pool:   Vec<VkDescriptorSet>,
    in_use: usize,
}

pub struct DescriptorPoolSet {
    layouts: FlatHashMap<VkDescriptorSetLayout,Layout>,
}

pub struct DescriptorPool {
    device:          VkDevice,
    descriptor_pool: Handle<VkDescriptorPool, VK_DELETER(DescriptorPool)>,
    set:             DescriptorPoolSet,
}

impl DescriptorPool {
    
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
    
    pub fn allocate(&mut self, shader_layout: &ShaderLayoutObject) -> DescriptorSet {
        
        todo!();
        /*
        
        */
    }
    
    pub fn purge(&mut self)  {
        
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

/**
  | This struct defines caches of descriptor pools,
  | and descriptor sets allocated from those pools,
  | intended to minimize redundant object
  | reconstructions or accelerate unavoidable
  | memory allocations, both at the cost of extra
  | memory consumption.
  |
  | A descriptor set is logically an array of
  | descriptors, each of which references
  | a resource (i.e. buffers and images), in turn
  | telling the core executing the shader, where in
  | GPU, or GPU-accessible system, memory the said
  | resource resides.
  |
  | To accelerate creation of the descriptor sets,
  | modern graphics APIs allocate them from a pool,
  | more elaborately referred to as descriptor
  | pools, which do need to be purged frequently
  | _after_ none of the descriptors the pools
  | contain is in use by the GPU.  Care must be
  | taken that descriptors are not freed while they
  | are in use by the pipeline, which considering
  | the asynchronous nature of CPU-GPU
  | interactions, can be anytime after the command
  | is issued until it is fully executed by the
  | GPU.
  |
  | As you can imagine, it is possible to have
  | multiple descriptor pools, each of which is
  | configured to house different types of
  | descriptor sets with different allocation
  | strategies. These descriptor pools themselves
  | are fairly stable objects in that they
  | theymself should not be created and destroyed
  | frequently. That is the reason why we store
  | them in a cache, which according to our usage
  | of the term 'cache' in this implementatoin, is
  | reserved for objects that are created
  | infrequently and stabilize to a manageable
  | number quickly over the lifetime of the
  | program.
  |
  | Descriptor sets though, on the other hand, are
  | allocated from pools which indeed does mean
  | that the pools must be purged on a regular
  | basis or else they will run out of free items.
  | Again, this is in line with our usage of the
  | term 'pool' in this implementation which we use
  | to refer to a container of objects that is
  | allocated out of and is required to be
  | frequently purged.
  |
  | It is important to point out that for
  | performance reasons, we intentionally do not
  | free the descriptor sets individually, and
  | instead opt to purge the pool in its totality,
  | even though Vulkan supports the former usage
  | pattern as well.  This behavior is by design.
  |
  */
pub struct Descriptor {

    /**
      | [thread_count]
      |
      */
    pool: DescriptorPool,
}

impl Descriptor {
    
    pub fn new(gpu: &Gpu) -> Self {
    
        todo!();
        /*
            : pool(gpu)
        */
    }
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/api/Descriptor.cpp]

pub fn create_descriptor_pool(device: VkDevice) -> VkDescriptorPool {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          device,
          "Invalid Vulkan device!");

      const struct {
        u32 capacity;
        SmallVector<VkDescriptorPoolSize, 4u> sizes;
      } descriptor {
        1024u,
        {
          /*
            Buffers
          */

          {
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            1024u,
          },
          {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            1024u,
          },

          /*
            Images
          */

          {
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            1024u,
          },
          {
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            1024u,
          },
        },
      };

      const VkDescriptorPoolCreateInfo descriptor_pool_create_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        nullptr,
        0u,
        descriptor.capacity,
        static_cast<u32>(descriptor.sizes.size()),
        descriptor.sizes.data(),
      };

      VkDescriptorPool descriptor_pool{};
      VK_CHECK(vkCreateDescriptorPool(
          device,
          &descriptor_pool_create_info,
          nullptr,
          &descriptor_pool));

      TORCH_CHECK(
          descriptor_pool,
          "Invalid Vulkan descriptor pool!");

      return descriptor_pool;
        */
}

pub fn allocate_descriptor_sets(
    device:                VkDevice,
    descriptor_pool:       VkDescriptorPool,
    descriptor_set_layout: VkDescriptorSetLayout,
    descriptor_sets:       *mut VkDescriptorSet,
    count:                 u32)  {

    todo!();
    /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          device,
          "Invalid Vulkan device!");

      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          descriptor_pool,
          "Invalid Vulkan descriptor pool!");

      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          descriptor_set_layout,
          "Invalid Vulkan descriptor set layout!");

      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          descriptor_sets && (count > 0u),
          "Invalid usage!");

      vector<VkDescriptorSetLayout> descriptor_set_layouts(count);
      fill(
        descriptor_set_layouts.begin(),
        descriptor_set_layouts.end(),
        descriptor_set_layout
      );

      const VkDescriptorSetAllocateInfo descriptor_set_allocate_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        nullptr,
        descriptor_pool,
        utils::safe_downcast<u32>(descriptor_set_layouts.size()),
        descriptor_set_layouts.data(),
      };

      VK_CHECK(vkAllocateDescriptorSets(
          device,
          &descriptor_set_allocate_info,
          descriptor_sets));
        */
}

impl DescriptorSet {
    
    pub fn new(
        device:                  VkDevice,
        descriptor_set:          VkDescriptorSet,
        shader_layout_signature: &ShaderLayoutSignature) -> Self {
    
        todo!();
        /*


            : device_(device),
        descriptor_set_(descriptor_set),
        shader_layout_signature_(shader_layout_signature),
        bindings_{} 

      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          device_,
          "Invalid Vulkan device!");

      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          descriptor_set_,
          "Invalid Vulkan descriptor set!");
        */
    }
    
    pub fn new(set: DescriptorSet) -> Self {
    
        todo!();
        /*


            : device_(move(set.device_)),
        descriptor_set_(move(set.descriptor_set_)),
        shader_layout_signature_(move(set.shader_layout_signature_)),
        bindings_(move(set.bindings_)) 
      set.invalidate();
        */
    }
    
    pub fn assign_from(&mut self, set: DescriptorSet) -> &mut DescriptorSet {
        
        todo!();
        /*
            if (&set != this) {
        device_ = move(set.device_);
        descriptor_set_ = move(set.descriptor_set_);
        shader_layout_signature_ = move(set.shader_layout_signature_);
        bindings_ = move(set.bindings_);

        set.invalidate();
      };

      return *this;
        */
    }
    
    pub fn bind(&mut self, 
        binding: u32,
        buffer:  &ResourceBufferObject) -> &mut DescriptorSet {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          device_ && descriptor_set_,
          "This descriptor set is in an invalid state! "
          "Potential reason: This descriptor set is moved from.");

      update({
          binding,
          shader_layout_signature_[binding],
          {
            .buffer = {
              buffer.handle,
              buffer.offset,
              buffer.range,
            },
          },
        });

      return *this;
        */
    }
    
    pub fn bind(&mut self, 
        binding: u32,
        image:   &ResourceImageObject) -> &mut DescriptorSet {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          device_ && descriptor_set_,
          "This descriptor set is in an invalid state! "
          "Potential reason: This descriptor set is moved from.");

      update(Item{
          binding,
          shader_layout_signature_[binding],
          {
            .image = {
              image.sampler,
              image.view,
              [](const VkDescriptorType type, const VkImageLayout layout) {
                return (VK_DESCRIPTOR_TYPE_STORAGE_IMAGE == type) ?
                        VK_IMAGE_LAYOUT_GENERAL : layout;
              }(shader_layout_signature_[binding], image.layout),
            },
          },
        });

      return *this;
        */
    }
    
    pub fn handle(&self) -> VkDescriptorSet {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          device_ && descriptor_set_,
          "This descriptor set is in an invalid state! "
          "Potential reason: This descriptor set is moved from.");

      if (bindings_.dirty) {
        const auto is_buffer = [](const VkDescriptorType type) {
          switch (type) {
            case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
            case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
              return true;

            default:
              return false;
          }
        };

        const auto is_image = [](const VkDescriptorType type) {
          switch (type) {
            case VK_DESCRIPTOR_TYPE_SAMPLER:
            case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
            case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
            case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
            case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
              return true;

            default:
              return false;
          }
        };

        SmallVector<VkWriteDescriptorSet, 6u> write_descriptor_sets;

        for (const Item& item : bindings_.items) {
          VkWriteDescriptorSet write{
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            descriptor_set_,
            item.binding,
            0u,
            1u,
            item.type,
            nullptr,
            nullptr,
            nullptr,
          };

          if (is_buffer(item.type)) {
            write.pBufferInfo = &item.info.buffer;
          }
          else if (is_image(item.type)) {
            write.pImageInfo = &item.info.image;
          }

          write_descriptor_sets.emplace_back(write);
        }

        vkUpdateDescriptorSets(
            device_,
            write_descriptor_sets.size(),
            write_descriptor_sets.data(),
            0u,
            nullptr);

        // Reset
        bindings_.dirty = false;
      }

      return descriptor_set_;
        */
    }
    
    pub fn invalidate(&mut self)  {
        
        todo!();
        /*
            device_ = VK_NULL_HANDLE;
      descriptor_set_ = VK_NULL_HANDLE;
        */
    }
    
    pub fn update(&mut self, item: &Item)  {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          device_ && descriptor_set_,
          "This descriptor set is in an invalid state! "
          "Potential reason: This descriptor set is moved from.");

      const auto items_itr = find_if(
          bindings_.items.begin(),
          bindings_.items.end(),
          [binding = item.binding](const Item& other) {
            return other.binding == binding;
          });

      if (bindings_.items.end() == items_itr) {
         bindings_.items.emplace_back(item);
      }
      else {
        *items_itr = item;
      }

      bindings_.dirty = true;
        */
    }
}

impl Drop for DescriptorPool {

    fn drop(&mut self) {
        todo!();
        /*
            try {
        if (device_ && descriptor_pool_) {
          purge();
        }
      }
      catch (const exception& e) {
        TORCH_WARN(
            "Vulkan: Descriptor pool destructor raised an exception! Error: ",
            e.what());
      }
      catch (...) {
        TORCH_WARN(
            "Vulkan: Descriptor pool destructor raised an exception! "
            "Error: Unknown");
      }
        */
    }
}

impl DescriptorPool {
    
    pub fn new(gpu: &Gpu) -> Self {
    
        todo!();
        /*


            : device_(gpu.device),
        descriptor_pool_(
            create_descriptor_pool(gpu.device),
            VK_DELETER(DescriptorPool)(device_)) 
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          device_,
          "Invalid Vulkan device!");

      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          descriptor_pool_,
          "Invalid Vulkan descriptor pool!");
        */
    }
    
    pub fn new(pool: Pool) -> Self {
    
        todo!();
        /*


            : device_(move(pool.device_)),
        descriptor_pool_(move(pool.descriptor_pool_)),
        set_(move(pool.set_)) 
      pool.invalidate();
        */
    }
    
    pub fn assign_from(&mut self, pool: Pool) -> &mut DescriptorPool {
        
        todo!();
        /*
            if (&pool != this) {
        device_ = move(pool.device_);
        descriptor_pool_ = move(pool.descriptor_pool_);
        set_ = move(pool.set_);

        pool.invalidate();
      };

      return *this;
        */
    }
    
    pub fn allocate(&mut self, shader_layout: &ShaderLayoutObject) -> DescriptorSet {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          device_ && descriptor_pool_,
          "This descriptor pool is in an invalid state! "
          "Potential reason: This descriptor pool is moved from.");

      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          shader_layout,
          "Invalid Vulkan shader layout!");

      auto iterator = set_.layouts.find(shader_layout.handle);
      if (set_.layouts.cend() == iterator) {
        iterator = set_.layouts.insert({shader_layout.handle, {}}).first;
        iterator->second.pool.reserve(Configuration::kReserve);
      }

      auto& layout = iterator->second;

      if (layout.pool.size() == layout.in_use) {
        layout.pool.resize(
            layout.pool.size() +
            Configuration::kQuantum);

        allocate_descriptor_sets(
            device_,
            descriptor_pool_.get(),
            shader_layout.handle,
            layout.pool.data() + layout.in_use,
            Configuration::kQuantum);
      }

      return Set(
          device_,
          layout.pool[layout.in_use++],
          shader_layout.signature);
        */
    }
    
    pub fn purge(&mut self)  {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          device_ && descriptor_pool_,
          "This descriptor pool is in an invalid state! "
          "Potential reason: This descriptor pool is moved from.");

      VK_CHECK(vkResetDescriptorPool(device_, descriptor_pool_.get(), 0u));
      set_.layouts.clear();
        */
    }
    
    pub fn invalidate(&mut self)  {
        
        todo!();
        /*
            device_ = VK_NULL_HANDLE;
      descriptor_pool_.reset();
        */
    }
}
