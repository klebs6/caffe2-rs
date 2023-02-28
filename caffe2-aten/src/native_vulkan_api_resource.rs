crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/api/Resource.h]

pub struct ResourceMemoryDescriptor {
    usage:     VmaMemoryUsage,
    required:  VkMemoryPropertyFlags,
    preferred: VkMemoryPropertyFlags,
}

pub struct ResourceMemoryBarrier {
    src: VkAccessFlags,
    dst: VkAccessFlags,
}

bitflags!{
    pub struct ResourceMemoryAccessType: u8 {
        const None = 0;
        const Read = 1;
        const Write = 2;
    }
}

pub struct ResourceMemoryAccess {

}

pub mod resource_memory_access {

    use super::*;

    pub type Flags = u8;

    lazy_static!{
        /*

            template<typename Type, Flags access>
                using Pointer = add_pointer_t<
                conditional_t<
                0u != (access & Write),
                Type,
                add_const_t<Type>>>;
        */
    }
}

/*
// Intentionally disabed to ensure memory access is always properly
// encapsualted in a scoped map-unmap region.  Allowing below overloads
// to be invoked on a temporary would open the door to the possibility
// of accessing the underlying memory out of the expected scope making
// for seemingly ineffective memory writes and hard to hunt down bugs.

template<typename Type, typename Pointer>
    Handle<Pointer> map() const && = delete;

template<typename Type, Access::Flags kAccess, typename Pointer>
    Handle<Pointer> map() && = delete;
*/
pub struct ResourceMemory {
    allocator:  VmaAllocator,
    allocation: VmaAllocation,
}

pub mod resource_memory {

    use super::*;

    pub type Handle<Type> = Handle<Type,Scope>;

    lazy_static!{
        /*

            template<
                typename Type,
                typename Pointer = Access::Pointer<Type, Access::Read>>
                    Handle<Pointer> map() const &;

            template<
                typename Type,
                Access::Flags kAccess,
                typename Pointer = Access::Pointer<Type, kAccess>>
                    Handle<Pointer> map() &;
        */
    }
}

pub struct ResourceBufferDescriptorUsage {
    buffer: VkBufferUsageFlags,
    memory: MemoryDescriptor,
}

pub struct ResourceBufferDescriptor {
    size:  VkDeviceSize,
    usage: ResourceBufferDescriptorUsage,
}

pub struct ResourceBufferObject {
    handle: vk::Buffer,
    offset: VkDeviceSize,
    range:  VkDeviceSize,
}

impl ResourceBufferObject {
    
    pub fn operator_bool(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
}

pub struct ResourceBufferBarrier {
    object: Object,
    memory: MemoryBarrier,
}

pub struct ResourceBuffer {
    object: Object,
    memory: Memory,
}

impl ResourceBuffer {
    
    pub fn operator_bool(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
}

pub struct ResourceImageSamplerDescriptor {
    filter:       VkFilter,
    mipmap_mode:  VkSamplerMipmapMode,
    address_mode: VkSamplerAddressMode,
    border:       VkBorderColor,
}

//---------------------------
pub struct ResourceImageSamplerFactoryHasher {

}

impl ResourceImageSamplerFactoryHasher {
    
    pub fn invoke(&self, descriptor: &Descriptor) -> usize {
        
        todo!();
        /*
        
        */
    }
}

//---------------------------
pub struct ResourceImageSamplerFactory {
    device: VkDevice,
}

pub mod resource_image_sampler_factory {

    use super::*;

    pub type Descriptor = SamplerDescriptor;
    pub type Handle     = Handle<VkSampler,Deleter>;

    lazy_static!{
        /*
        typedef VK_DELETER(Sampler) Deleter;
        */
    }
}

impl ResourceImageSamplerFactory {
    
    pub fn new(gpu: &Gpu) -> Self {
    
        todo!();
        /*
        
        */
    }
    
    pub fn invoke(&self, descriptor: &Descriptor) -> Handle {
        
        todo!();
        /*
        
        */
    }
}

//---------------------------
pub struct ResourceImageSampler {
    cache: Cache,
}

pub mod resource_image_sampler {

    use super::*;

    pub type Cache = Cache<Factory>;
}

impl ResourceImageSampler {
    
    pub fn new(gpu: &Gpu) -> Self {
    
        todo!();
        /*
            : cache(Factory(gpu))
        */
    }
}

pub struct ResourceImageDescriptorUsage {
    image:  VkImageUsageFlags,
    memory: MemoryDescriptor,
}

pub struct ResourceImageDescriptorView {
    ty:     VkImageViewType,
    format: VkFormat,
}

//---------------------------
pub struct ResourceImageDescriptor {
    ty:      VkImageType,
    format:  VkFormat,
    extent:  VkExtent3D,
    usage:   ResourceImageDescriptorUsage,
    view:    ResourceImageDescriptorView,
    sampler: SamplerDescriptor,
}

pub struct ResourceImageObject {
    handle:  VkImage,
    layout:  VkImageLayout,
    view:    VkImageView,
    sampler: VkSampler,
}

impl ResourceImageObject {
    
    pub fn operator_bool(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
}

pub struct ResourceImageBarrierLayour {
    src: VkImageLayout,
    dst: VkImageLayout,
}

pub struct ResourceImageBarrier {
    object: Object,
    memory: MemoryBarrier,
    layout: ResourceImageBarrierLayour,
}

//---------------------------
pub struct ResourceImage {
    object: Object,
    memory: Memory,
}

impl ResourceImage {
    
    pub fn operator_bool(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
}

//---------------------------
pub struct ResourceFence {
    pool: *mut Pool,
    id:   usize,
}

impl ResourceFence {
    
    pub fn operator_bool(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn handle(&self, add_to_waitlist: bool) -> VkFence {
        let add_to_waitlist: bool = add_to_waitlist.unwrap_or(true);

        todo!();
        /*
        
        */
    }
    
    pub fn wait(&mut self, timeout_nanoseconds: u64)  {
        let timeout_nanoseconds: u64 = timeout_nanoseconds.unwrap_or(UINT64_MAX);

        todo!();
        /*
        
        */
    }
}

//------------------------------------------
pub struct ResourcePoolPolicy {

}

pub trait ResourcePoolPolicyInterface: Enact {}

pub trait Enact {

    fn enact(&mut self, 
        allocator:              VmaAllocator,
        memory_requirements:    &VkMemoryRequirements,
        allocation_create_info: &mut VmaAllocationCreateInfo);
}

impl ResourcePoolPolicy {
    
    pub fn linear(
        block_size:      VkDeviceSize,
        min_block_count: u32,
        max_block_count: u32) -> Box<Policy> {

        let block_size:      VkDeviceSize = block_size.unwrap_or(VMA_DEFAULT_LARGE_HEAP_BLOCK_SIZE);

        let min_block_count: u32 = min_block_count.unwrap_or(1);
        let max_block_count: u32 = max_block_count.unwrap_or(u32::MAX);

        todo!();
        /*
        
        */
    }
}

pub struct ResourcePoolMemory {
    policy: Box<Policy>,
}

pub type BufferHandler = fn(_0: &Buffer) -> ();

pub struct ResourcePoolBuffer{

    pool: Vec<Handle<Buffer, BufferHandler>>,
}

pub type ImageHandler = fn(_0: &Image) -> ();

pub struct ResourcePoolImage {

    pool:    Vec<Handle<Image, ImageHandler>>,
    sampler: ImageSampler,
}

pub struct ResourcePoolFence {
    pool:     Vec<Handle<VkFence, VkDeleter<Fence>>>,
    waitlist: RefCell<Vec<VkFence>>,
    in_use:   usize,
}

//------------------------------------------
pub struct ResourcePool {
    device:    VkDevice,
    allocator: Handle<VmaAllocator, fn() -> VmaAllocator>,
    memory:    ResourcePoolMemory,
    buffer:    ResourcePoolBuffer,
    image:     ResourcePoolImage,
    fence:     ResourcePoolFence,
}

pub mod resource_pool_configuration {

    use super::*;

    pub const RESERVE: u32 = 256;
}

impl ResourcePool {
    
    pub fn new(
        gpu: &Gpu,
        _1:  Box<Policy>) -> Self {
    
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
    
    pub fn buffer(&mut self, descriptor: &BufferDescriptor) -> Buffer {
        
        todo!();
        /*
        
        */
    }
    
    pub fn image(&mut self, descriptor: &ImageDescriptor) -> Image {
        
        todo!();
        /*
        
        */
    }
    
    pub fn fence(&mut self) -> Fence {
        
        todo!();
        /*
        
        */
    }
    
    pub fn purge(&mut self)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn uniform<Block>(&mut self, block: &Block) -> Buffer {
    
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

pub struct Resource {
    pool: ResourcePool,
}

impl ResourcePool {
    
    pub fn new(gpu: &Gpu) -> Self {
    
        todo!();
        /*
            : pool(gpu, Pool::Policy::linear())
        */
    }
}

pub struct ResourceMemoryScope {
    allocator:  VmaAllocator,
    allocation: VmaAllocation,
    access:     AccessFlags,
}

impl ResourceMemoryScope {

    pub fn new(
        allocator:  VmaAllocator,
        allocation: VmaAllocation,
        access:     AccessFlags) -> Self {
    
        todo!();
        /*
        
        */
    }
    
    pub fn invoke(&self, data: *const ())  {
        
        todo!();
        /*
        
        */
    }
}

impl ResourceMemory {
    
    #[inline] pub fn map<typename, Pointer>(&self) -> ResourceMemoryHandle<Pointer> {
    
        todo!();
        /*
            // Forward declaration
      void* map(const Memory&, Access::Flags);

      return Handle<Pointer>{
        reinterpret_cast<Pointer>(map(*this, Access::Read)),
        Scope(allocator, allocation, Access::Read),
      };
        */
    }
    
    #[inline] pub fn map<typename, const kAccess: ResourceMemoryAccessFlags, Pointer>(&mut self) -> ResourceMemoryHandle<Pointer> {
    
        todo!();
        /*
            // Forward declaration
      void* map(const Memory&, Access::Flags);

      static_assert(
          (kAccess == Access::Read) ||
          (kAccess == Access::Write) ||
          (kAccess == (Access::Read | Access::Write)),
          "Invalid memory access!");

      return Handle<Pointer>{
        reinterpret_cast<Pointer>(map(*this, kAccess)),
        Scope(allocator, allocation, kAccess),
      };
        */
    }
}

impl ResourceBufferObject {
    
    #[inline] pub fn operator_bool(&self) -> bool {
        
        todo!();
        /*
            return VK_NULL_HANDLE != handle;
        */
    }
}

impl ResourceBuffer {
    
    #[inline] pub fn operator_bool(&self) -> bool {
        
        todo!();
        /*
            return object;
        */
    }
}


impl PartialEq<ResourceImageSamplerDescriptor> for ResourceImageSamplerDescriptor {
    
    fn eq(&self, other: &ResourceImageSamplerDescriptor) -> bool {
        todo!();
        /*
            return (_1.filter == _2.filter && \
              _1.mipmap_mode == _2.mipmap_mode && \
              _1.address_mode == _2.address_mode && \
              _1.border == _2.border);
        */
    }
}

impl ResourceImageSamplerFactoryHasher {
    
    #[inline] pub fn invoke(&self, descriptor: &Descriptor) -> usize {
        
        todo!();
        /*
            return get_hash(
          descriptor.filter,
          descriptor.mipmap_mode,
          descriptor.address_mode,
          descriptor.border);
        */
    }
}

impl ResourceImageObject {
    
    #[inline] pub fn operator_bool(&self) -> bool {
        
        todo!();
        /*
            return VK_NULL_HANDLE != handle;
        */
    }
}

impl ResourceImage {
    
    #[inline] pub fn operator_bool(&self) -> bool {
        
        todo!();
        /*
            return object;
        */
    }
}

impl ResourceFence {
    
    #[inline] pub fn operator_bool(&self) -> bool {
        
        todo!();
        /*
            return pool;
        */
    }
}

impl ResourcePool {
    
    #[inline] pub fn uniform<Block>(&mut self, block: &Block) -> ResourceBuffer {
    
        todo!();
        /*
            Buffer uniform = this->buffer({
          sizeof(Block),
          {
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            {
              VMA_MEMORY_USAGE_CPU_TO_GPU,
              0u,
              0u,
            },
          },
        });

      {
        Memory::Handle<Block*> memory = uniform.memory.template map<
            Block,
            Memory::Access::Write>();

        *memory.get() = block;
      }

      return uniform;
        */
    }
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/api/Resource.cpp]

pub fn create_allocator(
    instance:        VkInstance,
    physical_device: VkPhysicalDevice,
    device:          VkDevice) -> VmaAllocator {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          instance,
          "Invalid Vulkan instance!");

      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          physical_device,
          "Invalid Vulkan physical device!");

      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          device,
          "Invalid Vulkan device!");

      const VmaAllocatorCreateInfo allocator_create_info{
        0u,
        physical_device,
        device,
        0u,
        nullptr,
        nullptr,
        1u,
        nullptr,
        nullptr,
        nullptr,
        instance,
        VK_API_VERSION_1_0,
      };

      VmaAllocator allocator{};
      VK_CHECK(vmaCreateAllocator(&allocator_create_info, &allocator));
      TORCH_CHECK(allocator, "Invalid VMA (Vulkan Memory Allocator) allocator!");

      return allocator;
        */
}

pub fn create_allocation_create_info(descriptor: &ResourceMemoryDescriptor) -> VmaAllocationCreateInfo {
    
    todo!();
        /*
            return VmaAllocationCreateInfo{
        VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT |
            /* VMA_ALLOCATION_CREATE_MAPPED_BIT - MoltenVK Issue #175 */
            0,
        descriptor.usage,
        descriptor.required,
        descriptor.preferred,
        0u,
        VK_NULL_HANDLE,
        nullptr,
        0.5f,
      };
        */
}

pub fn release_buffer(buffer: &ResourceBuffer)  {
    
    todo!();
        /*
            // Safe to pass null as buffer or allocation.
      vmaDestroyBuffer(
          buffer.memory.allocator,
          buffer.object.handle,
          buffer.memory.allocation);
        */
}

pub fn release_image(image: &ResourceImage)  {
    
    todo!();
        /*
            // Sampler is an immutable object. Its lifetime is managed through the cache.

      if (VK_NULL_HANDLE != image.object.view) {
        VmaAllocatorInfo allocator_info{};
        vmaGetAllocatorInfo(image.memory.allocator, &allocator_info);
        vkDestroyImageView(allocator_info.device, image.object.view, nullptr);
      }

      // Safe to pass null as image or allocation.
      vmaDestroyImage(
          image.memory.allocator,
          image.object.handle,
          image.memory.allocation);
        */
}

pub fn map(
    memory: &ResourceMemory,
    access: ResourceMemoryAccessFlags)  {

    todo!();
        /*
            void* data = nullptr;
      VK_CHECK(vmaMapMemory(memory.allocator, memory.allocation, &data));

      if (access & Resource::Memory::Access::Read) {
        // Call will be ignored by implementation if the memory type this allocation
        // belongs to is not HOST_VISIBLE or is HOST_COHERENT, which is the behavior
        // we want.
        VK_CHECK(vmaInvalidateAllocation(
            memory.allocator, memory.allocation, 0u, VK_WHOLE_SIZE));
      }

      return data;
        */
}

impl ResourceMemoryScope {
    
    pub fn new(
        allocator:  VmaAllocator,
        allocation: VmaAllocation,
        access:     AccessFlags) -> Self {
    
        todo!();
        /*


            : allocator_(allocator),
        allocation_(allocation),
        access_(access) 

      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          allocator,
          "Invalid VMA (Vulkan Memory Allocator) allocator!");

      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          allocation,
          "Invalid VMA (Vulkan Memory Allocator) allocation!");
        */
    }
    
    pub fn invoke(&self, data: *const ())  {
        
        todo!();
        /*
            if (C10_UNLIKELY(!data)) {
        return;
      }

      if (access_ & Access::Write) {
        // Call will be ignored by implementation if the memory type this allocation
        // belongs to is not HOST_VISIBLE or is HOST_COHERENT, which is the behavior
        // we want.
        VK_CHECK(vmaFlushAllocation(allocator_, allocation_, 0u, VK_WHOLE_SIZE));
      }

      vmaUnmapMemory(allocator_, allocation_);
        */
    }
}

impl ResourceImageSamplerFactory {
    
    pub fn new(gpu: &Gpu) -> Self {
    
        todo!();
        /*


            : device_(gpu.device) 

      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          device_,
          "Invalid Vulkan device!");
        */
    }
    
    pub fn invoke(&self, descriptor: &Descriptor) -> ResourceImageSamplerFactoryHandle {
        
        todo!();
        /*
            const VkSamplerCreateInfo sampler_create_info{
        VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        nullptr,
        0u,
        descriptor.filter,
        descriptor.filter,
        descriptor.mipmap_mode,
        descriptor.address_mode,
        descriptor.address_mode,
        descriptor.address_mode,
        0.0f,
        VK_FALSE,
        1.0f,
        VK_FALSE,
        VK_COMPARE_OP_NEVER,
        0.0f,
        VK_LOD_CLAMP_NONE,
        descriptor.border,
        VK_FALSE,
      };

      VkSampler sampler{};
      VK_CHECK(vkCreateSampler(
          device_,
          &sampler_create_info,
          nullptr,
          &sampler));

      TORCH_CHECK(
          sampler,
          "Invalid Vulkan image sampler!");

      return Handle{
        sampler,
        Deleter(device_),
      };
        */
    }
}

impl ResourceFence {
    
    pub fn handle(&self, add_to_waitlist: bool) -> VkFence {
        
        todo!();
        /*
            if (!pool) {
        return VK_NULL_HANDLE;
      }

      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          id < pool->fence_.pool.size(),
          "Invalid Vulkan fence!");

      const VkFence fence = pool->fence_.pool[id].get();

      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          fence,
          "Invalid Vulkan fence!");

      if (add_to_waitlist) {
        pool->fence_.waitlist.push_back(fence);
      }

      return fence;
        */
    }
    
    pub fn wait(&mut self, timeout_nanoseconds: u64)  {
        
        todo!();
        /*
            const VkFence fence = handle(/* add_to_waitlist = */ false);

      const auto waitlist_itr = find(
          pool->fence_.waitlist.cbegin(),
          pool->fence_.waitlist.cend(),
          fence);

      if (pool->fence_.waitlist.cend() != waitlist_itr) {
        VK_CHECK(vkWaitForFences(
            pool->device_,
            1u,
            &fence,
            VK_TRUE,
            timeout_nanoseconds));

        VK_CHECK(vkResetFences(
            pool->device_,
            1u,
            &fence));

        pool->fence_.waitlist.erase(waitlist_itr);
      }
        */
    }
}

pub struct LinearEntryDeleter {
    allocator: VmaAllocator,
}

impl LinearEntryDeleter {
    
    pub fn new(_0: VmaAllocator) -> Self {
    
        todo!();
        /*
        
        */
    }
    
    pub fn invoke(&self, _0: VmaPool)  {
        
        todo!();
        /*
        
        */
    }
}

pub struct LinearEntry {
    memory_type_index: u32,
    handle:            Handle<VmaPool,LinearEntryDeleter>,
}

pub struct LinearBlock {
    size: VkDeviceSize,
    min:  u32,
    max:  u32,
}

pub struct Linear {
    base:  ResourcePoolPolicy,
    pools: Vec<LinearEntry>,
    block: LinearBlock,
}

impl Linear {
    
    pub fn new(
        block_size:      VkDeviceSize,
        min_block_count: u32,
        max_block_count: u32) -> Self {
    
        todo!();
        /*


        
        */
    }
}

pub mod linear_configuration {

    pub const RESERVE: u32 = 16;
}

impl Enact for Linear {
    
    fn enact(&mut self, 
        allocator:              VmaAllocator,
        memory_requirements:    &VkMemoryRequirements,
        allocation_create_info: &mut VmaAllocationCreateInfo)  {
        
        todo!();
        /*
        
        */
    }
}

impl LinearEntryDeleter {
    
    pub fn new(allocator: VmaAllocator) -> Self {
    
        todo!();
        /*
            : allocator_(allocator)
        */
    }
    
    pub fn invoke(&self, pool: VmaPool)  {
        
        todo!();
        /*
            vmaDestroyPool(allocator_, pool);
        */
    }
}

impl Linear {
    
    pub fn new(
        block_size:      VkDeviceSize,
        min_block_count: u32,
        max_block_count: u32) -> Self {
    
        todo!();
        /*


            : block_ {
          block_size,
          min_block_count,
          max_block_count,
        } 
      pools_.reserve(Configuration::kReserve);
        */
    }
    
    pub fn enact(&mut self, 
        allocator:              VmaAllocator,
        memory_requirements:    &VkMemoryRequirements,
        allocation_create_info: &mut VmaAllocationCreateInfo)  {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          allocator,
          "Invalid VMA (Vulkan Memory Allocator) allocator!");

      u32 memory_type_index = 0u;
      VK_CHECK(vmaFindMemoryTypeIndex(
          allocator,
          memory_requirements.memoryTypeBits,
          &allocation_create_info,
          &memory_type_index));

      auto pool_itr = find_if(
          pools_.begin(),
          pools_.end(),
          [memory_type_index](const Entry& entry) {
        return entry.memory_type_index == memory_type_index;
      });

      if (pools_.end() == pool_itr) {
        const VmaPoolCreateInfo pool_create_info{
          memory_type_index,
          VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT,
          block_.size,
          block_.min,
          block_.max,
          0u,
        };

        VmaPool pool{};
        VK_CHECK(vmaCreatePool(
            allocator,
            &pool_create_info,
            &pool));

        TORCH_CHECK(
            pool,
            "Invalid VMA (Vulkan Memory Allocator) memory pool!");

        pools_.push_back({
          memory_type_index,
          {
            pool,
            Entry::Deleter(allocator),
          },
        });

        pool_itr = prev(pools_.end());
      }

      allocation_create_info.pool = pool_itr->handle.get();
        */
    }
}

impl ResourcePoolPolicy {
    
    pub fn linear(&mut self, 
        block_size:      VkDeviceSize,
        min_block_count: u32,
        max_block_count: u32) -> Box<ResourcePoolPolicy> {
        
        todo!();
        /*
            return make_unique<Linear>(
          block_size,
          min_block_count,
          max_block_count);
        */
    }
}

impl ResourcePool {
    
    pub fn new(
        gpu:    &Gpu,
        policy: Box<Policy>) -> Self {
    
        todo!();
        /*


            : device_(gpu.device),
        allocator_(
            create_allocator(
                gpu.adapter->runtime->instance(),
                gpu.adapter->handle,
                device_),
            vmaDestroyAllocator),
        memory_{
          move(policy),
        },
        image_{
          .sampler = Image::Sampler{gpu},
        },
        fence_{} 
      buffer_.pool.reserve(Configuration::kReserve);
      image_.pool.reserve(Configuration::kReserve);
      fence_.pool.reserve(Configuration::kReserve);
        */
    }
    
    pub fn new(pool: Pool) -> Self {
    
        todo!();
        /*


            : device_(move(pool.device_)),
        allocator_(move(pool.allocator_)),
        memory_(move(pool.memory_)),
        buffer_(move(pool.buffer_)),
        image_(move(pool.image_)),
        fence_(move(pool.fence_)) 
      pool.invalidate();
        */
    }
    
    pub fn assign_from(&mut self, pool: Pool) -> &mut ResourcePool {
        
        todo!();
        /*
            if (&pool != this) {
        device_ = move(pool.device_);
        allocator_ = move(pool.allocator_);
        memory_ = move(pool.memory_);
        buffer_ = move(pool.buffer_);
        image_ = move(pool.image_);
        fence_ = move(pool.fence_);

        pool.invalidate();
      };

      return *this;
        */
    }
}

impl Drop for ResourcePool {

    fn drop(&mut self) {
        todo!();
        /*
            try {
        if (device_ && allocator_) {
          purge();
        }
      }
      catch (const exception& e) {
        TORCH_WARN(
            "Vulkan: Resource pool destructor raised an exception! Error: ",
            e.what());
      }
      catch (...) {
        TORCH_WARN(
            "Vulkan: Resource pool destructor raised an exception! "
            "Error: Unknown");
      }
        */
    }
}

impl ResourcePool {
    
    pub fn buffer(&mut self, descriptor: &BufferDescriptor) -> ResourceBuffer {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          device_ && allocator_,
          "This resource pool is in an invalid state! ",
          "Potential reason: This resource pool is moved from.");

      const VkBufferCreateInfo buffer_create_info{
        VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        nullptr,
        0u,
        descriptor.size,
        descriptor.usage.buffer,
        VK_SHARING_MODE_EXCLUSIVE,
        0u,
        nullptr,
      };

      VkBuffer buffer{};
      VK_CHECK(vkCreateBuffer(
          device_,
          &buffer_create_info,
          nullptr,
          &buffer));

      TORCH_CHECK(
          buffer,
          "Invalid Vulkan buffer!");

      VkMemoryRequirements memory_requirements{};
      vkGetBufferMemoryRequirements(
          device_,
          buffer,
          &memory_requirements);

      VmaAllocationCreateInfo allocation_create_info =
          create_allocation_create_info(descriptor.usage.memory);

      if (memory_.policy) {
        memory_.policy->enact(
            allocator_.get(),
            memory_requirements,
            allocation_create_info);
      }

      VmaAllocation allocation{};
      VK_CHECK(vmaAllocateMemory(
          allocator_.get(),
          &memory_requirements,
          &allocation_create_info,
          &allocation,
          nullptr));

      TORCH_CHECK(
          allocation,
          "Invalid VMA (Vulkan Memory Allocator) allocation!");

      VK_CHECK(vmaBindBufferMemory(
          allocator_.get(),
          allocation,
          buffer));

      buffer_.pool.emplace_back(
          Buffer{
            Buffer::Object{
              buffer,
              0u,
              descriptor.size,
            },
            Memory{
              allocator_.get(),
              allocation,
            },
          },
          &release_buffer);

      return buffer_.pool.back().get();
        */
    }
    
    pub fn image(&mut self, descriptor: &ImageDescriptor) -> ResourceImage {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          device_ && allocator_,
          "This resource pool is in an invalid state! ",
          "Potential reason: This resource pool is moved from.");

      const VkImageCreateInfo image_create_info{
        VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        nullptr,
        0u,
        descriptor.type,
        descriptor.format,
        descriptor.extent,
        1u,
        1u,
        VK_SAMPLE_COUNT_1_BIT,
        VK_IMAGE_TILING_OPTIMAL,
        descriptor.usage.image,
        VK_SHARING_MODE_EXCLUSIVE,
        0u,
        nullptr,
        VK_IMAGE_LAYOUT_UNDEFINED,
      };

      VkImage image{};
      VK_CHECK(vkCreateImage(
          device_,
          &image_create_info,
          nullptr,
          &image));

      TORCH_CHECK(
          image,
          "Invalid Vulkan image!");

      VkMemoryRequirements memory_requirements{};
      vkGetImageMemoryRequirements(
          device_,
          image,
          &memory_requirements);

      VmaAllocationCreateInfo allocation_create_info =
          create_allocation_create_info(descriptor.usage.memory);

      if (memory_.policy) {
        memory_.policy->enact(
            allocator_.get(),
            memory_requirements,
            allocation_create_info);
      }

      VmaAllocation allocation{};
      VK_CHECK(vmaAllocateMemory(
          allocator_.get(),
          &memory_requirements,
          &allocation_create_info,
          &allocation,
          nullptr));

      TORCH_CHECK(
          allocation,
          "Invalid VMA (Vulkan Memory Allocator) allocation!");

      VK_CHECK(vmaBindImageMemory(
          allocator_.get(),
          allocation,
          image));

      const VkImageViewCreateInfo image_view_create_info{
        VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        nullptr,
        0u,
        image,
        descriptor.view.type,
        descriptor.view.format,
        {
          VK_COMPONENT_SWIZZLE_IDENTITY,
          VK_COMPONENT_SWIZZLE_IDENTITY,
          VK_COMPONENT_SWIZZLE_IDENTITY,
          VK_COMPONENT_SWIZZLE_IDENTITY,
        },
        {
          VK_IMAGE_ASPECT_COLOR_BIT,
          0u,
          VK_REMAINING_MIP_LEVELS,
          0u,
          VK_REMAINING_ARRAY_LAYERS,
        },
      };

      VkImageView view{};
      VK_CHECK(vkCreateImageView(
          device_,
          &image_view_create_info,
          nullptr,
          &view));

      TORCH_CHECK(
          view,
          "Invalid Vulkan image view!");

      image_.pool.emplace_back(
          Image{
            Image::Object{
              image,
              VK_IMAGE_LAYOUT_UNDEFINED,
              view,
              image_.sampler.cache.retrieve(descriptor.sampler),
            },
            Memory{
              allocator_.get(),
              allocation,
            },
          },
          &release_image);

      return image_.pool.back().get();
        */
    }
    
    pub fn fence(&mut self) -> ResourceFence {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          device_ && allocator_,
          "This resource pool is in an invalid state! ",
          "Potential reason: This resource pool is moved from.");

      if (fence_.pool.size() == fence_.in_use) {
        const VkFenceCreateInfo fence_create_info{
          VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
          nullptr,
          0u,
        };

        VkFence fence{};
        VK_CHECK(vkCreateFence(
            device_,
            &fence_create_info,
            nullptr,
            &fence));

        TORCH_CHECK(
            fence,
            "Invalid Vulkan fence!");

        fence_.pool.emplace_back(fence, VK_DELETER(Fence)(device_));
      }

      return Fence{
        this,
        fence_.in_use++,
      };
        */
    }
    
    pub fn purge(&mut self)  {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          device_ && allocator_,
          "This resource pool is in an invalid state! ",
          "Potential reason: This resource pool is moved from.");

      if (!fence_.waitlist.empty()) {
        VK_CHECK(vkWaitForFences(
            device_,
            fence_.waitlist.size(),
            fence_.waitlist.data(),
            VK_TRUE,
            UINT64_MAX));

        VK_CHECK(vkResetFences(
            device_,
            fence_.waitlist.size(),
            fence_.waitlist.data()));

        fence_.waitlist.clear();
      }

      fence_.in_use = 0u;
      image_.pool.clear();
      buffer_.pool.clear();
        */
    }
    
    pub fn invalidate(&mut self)  {
        
        todo!();
        /*
            device_ = VK_NULL_HANDLE;
      allocator_.reset();
        */
    }
}
