crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/api/Pipeline.h]
pub struct PipelineBarrierStage {
    src: VkPipelineStageFlags,
    dst: VkPipelineStageFlags,
}

//-------------------------------------
pub struct PipelineBarrier {
    stage:   PipelineBarrierStage,
    buffers: SmallVec<[ResourceBufferBarrier;4]>,
    images:  SmallVec<[ResourceImageBarrier;4]>,
}

impl PipelineBarrier {
    
    pub fn operator_bool(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
}


//-------------------------------------
pub struct PipelineLayoutDescriptor {
    descriptor_set_layout: VkDescriptorSetLayout,
}

//-------------------------------------
pub struct PipelineLayoutFactoryHasher {

}

impl PipelineLayoutFactoryHasher {
    
    pub fn invoke(&self, descriptor: &Descriptor) -> usize {
        
        todo!();
        /*
        
        */
    }
}


//-------------------------------------
pub struct PipelineLayoutFactory {
    device: VkDevice,
}

pub mod pipeline_layout_factory {

    use super::*;

    pub type Descriptor = LayoutDescriptor;
    pub type Handle     = Handle<VkPipelineLayout,Deleter>;

    lazy_static!{
        /*
        typedef VK_DELETER(PipelineLayout) Deleter;
        */
    }
}

impl PipelineLayoutFactory {

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

//-----------------------------------
pub struct PipelineLayout {
    cache: Cache,
}

impl PipelineLayout {
    
    pub fn new(gpu: &Gpu) -> Self {
    
        todo!();
        /*
            : cache(Factory(gpu))
        */
    }
}

pub mod pipeline_layout {

    use super::*;

    pub type Cache = Cache<Factory>;
}

bitflags!{
    pub struct PipelineStageType: u8 {
        const None     = 0;
        const Compute  = 1;
        const Host     = 2;
        const Transfer = 4;
    }
}

pub mod pipeline_stage {

    pub type Flags = u8;
}

pub struct PipelineDescriptor {
    pipeline_layout:  VkPipelineLayout,
    shader_module:    vk::ShaderModule,
    local_work_group: ShaderWorkGroup,
}

//-----------------------------------
pub struct PipelineFactoryHasher {

}

impl PipelineFactoryHasher {
    
    pub fn invoke(&self, descriptor: &Descriptor) -> usize {
        
        todo!();
        /*
        
        */
    }
}

//-----------------------------------
pub struct PipelineFactory {
    device:         VkDevice,
    pipeline_cache: Handle<VkPipelineCache, VK_DELETER(PipelineCache)>,
}

pub mod pipeline_factory {

    use super::*;

    pub type Descriptor = PipelineDescriptor;
    pub type Handle     = Handle<VkPipeline,Deleter>;

    lazy_static!{
        /*
        typedef VK_DELETER(Pipeline) Deleter;
        */
    }
}

impl PipelineFactory {
    
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

//-----------------------------------
pub struct PipelineObject {
    handle:           VkPipeline,
    layout:           VkPipelineLayout,
    local_work_group: ShaderWorkGroup,
}

impl PipelineObject {
    
    pub fn operator_bool(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
}

//----------------------------------
pub struct PipelineCache {
    cache: Cache<Factory>,
}

impl PipelineCache {
    
    pub fn new(factory: Factory) -> Self {
    
        todo!();
        /*


        
        */
    }
    
    pub fn retrieve(&mut self, descriptor: &Descriptor) -> Object {
        
        todo!();
        /*
        
        */
    }
    
    pub fn purge(&mut self)  {
        
        todo!();
        /*
        
        */
    }
}

/**
  | This struct defines pipeline, and pipeline
  | layout, caches intended to minimize redundant
  | object reconstructions at the cost of extra
  | memory consumption.
  |
  | A Vulkan pipeline contains the entirety of
  | states, as one coherent monolithic bundle,
  | required to configure the GPU's execution
  | pipeline.
  |
  | This usage pattern minimizes driver overhead,
  | promotes pipeline state reuse, and is
  | a departure from, and in direct contrast with,
  | OpenGL's individually confiurable state
  | machine.
  |
  | A Vulkan pipeline layout represents a sequence
  | of Vulkan descriptor sets each having
  | a specific layout, and deterimines the
  | interface between all shader stages and shader
  | resources.
  |
  | For more information on shaders and shader
  | layouts check the description of
  | navie::vulkan::api::Shader.
  |
  | This struct defines the facilities required to
  | create, reuse, and destruct these Vulkan
  | objects.
  |
  */
pub struct Pipeline {
    layout: PipelineLayout,
    cache:  PipelineCache,
}

impl Pipeline {
    
    pub fn new(gpu: &Gpu) -> Self {
    
        todo!();
        /*


            : layout(gpu),
          cache(Factory(gpu))
        */
    }
}

impl PipelineBarrier {
    
    #[inline] pub fn operator_bool(&self) -> bool {
        
        todo!();
        /*
            return (0u != stage.src) ||
             (0u != stage.dst) ||
             !buffers.empty() ||
             !images.empty();
        */
    }
}

impl PartialEq<PipelineLayoutDescriptor> for PipelineLayoutDescriptor {
    
    fn eq(&self, other: &PipelineLayoutDescriptor) -> bool {
        todo!();
        /*
            return (_1.descriptor_set_layout == _2.descriptor_set_layout);
        */
    }
}

impl PipelineLayoutFactoryHasher {
    
    #[inline] pub fn invoke(&self, descriptor: &Descriptor) -> usize {
        
        todo!();
        /*
            return get_hash(descriptor.descriptor_set_layout);
        */
    }
}


impl PartialEq<PipelineDescriptor> for PipelineDescriptor {
    
    fn eq(&self, other: &PipelineDescriptor) -> bool {
        todo!();
        /*
            return (_1.pipeline_layout == _2.pipeline_layout && \
              _1.shader_module == _2.shader_module && \
              _1.local_work_group == _2.local_work_group);
        */
    }
}

impl PipelineFactoryHasher {
    
    #[inline] pub fn invoke(&self, descriptor: &Descriptor) -> usize {
        
        todo!();
        /*
            return get_hash(
          descriptor.pipeline_layout,
          descriptor.shader_module,
          descriptor.local_work_group.data[0u],
          descriptor.local_work_group.data[1u],
          descriptor.local_work_group.data[2u]);
        */
    }
}

impl PipelineObject {
    
    #[inline] pub fn operator_bool(&self) -> bool {
        
        todo!();
        /*
            return (VK_NULL_HANDLE != handle) &&
             (VK_NULL_HANDLE != layout);
        */
    }
}

impl PipelineCache {
    
    #[inline] pub fn retrieve(&mut self, descriptor: &Descriptor) -> PipelineObject {
        
        todo!();
        /*
            return {
        cache_.retrieve(descriptor),
        descriptor.pipeline_layout,
        descriptor.local_work_group,
      };
        */
    }
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/api/Pipeline.cpp]

impl PipelineLayoutFactory {
    
    pub fn new(gpu: &Gpu) -> Self {
    
        todo!();
        /*


            : device_(gpu.device) 

      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          device_,
          "Invalid Vulkan device!");
        */
    }
    
    pub fn invoke(&self, descriptor: &Descriptor) -> PipelineLayoutFactoryHandle {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          descriptor.descriptor_set_layout,
          "Invalid Vulkan descriptor set layout!");

      const VkPipelineLayoutCreateInfo pipeline_layout_create_info{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        nullptr,
        0u,
        1u,
        &descriptor.descriptor_set_layout,
        0u,
        nullptr,
      };

      VkPipelineLayout pipeline_layout{};
      VK_CHECK(vkCreatePipelineLayout(
          device_,
          &pipeline_layout_create_info,
          nullptr,
          &pipeline_layout));

      TORCH_CHECK(
          pipeline_layout,
          "Invalid Vulkan pipeline layout!");

      return Handle{
        pipeline_layout,
        Deleter(device_),
      };
        */
    }
}

pub fn create_pipeline_cache(device: VkDevice) -> VkPipelineCache {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          device,
          "Invalid Vulkan device!");

      const VkPipelineCacheCreateInfo pipeline_cache_create_info{
        VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
        nullptr,
        0u,
        0u,
        nullptr,
      };

      VkPipelineCache pipeline_cache{};
      VK_CHECK(vkCreatePipelineCache(
          device,
          &pipeline_cache_create_info,
          nullptr,
          &pipeline_cache));

      TORCH_CHECK(
          pipeline_cache,
          "Invalid Vulkan pipeline cache!");

      return pipeline_cache;
        */
}

impl PipelineFactory {
    
    pub fn new(gpu: &Gpu) -> Self {
    
        todo!();
        /*


            : device_(gpu.device),
       pipeline_cache_(
          create_pipeline_cache(device_),
          VK_DELETER(PipelineCache)(device_)) 

      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          device_,
          "Invalid Vulkan device!");

      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          pipeline_cache_,
          "Invalid Vulkan pipeline cache!");
        */
    }
    
    pub fn invoke(&self, descriptor: &Descriptor) -> PipelineFactoryHandle {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          descriptor.pipeline_layout,
          "Invalid Vulkan pipeline layout!");

      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          descriptor.shader_module,
          "Invalid Vulkan shader module!");

      constexpr VkSpecializationMapEntry specialization_map_entires[3]{
        // X
        {
          0u,
          offsetof(Shader::WorkGroup, data[0u]),
          sizeof(Shader::WorkGroup::data[0u]),
        },
        // Y
        {
          1u,
          offsetof(Shader::WorkGroup, data[1u]),
          sizeof(Shader::WorkGroup::data[1u]),
        },
        // Z
        {
          2u,
          offsetof(Shader::WorkGroup, data[2u]),
          sizeof(Shader::WorkGroup::data[2u]),
        },
      };

      const VkSpecializationInfo specialization_info{
        3u,
        specialization_map_entires,
        sizeof(descriptor.local_work_group),
        &descriptor.local_work_group,
      };

      const VkComputePipelineCreateInfo compute_pipeline_create_info{
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        nullptr,
        0u,
        VkPipelineShaderStageCreateInfo{
          VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          nullptr,
          0u,
          VK_SHADER_STAGE_COMPUTE_BIT,
          descriptor.shader_module,
          "main",
          &specialization_info,
        },
        descriptor.pipeline_layout,
        VK_NULL_HANDLE,
        0u,
      };

      VkPipeline pipeline{};
      VK_CHECK(vkCreateComputePipelines(
          device_,
          pipeline_cache_.get(),
          1u,
          &compute_pipeline_create_info,
          nullptr,
          &pipeline));

      TORCH_CHECK(
          pipeline,
          "Invalid Vulkan pipeline!");

      return Handle{
        pipeline,
        Deleter(device_),
      };
        */
    }
}

impl PipelineCache {
    
    pub fn new(factory: Factory) -> Self {
    
        todo!();
        /*


            : cache_(move(factory))
        */
    }
    
    pub fn purge(&mut self)  {
        
        todo!();
        /*
            cache_.purge();
        */
    }
}
