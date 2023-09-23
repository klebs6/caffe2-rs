crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/api/Shader.h]
pub enum ShaderDescriptorType {
    Source,
    Binary,
}

pub struct ShaderDescriptorShaderSource {

    /**
      | Null-terminated
      |
      */
    glsl:   *const u8,

    /**
      | Padding
      |
      */
    unused: u32,
}

pub struct ShaderDescriptorShaderBinary {
    spirv: *const u32,

    /**
      | Bytes
      |
      */
    size:  u32,
}

pub union ShaderDescriptorShader {
    source: ShaderDescriptorShaderSource,
    binary: ShaderDescriptorShaderBinary,
}

pub struct ShaderDescriptor {
    ty:     ShaderDescriptorType,
    shader: ShaderDescriptorShader,
}

impl ShaderDescriptor {
    
    pub fn new(glsl: *const u8) -> Self {
    
        todo!();
        /*


        
        */
    }
    
    pub fn new(
        spirv: *const u32,
        bytes: u32) -> Self {
    
        todo!();
        /*


        
        */
    }
}

pub struct ShaderLayoutDescriptor {
    signature: Signature,
}

pub struct ShaderLayoutFactoryHasher {

}

impl ShaderLayoutFactoryHasher {
    
    pub fn invoke(&self, descriptor: &Descriptor) -> usize {
        
        todo!();
        /*
        
        */
    }
}

pub struct ShaderLayoutFactory {
    device: VkDevice,
}

impl HasDescriptor for ShaderLayoutFactory {
    type Descriptor = LayoutDescriptor;
}

impl HasDeleter for ShaderLayoutFactory {
    type Deleter = VK_DELETER(DescriptorSetLayout);
}

impl HasHandle for ShaderLayoutFactory {
    type Handle = Handle<VkDescriptorSetLayout,Deleter>;
}

impl ShaderLayoutFactory {
    
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

pub struct ShaderLayoutObject {
    handle:    VkDescriptorSetLayout,
    signature: Signature,
}

impl ShaderLayoutObject {
    
    pub fn operator_bool(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
}

pub struct ShaderLayoutCache {
    cache: Cache<Factory>,
}

impl ShaderLayoutCache {

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

pub struct ShaderLayout {
    cache: ShaderLayoutCache,
}

impl HasSignature for ShaderLayout {
    type Signature = SmallVec<[VkDescriptorType;6]>;
}

impl ShaderLayout {
    
    pub fn new(gpu: &Gpu) -> Self {
    
        todo!();
        /*


            : cache(Factory(gpu))
        */
    }
}

pub struct ShaderFactoryHasher {

}

impl ShaderFactoryHasher {
    
    pub fn invoke(&self, descriptor: &Descriptor) -> usize {
        
        todo!();
        /*
        
        */
    }
}

pub struct ShaderFactory {
    device:   VkDevice,
    compiler: Box<Compiler>,
}

impl ShaderFactory {
    
    pub fn new(gpu: &Gpu) -> Self {
    
        todo!();
        /*


        
        */
    }
    
    pub fn new(_0: Factory) -> Self {
    
        todo!();
        /*


        
        */
    }
    
    pub fn assign_from(&mut self, _0: Factory) -> &mut Factory {
        
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

impl HasDescriptor for ShaderFactory {
    type Descriptor = ShaderDescriptor;
}

impl HasDeleter for ShaderFactory {
    type Deleter = VK_DELETER(ShaderModule);
}

impl HasHandle for ShaderFactory {
    type Handle = Handle<vk::ShaderModule,Deleter>;
}

/**
  | This struct defines shader, and shader layout,
  | caches intended to minimize redundant object
  | reconstructions at the cost of extra memory
  | consumption.
  |
  | A shader is a small, usually simple, program
  | that typically runs on a GPU as part of the
  | graphics or compute pipelines.  The shader
  | layout defines the interface between that
  | program and the outside world, namely what the
  | host (i.e. CPU) sees as configurable parameters
  | of the said shader per dispatch.
  |
  | If the shader was a regular function, the
  | shader layout would have been its function
  | prototype declaring the number and type of its
  | arguments.
  |
  | Furthermore, shader layouts, or as Vulkan calls
  | them descriptor set layouts, define the
  | blueprint out of which descriptor sets are
  | instantiated.  Descriptor sets themselves,
  | bundle the input to and output from a shader
  | and contain pointers to GPU, and GPU accessible
  | system, memory locations where the actual
  | resources reside.  Shader layouts are also used
  | in creation of Vulkan pipeline layouts, while
  | multiple shaders are bundled together to form
  | a portion of the the monolithic state objects
  | that are Vulkan pipelines.
  |
  | This struct defines the facilities required to
  | create, compile, reuse, and destruct the
  | aforementioned Vulkan objects.
  |
  */
pub struct Shader {
    layout: ShaderLayout,
    cache: Cache,
}

impl HasWorkGroup for Shader {
    type WorkGroup = uvec3;
}

impl HasCache for Shader {
    type Cache = Cache<Factory>;
}

impl Shader {
    
    pub fn new(gpu: &Gpu) -> Self {
    
        todo!();
        /*
        : layout(gpu),
          cache(Factory(gpu))
        */
    }
}

impl PartialEq<ShaderLayoutDescriptor> for ShaderLayoutDescriptor {
    
    fn eq(&self, other: &ShaderLayoutDescriptor) -> bool {
        todo!();
        /*
            return _1.signature == _2.signature;
        */
    }
}

impl ShaderLayoutFactoryHasher {
    
    #[inline] pub fn invoke(&self, descriptor: &Descriptor) -> usize {
        
        todo!();
        /*
            usize hash = 0u;

      for (const VkDescriptorType type : descriptor.signature) {
        hash = hash_combine(
            hash,
            get_hash(type));
      }

      return hash;
        */
    }
}

impl ShaderLayoutObject {

    #[inline] pub fn operator_bool(&self) -> bool {
        
        todo!();
        /*
            return VK_NULL_HANDLE != handle;
        */
    }
}

impl ShaderLayoutCache {
    
    #[inline] pub fn retrieve(&mut self, descriptor: &Descriptor) -> ShaderLayoutObject {
        
        todo!();
        /*
            return {
        cache_.retrieve(descriptor),
        descriptor.signature,
      };
        */
    }
}

impl PartialEq<ShaderWorkGroup> for ShaderWorkGroup {
    
    fn eq(&self, other: &ShaderWorkGroup) -> bool {
        todo!();
        /*
            return (_1.data[0u] == _2.data[0u] && _1.data[1u] == _2.data[1u] && _1.data[2u] == _2.data[2u]);
        */
    }
}

impl ShaderDescriptor {
    
    pub fn new(glsl: *const u8) -> Self {
    
        todo!();
        /*


            : type(Type::Source),
       shader{
        .source = {
          glsl,
          0u,
        },
       } 

      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          glsl,
          "Invalid shader source code!");
        */
    }
    
    pub fn new(
        code: *const u32,
        size: u32) -> Self {
    
        todo!();
        /*


            : type(Type::Binary),
       shader{
        .binary = {
          code,
          size,
        },
       } 


      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          code && (0u != size),
          "Invalid shader binary!");
        */
    }
}

impl PartialEq<ShaderDescriptor> for ShaderDescriptor {
    
    fn eq(&self, other: &ShaderDescriptor) -> bool {
        todo!();
        /*
            if (_1.type != _2.type)
        return false;

      if (_1.type == Shader::Descriptor::Type::Binary) {
        return (_1.shader.binary.spirv == _2.shader.binary.spirv && \
                _1.shader.binary.size == _2.shader.binary.size);
      }
      else {
        return (_1.shader.source.glsl == _2.shader.source.glsl);
      }
        */
    }
}

impl ShaderFactoryHasher {
    
    #[inline] pub fn invoke(&self, descriptor: &Descriptor) -> usize {
        
        todo!();
        /*
            static_assert(
          sizeof(Descriptor::shader.source) == sizeof(Descriptor::shader.binary),
          "This implementation requires sizeof(Source) to be equal to sizeof(Binary).");

      return get_hash(
          descriptor.type,
          descriptor.shader.binary.spirv,
          descriptor.shader.binary.size);
        */
    }
}

impl PartialEq<VkDescriptorSetLayoutBinding> for VkDescriptorSetLayoutBinding {
    
    fn eq(&self, other: &VkDescriptorSetLayoutBinding) -> bool {
        todo!();
        /*
            return (_1.binding == _2.binding && \
              _1.descriptorType == _2.descriptorType && \
              _1.descriptorCount == _2.descriptorCount && \
              _1.stageFlags == _2.stageFlags && \
              _1.pImmutableSamplers == _2.pImmutableSamplers);
        */
    }
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/api/Shader.cpp]

impl ShaderLayoutFactory {
    
    pub fn new(gpu: &Gpu) -> Self {
    
        todo!();
        /*


            : device_(gpu.device) 
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          device_,
          "Invalid Vulkan device!");
        */
    }
    
    pub fn invoke(&self, descriptor: &Descriptor) -> ShaderLayoutFactoryHandle {
        
        todo!();
        /*
            SmallVector<VkDescriptorSetLayoutBinding, 6u> bindings;

      u32 binding = 0u;
      for (const VkDescriptorType type : descriptor.signature) {
        bindings.push_back({
          binding++,
          type,
          1u,
          VK_SHADER_STAGE_COMPUTE_BIT,
          nullptr,
        });
      }

      const VkDescriptorSetLayoutCreateInfo descriptor_set_layout_create_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        nullptr,
        0u,
        static_cast<u32>(bindings.size()),
        bindings.data(),
      };

      VkDescriptorSetLayout descriptor_set_layout{};
      VK_CHECK(vkCreateDescriptorSetLayout(
          device_,
          &descriptor_set_layout_create_info,
          nullptr,
          &descriptor_set_layout));

      TORCH_CHECK(
          descriptor_set_layout,
          "Invalid Vulkan descriptor set layout!");

      return Handle{
        descriptor_set_layout,
        Deleter(device_),
      };
        */
    }
}

impl ShaderLayoutCache {
    
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

#[cfg(USE_VULKAN_SHADERC_RUNTIME)]
pub struct ShaderFactoryCompiler {
    context: ShadercCompiler,
    options: ShadercCompileOptions,
}

#[cfg(USE_VULKAN_SHADERC_RUNTIME)]
impl Default for ShaderFactoryCompiler {
    
    fn default() -> Self {
        todo!();
        /*


            options.SetNanClamp(/*enable =*/ true);
        options.SetSourceLanguage(shaderc_source_language_glsl);
        options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_0);
        options.SetWarningsAsErrors();
      #ifdef DEBUG
        options.SetGenerateDebugInfo();
        options.SetOptimizationLevel(shaderc_optimization_level_zero);
      #else
        options.SetOptimizationLevel(shaderc_optimization_level_performance);
      #endif /* DEBUG */
        */
    }
}

#[cfg(USE_VULKAN_SHADERC_RUNTIME)]
impl ShaderFactoryCompiler {
    
    pub fn compile(&self, source: *const u8) -> Vec<u32> {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
            source,
            "Invalid shader source code!");

        const shaderc::SpvCompilationResult result = context.CompileGlslToSpv(
            source,
            ::strlen(source),
            shaderc_compute_shader,
            "vulkan_shader.comp",
            options);

        const shaderc_compilation_status status = result.GetCompilationStatus();
        TORCH_INTERNAL_ASSERT(
            shaderc_compilation_status_success == status,
            "Shader compilation error: ",
            result.GetErrorMessage());

        return vector<u32>(result.cbegin(), result.cend());
        */
    }
}

#[cfg(not(USE_VULKAN_SHADERC_RUNTIME))]
pub struct ShaderFactoryCompiler {

}

#[cfg(not(USE_VULKAN_SHADERC_RUNTIME))]
impl ShaderFactoryCompiler {
    
    pub fn compile(&self, source: *const u8) -> Vec<u32> {
        
        todo!();
        /*
            return vector<u32>{};
        */
    }
}

impl ShaderFactory {
    
    pub fn new(gpu: &Gpu) -> Self {
    
        todo!();
        /*


            : device_(gpu.device),
       compiler_(new Compiler)
        */
    }
}

impl ShaderFactory {
    
    /**
      | unique_ptr requires its template parameter to
      | be fully defined.
      |
      | For that reason pimpl through unique_ptr
      | requires the definition of the [default]
      | constructor and move assignment operator to
      | appear after impl is fully defined.
      |
      */
    pub fn invoke(&self, descriptor: &Descriptor) -> ShaderFactoryHandle {
        
        todo!();
        /*
            vector<u32> binary;

      const u32* code = nullptr;
      u32 size = 0u;

      if (Descriptor::Type::Source == descriptor.type) {
        binary = compiler_->compile(descriptor.shader.source.glsl);
        code = binary.data();
        size = sizeof(u32) * static_cast<u32>(binary.size());
      }
      else if (Descriptor::Type::Binary == descriptor.type) {
        code = descriptor.shader.binary.spirv;
        size = descriptor.shader.binary.size;
      }
      else {
        TORCH_INTERNAL_ASSERT(false, "Invalid descriptor type!");
      }

      const VkShaderModuleCreateInfo shader_module_create_info{
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        nullptr,
        0u,
        size,
        code,
      };

      VkShaderModule shader_module{};
      VK_CHECK(vkCreateShaderModule(
          device_,
          &shader_module_create_info,
          nullptr,
          &shader_module));

      TORCH_CHECK(
          shader_module,
          "Invalid Vulkan shader module!");

      return Handle{
        shader_module,
        Deleter(device_),
      };
        */
    }
}
