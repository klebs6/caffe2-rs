crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/Vulkan.h]

#[cfg(USE_VULKAN_SHADERC_RUNTIME)]
#[macro_export] macro_rules! glsl_spv {
    ($name:ident) => {
        /*
                #name, name##_glsl
        */
    }
}

#[cfg(not(USE_VULKAN_SHADERC_RUNTIME))]
#[macro_export] macro_rules! glsl_spv {
    ($name:ident) => {
        /*
                #name, name##_spv, name##_spv_len
        */
    }
}

#[cfg(debug_assertions)]
pub const ENABLE_VALIDATION_LAYERS: bool = true;

#[cfg(not(debug_assertions))]
pub const ENABLE_VALIDATION_LAYERS: bool = false;

// VulkanTensor is a handle that holds shared
// pointer to VulkanTensor:Impl, that owns Tensor
// representation on GPU.
//
// VulkanTensor is copyable and moveable (copying
// and moving pointer to Impl).
//
// VulkanTensor::Impl is moveable only, owns
// Vulkan device memory for Tensor data. Tensor
// can be represented in several formats.
//
// 0. VBuffer - (wrapper on  vulkan VkBuffer),
// supports all tensor dimensions, data is in
// Contiguous format (NCHW), in plan to preserve
// Tensor memory format (3d or 4d tensors can be
// in NHWC ChannelsLast format). 
//
// It is located in host visible memory that can
// be memory mapped to CPU memory.
//
// 1. VImage(TexC4) - (wrapper on vulkan VkImage),
// optional representation of tensors with
// dimension <= 4 as VkImage, used in shaders as
// texture or storage image. It is 3-dimensional
// image (x, y, z) with 4 component * 16 bit for
// each triple (x, y, z).
//
// For NCHW, NHWC:
//
// For dim==4: image.x - W sizes[3]; image.y -  H sizes[2]; image.z - (N
// sizes[0] * C sizes[1]) / 4;
//
// For dim==3: image.x - W sizes[2]; image.y - H sizes[1]; image.z - (C
// sizes[0]) / 4
//
// For dim==2: image.x - W sizes[1]; image.y - H sizes[0]; image.z : 1
//
// For dim==1: image.x - W sizes[0]; image.y : 1; image.z : 1
//
// 2. VImage (other format) - Currently not added,
// but for some operations another texture packing
// format can be beneficial for performance.
//
// Contract about synchronization between
// representations:
//
// 1.VImage(TexC4) representation is allocated
// lazily with calling image(), fails for
// dimensions > 4.
//
// Tensor data can be in 0.VBuffer and/or
// 1.VImage(TexC4), If Tensor can be represented
// as image - VulkanTensor::Impl::can_be_image()
// returns true. Image representation created
// lazily by call VulkanTensor::Impl::image(), if
// it is called on Tensor with !can_be_image()
// - it fails.
//
// If image allocated - image data has
// priority. VulkanTensor::copy_data_to_host
// checks if image allocated
// - copy_image_to_buffer first.
//
pub type ImageSize = [i32; 3];

pub struct ImageSizes {
    image_size: ImageSize,
    data_size:  ImageSize,
}

#[derive(Default)]
pub struct VulkanTensor {
    impl_: Arc<Impl>,
}

impl VulkanTensor {
    
    pub fn new(sizes: Vec<i64>) -> Self {
    
        todo!();
        /*


        
        */
    }
    
    pub fn defined(&self) -> bool {
        
        todo!();
        /*
            return static_cast<bool>(impl_);
        */
    }
    
    pub fn sizes(&self) -> Vec<i64> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn strides(&self) -> Vec<i64> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn dim(&self) -> i64 {
        
        todo!();
        /*
        
        */
    }
    
    pub fn numel(&self) -> i64 {
        
        todo!();
        /*
        
        */
    }
    
    pub fn has_storage(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn allocate_storage(&mut self)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn set_data_from_host(&mut self, input_data: *const f32)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn copy_data_to_host(&self, output_data: *mut f32)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn has_buffer(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn buffer(&mut self) -> *mut VBuffer {
        
        todo!();
        /*
        
        */
    }
    
    pub fn buffer(&self) -> *const VBuffer {
        
        todo!();
        /*
        
        */
    }
    
    pub fn can_be_image(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn has_image(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn sync_image_to_buffer(&self)  {
        
        todo!();
        /*
        
        */
    }

    /**
      | if imageSizes argument is not specified:
      |
      | Allocates VImage of sizes{W,H,NC4} and fills
      | it from tensor VBuffer if it exists, see
      | comment for VulkanTensor.
      |
      | if imageSizes argument is specified:
      |
      | Only allocates VImage of specified sizes,
      | that will be returned on subsequent image()
      | calls. Can be used when user wants to store
      | tensor image not in default{W, H, NC4} format
      | (For performance or other reasons).
      |
      */
    pub fn image(&mut self, image_sizes: Option<ImageSizes>) -> *mut VImage {
        let image_sizes: Option<ImageSizes> = image_sizes.unwrap_or(nullopt);

        todo!();
        /*
        
        */
    }
    
    pub fn image(&self, image_sizes: Option<ImageSizes>) -> *const VImage {
        let image_sizes: Option<ImageSizes> = image_sizes.unwrap_or(nullopt);

        todo!();
        /*
        
        */
    }
    
    pub fn impl_(&mut self) -> Arc<Impl> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn impl_(&self) -> Arc<Impl> {
        
        todo!();
        /*
        
        */
    }
}

pub struct VContext {
    instance:                  VkInstance,
    debug_report_callback:     VkDebugReportCallbackEXT,
    device:                    VkDevice,
    physical_device:           VkPhysicalDevice,
    physical_device_limits:    VkPhysicalDeviceLimits,
    enabled_validation_layers: Vec<*const u8>,
    queue:                     VkQueue,
    queue_family_index:        u32,
    enable_validation_layers:  bool,
    command_pool:              VkCommandPool,
    compute_unit_factory:      Box<ComputeUnitFactory>,
}

impl VContext {

    pub fn new(enable_validation_layers: bool) -> Self {
    
        todo!();
        /*


        
        */
    }
    
    #[inline] pub fn device(&self) -> VkDevice {
        
        todo!();
        /*
            return device_;
        */
    }
    
    #[inline] pub fn physical_device(&self) -> VkPhysicalDevice {
        
        todo!();
        /*
            return physicalDevice_;
        */
    }
    
    #[inline] pub fn limits(&self) -> VkPhysicalDeviceLimits {
        
        todo!();
        /*
            return physicalDeviceLimits_;
        */
    }
    
    #[inline] pub fn command_pool(&self) -> VkCommandPool {
        
        todo!();
        /*
            return commandPool_;
        */
    }
    
    #[inline] pub fn queue(&self) -> VkQueue {
        
        todo!();
        /*
            return queue_;
        */
    }
    
    pub fn compute_unit_factory(&self) -> &mut ComputeUnitFactory {
        
        todo!();
        /*
            return *(computeUnitFactory_.get());
        */
    }
    
    pub fn create_instance(&mut self)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn find_physical_device(&mut self)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn create_device(&mut self)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn get_compute_queue_family_index(&mut self) -> u32 {
        
        todo!();
        /*
        
        */
    }
}

pub struct VBufferMapMemory {
    device:        VkDevice,
    device_memory: vk::DeviceMemory,
    offset:        VkDeviceSize,
    size:          VkDeviceSize,
    mapped_memory: *mut c_void,
}

impl Drop for VBufferMapMemory {

    fn drop(&mut self) {
        todo!();
        /*
            vkUnmapMemory(device_, deviceMemory_);
        */
    }
}

impl VBufferMapMemory {

    pub fn new(
        device:        VkDevice,
        device_memory: vk::DeviceMemory,
        offset:        VkDeviceSize,
        size:          VkDeviceSize) -> Self {
    
        todo!();
        /*


            : device_(device),
            deviceMemory_(deviceMemory),
            offset_(offset),
            size_(size) 
                vkMapMemory(device_, deviceMemory_, 0, size, 0, &mappedMemory_);
        */
    }
    
    #[inline] pub fn ptr(&self)  {
        
        todo!();
        /*
            return mappedMemory_;
        */
    }
    
    #[inline] pub fn ptr(&mut self)  {
        
        todo!();
        /*
            return mappedMemory_;
        */
    }
    
    pub fn flush_write_to_host(&mut self)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn flush_write_to_device(&mut self)  {
        
        todo!();
        /*
        
        */
    }
}

pub struct VBuffer {
    buffer_size_bytes: VkDeviceSize,
    descriptor_type:   VkDescriptorType,
    buffer:            vk::Buffer,
    buffer_memory:     vk::DeviceMemory,
}

impl VBuffer {
    
    pub fn new(
        buffer_size_bytes:  VkDeviceSize,
        buffer_usage_flags: VkBufferUsageFlags,
        descriptor_type:    VkDescriptorType) -> Self {

        let buffer_usage_flags: VkBufferUsageFlags = buffer_usage_flags.unwrap_or(
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT 
            | VK_BUFFER_USAGE_TRANSFER_SRC_BIT 
            | VK_BUFFER_USAGE_TRANSFER_DST_BIT
        );

        let descriptor_type: VkDescriptorType = descriptor_type.unwrap_or(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

        todo!();
        /*


        
        */
    }
    
    #[inline] pub fn make_uniform_buffer(buffer_size: VkDeviceSize) -> VBuffer {
        
        todo!();
        /*
            return VBuffer{bufferSize,
                       VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                       VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
        */
    }
    
    pub fn map(&self) -> MapMemory {
        
        todo!();
        /*
            return MapMemory{context().device(), bufferMemory_, 0, bufferSizeBytes_};
        */
    }
    
    pub fn copy_from_device_to_host(&self, 
        output_data: *mut c_void,
        size:        i64)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn copy_from_host_to_device(&mut self, 
        data: *const c_void,
        size: i64)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn set_zeros(&mut self)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn make_descriptor_buffer_info(&self) -> VkDescriptorBufferInfo {
        
        todo!();
        /*
        
        */
    }
    
    pub fn make_write_descriptor_set(&self, 
        descriptor_set: VkDescriptorSet,
        binding:        u32,
        buffer_info:    *const VkDescriptorBufferInfo) -> VkWriteDescriptorSet {
        
        todo!();
        /*
        
        */
    }
    
    pub fn bind(&self, 
        descriptor_set: VkDescriptorSet,
        binding:        u32)  {
        
        todo!();
        /*
        
        */
    }
    
    #[inline] pub fn size_bytes(&self) -> VkDeviceSize {
        
        todo!();
        /*
            return bufferSizeBytes_;
        */
    }
    
    pub fn add_buffer_memory_barrier(&self, 
        command_buffer: VkCommandBuffer,
        offset:         VkDeviceSize,
        size:           VkDeviceSize)  {
        
        todo!();
        /*
        
        */
    }
    
    #[inline] pub fn vkbuffer(&self) -> vk::Buffer {
        
        todo!();
        /*
            return buffer_;
        */
    }
}

pub struct VImage {

    image_size:   ImageSize,
    data_size:    ImageSize,
    image:        VkImage,
    image_memory: vk::DeviceMemory,
    image_view:   VkImageView,
    sampler:      VkSampler,

    /**
      | Holds current image layout that will
      | be used in addImageMemoryBarrier as
      | the previous layout. Need to be mutable
      | to use addImageMemoryBarrier() for
      | const VImage.
      |
      */
    image_layout: RefCell<VkImageLayout>,
}

pub mod v_image {

    use super::*;

    pub const IMAGE_TYPE:           VkImageType          = VK_IMAGE_TYPE_3D;
    pub const FILTER:               VkFilter             = VK_FILTER_NEAREST;
    pub const FORMAT:               VkFormat             = VK_FORMAT_R16G16B16A16_SFLOAT;
    pub const SAMPLER_ADDRESS_MODE: VkSamplerAddressMode = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    pub const IMAGE_VIEW_TYPE:      VkImageViewType      = VK_IMAGE_VIEW_TYPE_3D;
}

impl VImage {
    
    pub fn new(image_sizes: ImageSizes) -> Self {
    
        todo!();
        /*
            : VImage(imageSizes.imageSize, imageSizes.dataSize)
        */
    }
    
    #[inline] pub fn w(&self) -> Auto {
        
        todo!();
        /*
            return imageSize_[0];
        */
    }
    
    #[inline] pub fn h(&self) -> Auto {
        
        todo!();
        /*
            return imageSize_[1];
        */
    }
    
    #[inline] pub fn d(&self) -> Auto {
        
        todo!();
        /*
            return imageSize_[2];
        */
    }
   
    #[inline] pub fn size_bytes(&self) -> VkDeviceSize {
        
        todo!();
        /*
            return sizeof(float) * dataSize_[0] * dataSize_[1] * dataSize_[2];
        */
    }
    
    #[inline] pub fn capacity_bytes(&self) -> VkDeviceSize {
        
        todo!();
        /*
            // Every VImage pixel(texel) contains 4 float elements
        return sizeof(float) * 4 * imageSize_[0] * imageSize_[1] * imageSize_[2];
        */
    }
    
    pub fn sizes(&self) -> ImageSize {
        
        todo!();
        /*
            return imageSize_;
        */
    }
}

pub struct WorkGroupSize {
    x: u32,
    y: u32,
    z: u32,
}

pub struct ComputeUnit {
    command_buffer:        VkCommandBuffer,
    pipeline:              VkPipeline,
    pipeline_layout:       VkPipelineLayout,
    compute_shader_module: vk::ShaderModule,
}

pub mod compute_unit {

    pub const FENCE_TIMEOUT_NANOS: u64 = 100000000000;
}

impl ComputeUnit {

    #[cfg(USE_VULKAN_SHADERC_RUNTIME)]
    pub fn new(
        glsl_src:         *const u8,
        pipeline_cache:   VkPipelineCache,
        descr_set_layout: VkDescriptorSetLayout,
        work_group_size:  WorkGroupSize) -> Self {
    
        todo!();
        /*


            createComputePipelineCompile(
            glslSrc, pipelineCache, descrSetLayout, workGroupSize);
        */
    }

    #[cfg(not(USE_VULKAN_SHADERC_RUNTIME))]
    pub fn new(
        spv_code:         *const u32,
        spv_code_size:    u32,
        pipeline_cache:   VkPipelineCache,
        descr_set_layout: &VkDescriptorSetLayout,
        work_group_size:  WorkGroupSize) -> Self {
    
        todo!();
        /*


            const auto codeSize = spvCodeSize;
        createComputePipeline(
            spvCode, codeSize, pipelineCache, descrSetLayout, workGroupSize);
        */
    }
    
    pub fn create_compute_pipeline(&mut self, 
        code:             *const u32,
        code_size:        u32,
        pipeline_cache:   VkPipelineCache,
        descr_set_layout: VkDescriptorSetLayout,
        work_group_size:  WorkGroupSize)  {
        
        todo!();
        /*
        
        */
    }

    #[cfg(USE_VULKAN_SHADERC_RUNTIME)]
    pub fn create_compute_pipeline_compile(&mut self, 
        glsl_src:         &String,
        pipeline_cache:   VkPipelineCache,
        descr_set_layout: VkDescriptorSetLayout,
        work_group_size:  WorkGroupSize)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn create_command_buffer(&mut self, descriptor_set: &mut VkDescriptorSet)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn add_memory_barrier(&mut self, 
        src_stage_mask:  VkPipelineStageFlags,
        src_access_mask: VkAccessFlags,
        dst_stage_mask:  VkPipelineStageFlags,
        dst_access_mask: VkAccessFlags)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn dispatch_command_buffer(&mut self, 
        group_countx: u32,
        group_county: u32,
        group_countz: u32)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn dispatch_command_buffer(&mut self, 
        gridx:           u32,
        gridy:           u32,
        gridz:           u32,
        work_group_size: WorkGroupSize)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn submit_and_wait_command_buffer(&mut self)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn end_command_buffer(&mut self)  {
        
        todo!();
        /*
        
        */
    }
    
    #[inline] pub fn command_buffer(&mut self) -> VkCommandBuffer {
        
        todo!();
        /*
            return commandBuffer_;
        */
    }
}

pub struct ComputeUnitFactory {
    device:         VkDevice,
    pipeline_cache: VkPipelineCache,
    compute_units:  HashMap<String,Arc<ComputeUnit>>,
}

impl ComputeUnitFactory {
    
    pub fn new(device: VkDevice) -> Self {
    
        todo!();
        /*


        
        */
    }

    #[cfg(USE_VULKAN_SHADERC_RUNTIME)]
    pub fn get(&mut self, 
        key:              *const u8,
        glsl_src:         *const u8,
        descr_set_layout: VkDescriptorSetLayout,
        work_group_size:  WorkGroupSize) -> &mut ComputeUnit {
        
        todo!();
        /*
        
        */
    }
    
    #[cfg(not(USE_VULKAN_SHADERC_RUNTIME))]
    pub fn get(&mut self, 
        key:              *const u8,
        code:             *const u32,
        code_size:        u32,
        descr_set_layout: VkDescriptorSetLayout,
        work_group_size:  WorkGroupSize) -> &mut ComputeUnit {
        
        todo!();
        /*
        
        */
    }
    
    pub fn get_cache_key(&mut self, 
        key:             *const u8,
        work_group_size: WorkGroupSize) -> String {
        
        todo!();
        /*
        
        */
    }
    
    pub fn get(&mut self, 
        cache_key:  &String,
        factory_fn: fn() -> Arc<ComputeUnit>) -> &mut ComputeUnit {
        
        todo!();
        /*
        
        */
    }
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/Vulkan.cpp]

macro_rules! vk_check {
    ($f:ident) => {
        /*
        
          {                                                                
            VkResult res = (f);                                            
            TORCH_CHECK(res == VK_SUCCESS, "Vulkan error VkResult:", res); 
          }
        */
    }
}

impl VContext {
    
    pub fn new(enable_validation_layers: bool) -> Self {
    
        todo!();
        /*


            : enableValidationLayers_(enableValidationLayers) 
      createInstance();
      findPhysicalDevice();
      createDevice();

      computeUnitFactory_ = make_unique<ComputeUnitFactory>(device_);
        */
    }
}

impl Drop for VContext {

    fn drop(&mut self) {
        todo!();
        /*
            if (enableValidationLayers_) {
        const auto func = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(
            instance_, "vkDestroyDebugReportCallbackEXT");
        if (func) {
          func(instance_, debugReportCallback_, nullptr);
        }
      }

      // ComputeUnitFactory_ owns ComputeUnits and VkPipelineCache, need valid
      // VkDevice for destructing, destructing before vkDestroyDevice
      computeUnitFactory_.reset();

      vkDestroyCommandPool(device_, commandPool_, nullptr);
      vkDestroyDevice(device_, nullptr);
      vkDestroyInstance(instance_, nullptr);
        */
    }
}

pub fn debug_report_callback_fn(
        msg_flags:      vk::DebugReportFlagsEXT,
        object_type:    vk::DebugReportObjectTypeEXT,
        object:         u64,
        location:       usize,
        msg_code:       i32,
        p_layer_prefix: *const u8,
        p_msg:          *const u8,
        p_user_data:    *mut c_void) -> vk::Bool32 {
    
    todo!();
        /*
            stringstream s;
      s << pLayerPrefix << " " << msgCode << " " << pMsg << endl;
      if (msgFlags & VK_DEBUG_REPORT_ERROR_BIT_EXT) {
        LOG(ERROR) << s.str();
      } else if (msgFlags & VK_DEBUG_REPORT_WARNING_BIT_EXT) {
        LOG(WARNING) << "WARN:" << s.str();
      } else if (msgFlags & VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT) {
        LOG(WARNING) << "PERF_WARN:" << s.str();
      } else if (msgFlags & VK_DEBUG_REPORT_INFORMATION_BIT_EXT) {
        LOG(INFO) << s.str();
      }
      return VK_FALSE;
        */
}

impl VContext {
    
    pub fn create_instance(&mut self)  {
        
        todo!();
        /*
            vector<const char*> enabledExtensions;
      if (enableValidationLayers_) {
        u32 layerPresentCount = 0;
        VK_CHECK(vkEnumerateInstanceLayerProperties(&layerPresentCount, nullptr));
        vector<VkLayerProperties> layerProps(layerPresentCount);
        VK_CHECK(vkEnumerateInstanceLayerProperties(&layerPresentCount, layerProps.data()));
        array<const char*, 6> instanceLayers{
            "VK_LAYER_GOOGLE_unique_objects",
            "VK_LAYER_GOOGLE_threading",
            "VK_LAYER_LUNARG_object_tracker",
            "VK_LAYER_LUNARG_core_validation",
            "VK_LAYER_LUNARG_parameter_validation",
            "VK_LAYER_KHRONOS_validation",
        };

        for (const auto& wantedLayer : instanceLayers) {
          for (const auto& presentLayer : layerProps) {
            if (strcmp(wantedLayer, presentLayer.layerName) == 0) {
              enabledValidationLayers_.push_back(wantedLayer);
              break;
            }
          }
        }

        u32 extCount = 0;
        VK_CHECK(vkEnumerateInstanceExtensionProperties(nullptr, &extCount, nullptr));
        vector<VkExtensionProperties> extProps(extCount);
        VK_CHECK(vkEnumerateInstanceExtensionProperties(nullptr, &extCount, extProps.data()));
        bool foundExt = false;
        for (VkExtensionProperties p : extProps) {
          if (strcmp(VK_EXT_DEBUG_REPORT_EXTENSION_NAME, p.extensionName) == 0) {
            foundExt = true;
            break;
          }
        }
        if (foundExt) {
          enabledExtensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
        }
      }

      VkApplicationInfo applicationInfo{};
      applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
      applicationInfo.pApplicationName = "PyTorch";
      applicationInfo.applicationVersion = 0;
      applicationInfo.pEngineName = "PyTorch";
      applicationInfo.engineVersion = 0;
      applicationInfo.apiVersion = VK_API_VERSION_1_0;

      VkInstanceCreateInfo createInfo{};
      createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
      createInfo.flags = 0;
      createInfo.pApplicationInfo = &applicationInfo;
      createInfo.enabledLayerCount = enabledValidationLayers_.size();
      createInfo.ppEnabledLayerNames = enabledValidationLayers_.data();
      createInfo.enabledExtensionCount = enabledExtensions.size();
      createInfo.ppEnabledExtensionNames = enabledExtensions.data();

      VK_CHECK(vkCreateInstance(&createInfo, nullptr, &instance_));

      if (enableValidationLayers_) {
        VkDebugReportCallbackCreateInfoEXT debugReportCallbackCreateInfo{};
        debugReportCallbackCreateInfo.sType =
            VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
        debugReportCallbackCreateInfo.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT |
            VK_DEBUG_REPORT_WARNING_BIT_EXT |
            VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT;
        debugReportCallbackCreateInfo.pfnCallback = &debugReportCallbackFn;

        const auto vkCreateDebugReportCallbackEXT =
            (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(
                instance_, "vkCreateDebugReportCallbackEXT");
        TORCH_CHECK(
            vkCreateDebugReportCallbackEXT,
            "Could not load vkCreateDebugReportCallbackEXT");
        VK_CHECK(vkCreateDebugReportCallbackEXT(
            instance_,
            &debugReportCallbackCreateInfo,
            nullptr,
            &debugReportCallback_));
      }
        */
    }
    
    pub fn find_physical_device(&mut self)  {
        
        todo!();
        /*
            u32 deviceCount = 0;
      VK_CHECK(vkEnumeratePhysicalDevices(instance_, &deviceCount, nullptr));
      TORCH_CHECK(
          deviceCount > 0, "Vulkan: Could not find a device with vulkan support");
      vector<VkPhysicalDevice> devices(deviceCount);
      VK_CHECK(vkEnumeratePhysicalDevices(instance_, &deviceCount, devices.data()));
      physicalDevice_ = devices[0];
        */
    }
    
    pub fn get_compute_queue_family_index(&mut self) -> u32 {
        
        todo!();
        /*
            u32 queueFamilyCount = 0;

      vkGetPhysicalDeviceQueueFamilyProperties(
          physicalDevice_, &queueFamilyCount, nullptr);
      TORCH_CHECK(
          queueFamilyCount > 0, "Vulkan: Invalid number of queue families");
      vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
      vkGetPhysicalDeviceQueueFamilyProperties(
          physicalDevice_, &queueFamilyCount, queueFamilies.data());

      for (u32 i = 0; i < queueFamilies.size(); ++i) {
        VkQueueFamilyProperties props = queueFamilies[i];
        if (props.queueCount > 0 && (props.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
          return i;
        }
      }

      TORCH_CHECK(
          false, "Vulkan: Could not find a queue family that supports operations");
        */
    }
    
    pub fn create_device(&mut self)  {
        
        todo!();
        /*
            VkDeviceQueueCreateInfo queueCreateInfo{};
      queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
      queueFamilyIndex_ = getComputeQueueFamilyIndex();
      queueCreateInfo.queueFamilyIndex = queueFamilyIndex_;
      queueCreateInfo.queueCount = 1;
      const float queuePriorities = 1.0f;
      queueCreateInfo.pQueuePriorities = &queuePriorities;
      VkDeviceCreateInfo deviceCreateInfo{};
      VkPhysicalDeviceFeatures deviceFeatures{};

      deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
      deviceCreateInfo.enabledLayerCount = enabledValidationLayers_.size();
      deviceCreateInfo.ppEnabledLayerNames = enabledValidationLayers_.data();
      deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;

      deviceCreateInfo.queueCreateInfoCount = 1;
      deviceCreateInfo.pEnabledFeatures = &deviceFeatures;

      VK_CHECK(
          vkCreateDevice(physicalDevice_, &deviceCreateInfo, nullptr, &device_));
      queue_ = {};
      vkGetDeviceQueue(device_, queueFamilyIndex_, 0, &queue_);

      VkPhysicalDeviceProperties physicalDeviceProperties{};
      vkGetPhysicalDeviceProperties(physicalDevice_, &physicalDeviceProperties);

      VkCommandPoolCreateInfo commandPoolCreateInfo{};
      commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
      commandPoolCreateInfo.flags = 0;
      commandPoolCreateInfo.queueFamilyIndex = queueFamilyIndex_;
      VK_CHECK(vkCreateCommandPool(
          device_, &commandPoolCreateInfo, nullptr, &commandPool_));
      physicalDeviceLimits_ = physicalDeviceProperties.limits;
        */
    }
}

lazy_static!{
    /*
    static unique_ptr<VContext> gContext;
    */
}

pub fn context() -> &VContext {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(gContext);
      return *gContext;
        */
}

pub fn init_vulkan_context_once() -> bool {
    
    todo!();
        /*
            static const int once = []() {
    #ifdef USE_VULKAN_WRAPPER
        if (!InitVulkan()) {
          TORCH_WARN("Vulkan Wrapper Failed to InitVulkan");
          return 1;
        }
    #endif
        gContext = make_unique<VContext>(kEnableValidationLayers);
        if (!gContext) {
          TORCH_WARN("Vulkan Failed to create Vulkan Context");
          return 2;
        }
        return 0;
      }();
      ((void)once);
      return static_cast<bool>(gContext);
        */
}

pub fn is_available() -> bool {
    
    todo!();
        /*
            return initVulkanContextOnce();
        */
}

pub fn find_memory_type(
        physical_device:  VkPhysicalDevice,
        memory_type_bits: u32,
        properties:       VkMemoryPropertyFlags) -> u32 {
    
    todo!();
        /*
            VkPhysicalDeviceMemoryProperties memoryProperties{};
      vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
      for (u32 i = 0; i < memoryProperties.memoryTypeCount; ++i) {
        if ((memoryTypeBits & (1 << i)) &&
            ((memoryProperties.memoryTypes[i].propertyFlags & properties) ==
             properties)) {
          return i;
        }
      }
      return -1;
        */
}

impl VBufferMapMemory {
    
    pub fn flush_write_to_device(&mut self)  {
        
        todo!();
        /*
            VkMappedMemoryRange range{};
      range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
      range.memory = deviceMemory_;
      range.offset = offset_;
      range.size = size_;
      range.pNext = nullptr;

      VK_CHECK(vkFlushMappedMemoryRanges(context().device(), 1, &range));
        */
    }
    
    pub fn flush_write_to_host(&mut self)  {
        
        todo!();
        /*
            VkMappedMemoryRange range{};
      range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
      range.memory = deviceMemory_;
      range.offset = offset_;
      range.size = size_;
      range.pNext = nullptr;

      VK_CHECK(vkInvalidateMappedMemoryRanges(context().device(), 1, &range));
        */
    }
}

impl Drop for VBuffer {

    fn drop(&mut self) {
        todo!();
        /*
            vkFreeMemory(context().device(), bufferMemory_, nullptr);
      vkDestroyBuffer(context().device(), buffer_, nullptr);
        */
    }
}

impl VBuffer {
    
    pub fn new(
        buffer_size_bytes:  VkDeviceSize,
        buffer_usage_flags: VkBufferUsageFlags,
        descriptor_type:    VkDescriptorType) -> Self {
    
        todo!();
        /*


            : bufferSizeBytes_(bufferSizeBytes), descriptorType_(descriptorType) 

      const auto device = context().device();
      const auto physicalDevice = context().physicalDevice();
      VkBufferCreateInfo bufferCreateInfo{};
      bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
      bufferCreateInfo.size = bufferSizeBytes_;
      bufferCreateInfo.usage = bufferUsageFlags;
      bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      VK_CHECK(vkCreateBuffer(device, &bufferCreateInfo, nullptr, &buffer_));
      VkMemoryRequirements memoryRequirements;
      vkGetBufferMemoryRequirements(device, buffer_, &memoryRequirements);
      VkMemoryAllocateInfo allocateInfo{};
      allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
      allocateInfo.allocationSize = memoryRequirements.size;
      allocateInfo.memoryTypeIndex = findMemoryType(
          physicalDevice,
          memoryRequirements.memoryTypeBits,
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
      VK_CHECK(vkAllocateMemory(device, &allocateInfo, nullptr, &bufferMemory_));
      VK_CHECK(vkBindBufferMemory(device, buffer_, bufferMemory_, 0));
        */
    }
    
    pub fn copy_from_device_to_host(&self, 
        output_data: *mut c_void,
        size:        i64)  {
        
        todo!();
        /*
            auto mm = map();
      TORCH_INTERNAL_ASSERT(mm.ptr(), "Vulkan: Failed to map Vulkan Buffer memory");
      ::memcpy(outputData, mm.ptr(), size);
      mm.flushWriteToHost();
        */
    }
    
    pub fn copy_from_host_to_device(&mut self, 
        data: *const c_void,
        size: i64)  {
        
        todo!();
        /*
            auto mm = map();
      TORCH_INTERNAL_ASSERT(mm.ptr(), "Vulkan: Failed to map Vulkan Buffer memory");
      ::memcpy(mm.ptr(), data, size);
      mm.flushWriteToDevice();
        */
    }
    
    pub fn set_zeros(&mut self)  {
        
        todo!();
        /*
            auto mm = map();
      TORCH_INTERNAL_ASSERT(mm.ptr(), "Vulkan: Failed to map Vulkan Buffer memory");
      ::memset(mm.ptr(), 0, bufferSizeBytes_);
        */
    }
    
    pub fn make_descriptor_buffer_info(&self) -> VkDescriptorBufferInfo {
        
        todo!();
        /*
            VkDescriptorBufferInfo info{};
      info.buffer = buffer_;
      info.offset = 0;
      info.range = bufferSizeBytes_;
      return info;
        */
    }
    
    pub fn make_write_descriptor_set(&self, 
        descriptor_set: VkDescriptorSet,
        binding:        u32,
        buffer_info:    *const VkDescriptorBufferInfo) -> VkWriteDescriptorSet {
        
        todo!();
        /*
            VkWriteDescriptorSet writeSet{};
      writeSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writeSet.pNext = nullptr;
      writeSet.dstSet = descriptorSet;
      writeSet.dstBinding = binding;
      writeSet.dstArrayElement = 0;
      writeSet.descriptorCount = 1;
      writeSet.descriptorType = descriptorType_;
      writeSet.pImageInfo = nullptr;
      writeSet.pBufferInfo = bufferInfo;
      writeSet.pTexelBufferView = nullptr;
      return writeSet;
        */
    }
    
    pub fn bind(&self, 
        descriptor_set: VkDescriptorSet,
        binding:        u32)  {
        
        todo!();
        /*
            const auto descrBufferInfo = makeDescriptorBufferInfo();
      const auto writeDescrSet =
          makeWriteDescriptorSet(descriptorSet, binding, &descrBufferInfo);
      vkUpdateDescriptorSets(context().device(), 1, &writeDescrSet, 0, nullptr);
        */
    }
    
    pub fn add_buffer_memory_barrier(&self, 
        command_buffer: VkCommandBuffer,
        offset:         VkDeviceSize,
        size:           VkDeviceSize)  {
        
        todo!();
        /*
            VkBufferMemoryBarrier barrier{};
      barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
      barrier.buffer = buffer_;
      barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.offset = offset;
      barrier.pNext = nullptr;
      barrier.size = size;
      barrier.srcAccessMask =
          VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
      barrier.dstAccessMask =
          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;

      vkCmdPipelineBarrier(
          commandBuffer,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
          0,
          0,
          nullptr,
          1,
          &barrier,
          0,
          nullptr);
        */
    }
}

impl Drop for VImage {

    fn drop(&mut self) {
        todo!();
        /*
            vkFreeMemory(context().device(), imageMemory_, nullptr);
      vkDestroySampler(context().device(), sampler_, nullptr);
      vkDestroyImageView(context().device(), imageView_, nullptr);
      vkDestroyImage(context().device(), image_, nullptr);
        */
    }
}

impl VImage {
    
    pub fn new(
        image_size: ImageSize,
        data_size:  ImageSize) -> Self {
    
        todo!();
        /*


            : imageSize_(imageSize), dataSize_(dataSize) 

      const auto device = context().device();
      const auto physicalDevice = context().physicalDevice();

      VkImageCreateInfo imageInfo{};
      imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
      imageInfo.imageType = kImageType;
      imageInfo.extent.width = imageSize_[0];
      imageInfo.extent.height = imageSize_[1];
      imageInfo.extent.depth = imageSize_[2];

      imageInfo.mipLevels = 1;
      imageInfo.arrayLayers = 1;
      imageInfo.format = kFormat;
      imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
      imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
      imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
      imageInfo.pNext = nullptr;
      imageInfo.flags = 0;
      imageLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;

      VK_CHECK(vkCreateImage(device, &imageInfo, nullptr, &image_));

      VkMemoryRequirements memReqs{};
      vkGetImageMemoryRequirements(device, image_, &memReqs);
      VkMemoryAllocateInfo allocInfo{};
      allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
      allocInfo.allocationSize = memReqs.size;
      allocInfo.memoryTypeIndex = findMemoryType(
          physicalDevice,
          memReqs.memoryTypeBits,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

      VK_CHECK(vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory_));
      VK_CHECK(vkBindImageMemory(device, image_, imageMemory_, 0));

      const VkImageViewCreateInfo imageViewCreateInfo = makeImageViewCreateInfo();
      VK_CHECK(
          vkCreateImageView(device, &imageViewCreateInfo, nullptr, &imageView_));

      const VkSamplerCreateInfo samplerCreateInfo = makeSamplerCreateInfo();
      VK_CHECK(vkCreateSampler(device, &samplerCreateInfo, nullptr, &sampler_));
        */
    }
    
    pub fn make_image_view_create_info(&self) -> VkImageViewCreateInfo {
        
        todo!();
        /*
            VkImageViewCreateInfo info{};
      info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      info.image = image_;
      info.viewType = kImageViewType;
      info.format = kFormat;
      info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      info.subresourceRange.baseMipLevel = 0;
      info.subresourceRange.levelCount = 1;
      info.subresourceRange.baseArrayLayer = 0;
      info.subresourceRange.layerCount = 1;
      return info;
        */
    }
    
    pub fn make_sampler_create_info(&self) -> VkSamplerCreateInfo {
        
        todo!();
        /*
            VkSamplerCreateInfo info{};
      info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
      info.magFilter = kFilter;
      info.minFilter = kFilter;
      info.addressModeU = kSamplerAddressMode;
      info.addressModeV = kSamplerAddressMode;
      info.addressModeW = kSamplerAddressMode;
      info.anisotropyEnable = VK_FALSE;
      info.maxAnisotropy = 1.0f;
      info.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
      info.compareEnable = VK_FALSE;
      info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
      info.mipLodBias = 0.0f;
      info.minLod = 0.0f;
      info.maxLod = 0.0f;
      return info;
        */
    }
    
    pub fn make_descriptor_image_info(&self, image_layout: VkImageLayout) -> VkDescriptorImageInfo {
        
        todo!();
        /*
            VkDescriptorImageInfo info{};
      info.sampler = sampler_;
      info.imageView = imageView_;
      info.imageLayout = imageLayout;
      return info;
        */
    }
    
    pub fn make_write_descriptor_set(&self, 
        descriptor_set:  VkDescriptorSet,
        binding:         u32,
        descriptor_type: VkDescriptorType,
        image_info:      *const VkDescriptorImageInfo) -> VkWriteDescriptorSet {
        
        todo!();
        /*
            VkWriteDescriptorSet writeSet{};
      writeSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writeSet.pNext = nullptr;
      writeSet.dstSet = descriptorSet;
      writeSet.dstBinding = binding;
      writeSet.dstArrayElement = 0;
      writeSet.descriptorCount = 1;
      writeSet.descriptorType = descriptorType, writeSet.pImageInfo = imageInfo;
      writeSet.pBufferInfo = nullptr;
      writeSet.pTexelBufferView = nullptr;
      return writeSet;
        */
    }
    
    pub fn bind(&self, 
        descriptor_set:  VkDescriptorSet,
        binding:         u32,
        descriptor_type: VkDescriptorType,
        image_layout:    VkImageLayout)  {
        
        todo!();
        /*
            const auto descrImageInfo = makeDescriptorImageInfo(imageLayout);
      const auto writeDescrSet = makeWriteDescriptorSet(
          descriptorSet, binding, descriptorType, &descrImageInfo);
      vkUpdateDescriptorSets(context().device(), 1, &writeDescrSet, 0, nullptr);
        */
    }
    
    pub fn bind_shader_read(&self, 
        descriptor_set: VkDescriptorSet,
        binding:        u32)  {
        
        todo!();
        /*
            bind(
          descriptorSet,
          binding,
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        */
    }
    
    pub fn bind_storage_image(&self, 
        descriptor_set: VkDescriptorSet,
        binding:        u32)  {
        
        todo!();
        /*
            bind(
          descriptorSet,
          binding,
          VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          VK_IMAGE_LAYOUT_GENERAL);
        */
    }
    
    pub fn add_image_memory_barrier(&self, 
        command_buffer: VkCommandBuffer,
        new_layout:     VkImageLayout)  {
        
        todo!();
        /*
            const VkImageLayout oldLayout = imageLayout_;
      if (oldLayout == newLayout) {
        return;
      }

      VkImageMemoryBarrier barrier{};
      barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.image = image_;
      barrier.newLayout = newLayout;
      barrier.oldLayout = oldLayout;
      barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      barrier.subresourceRange.levelCount = 1;
      barrier.subresourceRange.layerCount = 1;

      barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

      VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
          newLayout == VK_IMAGE_LAYOUT_GENERAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
      } else if (
          oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
          newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      } else if (
          oldLayout == VK_IMAGE_LAYOUT_GENERAL &&
          newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      } else if (
          oldLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL &&
          newLayout == VK_IMAGE_LAYOUT_GENERAL) {
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      } else {
        TORCH_INTERNAL_ASSERT(
            false, "Vulkan: Unsupported Vulkan Image Layout transition");
      }
      vkCmdPipelineBarrier(
          commandBuffer,
          srcStageMask,
          dstStageMask,
          0,
          0,
          nullptr,
          0,
          nullptr,
          1,
          &barrier);
      imageLayout_ = newLayout;
        */
    }
    
    pub fn add_image_memory_barrier_to_general(&self, command_buffer: VkCommandBuffer)  {
        
        todo!();
        /*
            addImageMemoryBarrier(commandBuffer, VK_IMAGE_LAYOUT_GENERAL);
        */
    }
    
    pub fn add_image_memory_barrier_to_shader_read(&self, command_buffer: VkCommandBuffer)  {
        
        todo!();
        /*
            addImageMemoryBarrier(
          commandBuffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        */
    }
}

pub fn descriptor_set_layout_binding(
        binding:         u32,
        descriptor_type: VkDescriptorType) -> VkDescriptorSetLayoutBinding {
    
    todo!();
        /*
            return {binding, descriptorType, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        */
}

pub fn create_descriptor_set_layout(
        device:        VkDevice,
        bindings:      *const VkDescriptorSetLayoutBinding,
        binding_count: u32,
        set_layout:    *mut VkDescriptorSetLayout)  {
    
    todo!();
        /*
            VkDescriptorSetLayoutCreateInfo createInfo{};
      createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
      createInfo.pNext = nullptr;
      createInfo.flags = 0;
      createInfo.bindingCount = bindingCount;
      createInfo.pBindings = bindings;
      VK_CHECK(
          vkCreateDescriptorSetLayout(device, &createInfo, nullptr, setLayout));
        */
}

pub fn create_descriptor_pool(
        device:          VkDevice,
        pool_sizes:      *const VkDescriptorPoolSize,
        pool_size_count: u32,
        max_sets:        u32,
        descriptor_pool: *mut VkDescriptorPool)  {
    
    todo!();
        /*
            VkDescriptorPoolCreateInfo createInfo{};
      createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
      createInfo.pNext = nullptr;
      createInfo.flags = 0;
      createInfo.maxSets = maxSets;
      createInfo.poolSizeCount = poolSizeCount;
      createInfo.pPoolSizes = poolSizes;
      VK_CHECK(
          vkCreateDescriptorPool(device, &createInfo, nullptr, descriptorPool));
        */
}


pub fn allocate_descriptor_set(
        device:                VkDevice,
        descriptor_pool:       VkDescriptorPool,
        descriptor_set_layout: *const VkDescriptorSetLayout,
        descriptor_set:        *mut VkDescriptorSet)  {
    
    todo!();
        /*
            VkDescriptorSetAllocateInfo allocateInfo{};
      allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
      allocateInfo.pNext = nullptr;
      allocateInfo.descriptorPool = descriptorPool;
      allocateInfo.descriptorSetCount = 1;
      allocateInfo.pSetLayouts = descriptorSetLayout;
      VK_CHECK(vkAllocateDescriptorSets(device, &allocateInfo, descriptorSet));
        */
}

pub fn create_descriptor_set_layout_single_pool(
        device:           VkDevice,
        descr_types:      &Vec<VkDescriptorType>,
        descr_set_layout: *mut VkDescriptorSetLayout,
        descr_pool:       *mut VkDescriptorPool,
        descr_set:        *mut VkDescriptorSet)  {
    
    todo!();
        /*
            const auto size = descrTypes.size();
      vector<VkDescriptorSetLayoutBinding> bindings;
      vector<VkDescriptorPoolSize> poolSizes;
      u32 i = 0;
      for (const auto& descrType : descrTypes) {
        bindings.push_back(descriptorSetLayoutBinding(i, descrType));
        poolSizes.push_back(VkDescriptorPoolSize{descrType, 1});
        i++;
      }
      createDescriptorSetLayout(device, bindings.data(), size, descrSetLayout);
      createDescriptorPool(
          device, poolSizes.data(), size, 1 /* maxSets */, descrPool);
      allocateDescriptorSet(device, *descrPool, descrSetLayout, descrSet);
        */
}

pub fn allocate_command_buffer(
        device:         VkDevice,
        command_buffer: *mut VkCommandBuffer)  {
    
    todo!();
        /*
            VkCommandBufferAllocateInfo commandBufferAllocateInfo{};
      commandBufferAllocateInfo.sType =
          VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
      commandBufferAllocateInfo.commandPool = context().commandPool();
      commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      commandBufferAllocateInfo.commandBufferCount = 1;

      VK_CHECK(vkAllocateCommandBuffers(
          device, &commandBufferAllocateInfo, commandBuffer));
        */
}

pub fn begin_command_buffer(command_buffer: VkCommandBuffer)  {
    
    todo!();
        /*
            VkCommandBufferBeginInfo beginInfo{};
      beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));
        */
}

pub fn end_command_buffer(command_buffer: VkCommandBuffer)  {
    
    todo!();
        /*
            VK_CHECK(vkEndCommandBuffer(commandBuffer));
        */
}

pub fn submit_and_wait_command_buffer(
        device:         VkDevice,
        queue:          VkQueue,
        command_buffer: VkCommandBuffer)  {
    
    todo!();
        /*
            VkSubmitInfo submitInfo{};
      submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
      submitInfo.commandBufferCount = 1;
      submitInfo.pCommandBuffers = &commandBuffer;

      VkFence fence;
      VkFenceCreateInfo fenceCreateInfo{};
      fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
      fenceCreateInfo.flags = 0;
      VK_CHECK(vkCreateFence(device, &fenceCreateInfo, NULL, &fence))

      VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, fence));
      vkWaitForFences(device, 1, &fence, VK_TRUE, ComputeUnit::kFenceTimeoutNanos);

      vkDestroyFence(device, fence, NULL);
        */
}

impl Drop for ComputeUnit {

    fn drop(&mut self) {
        todo!();
        /*
            vkDestroyShaderModule(context().device(), computeShaderModule_, nullptr);
      vkDestroyPipelineLayout(context().device(), pipelineLayout_, nullptr);
      vkDestroyPipeline(context().device(), pipeline_, nullptr);
        */
    }
}

impl ComputeUnit {
    
    pub fn create_compute_pipeline(&mut self, 
        code:             *const u32,
        code_size:        u32,
        pipeline_cache:   VkPipelineCache,
        descr_set_layout: VkDescriptorSetLayout,
        work_group_size:  WorkGroupSize)  {
        
        todo!();
        /*
            const auto device = context().device();
      VkShaderModuleCreateInfo createInfo{};
      createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
      createInfo.pCode = code;
      createInfo.codeSize = codeSize;

      VK_CHECK(vkCreateShaderModule(
          device, &createInfo, nullptr, &computeShaderModule_));

      VkSpecializationMapEntry spMapEntries[3];
      {
        u32 offset = 0;
        usize size = sizeof(WorkGroupSize::x);
        spMapEntries[0].constantID = 0;
        spMapEntries[0].offset = offset;
        spMapEntries[0].size = size;
        offset += size;
        size = sizeof(WorkGroupSize::y);
        spMapEntries[1].constantID = 1;
        spMapEntries[1].offset = offset;
        spMapEntries[1].size = size;
        offset += size;
        size = sizeof(WorkGroupSize::z);
        spMapEntries[2].constantID = 2;
        spMapEntries[2].offset = offset;
        spMapEntries[2].size = size;
      }
      VkSpecializationInfo spInfo;
      spInfo.mapEntryCount = 3;
      spInfo.pMapEntries = spMapEntries;
      spInfo.dataSize = sizeof(workGroupSize);
      spInfo.pData = &workGroupSize;

      VkPipelineShaderStageCreateInfo shaderStageCreateInfo{};
      shaderStageCreateInfo.sType =
          VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      shaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
      shaderStageCreateInfo.module = computeShaderModule_;
      shaderStageCreateInfo.pName = "main";
      shaderStageCreateInfo.pSpecializationInfo = &spInfo;

      VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{};
      pipelineLayoutCreateInfo.sType =
          VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
      pipelineLayoutCreateInfo.setLayoutCount = 1;
      pipelineLayoutCreateInfo.pSetLayouts = &descrSetLayout;

      VK_CHECK(vkCreatePipelineLayout(
          device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout_));

      VkComputePipelineCreateInfo pipelineCreateInfo{};
      pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
      pipelineCreateInfo.stage = shaderStageCreateInfo;
      pipelineCreateInfo.layout = pipelineLayout_;

      VK_CHECK(vkCreateComputePipelines(
          device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipeline_));
        */
    }

    #[cfg(USE_VULKAN_SHADERC_RUNTIME)]
    pub fn create_compute_pipeline_compile(&mut self, 
        glsl_src:         &String,
        pipeline_cache:   VkPipelineCache,
        descr_set_layout: VkDescriptorSetLayout,
        work_group_size:  WorkGroupSize)  {
        
        todo!();
        /*
            shaderc::Compiler compiler{};
      shaderc::CompileOptions options{};
    #ifdef DEBUG
      options.SetGenerateDebugInfo();
    #endif
      options.SetTargetEnvironment(
          shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_0);
      options.SetForcedVersionProfile(450, shaderc_profile_core);
      const shaderc::SpvCompilationResult compilationResult = compiler.CompileGlslToSpv(
          glslSrc.c_str(),
          glslSrc.size(),
          shaderc_compute_shader,
          "vulkan_shader.comp",
          "main",
          options);
      const auto compilationStatus = compilationResult.GetCompilationStatus();
      TORCH_INTERNAL_ASSERT(
          compilationStatus == shaderc_compilation_status_success,
          "Shader compilation error: status:",
          compilationStatus,
          compilationResult.GetErrorMessage());
      const vector<u32> shaderSpvCode(
          compilationResult.cbegin(), compilationResult.cend());
      const auto codeSizeBytes = 4 * shaderSpvCode.size();
      createComputePipeline(
          shaderSpvCode.data(),
          codeSizeBytes,
          pipelineCache,
          descrSetLayout,
          workGroupSize);
        */
    }
    
    pub fn create_command_buffer(&mut self, descriptor_set: &mut VkDescriptorSet)  {
        
        todo!();
        /*
            const auto device = context().device();
      VkCommandBufferAllocateInfo commandBufferAllocateInfo{};
      commandBufferAllocateInfo.sType =
          VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
      commandBufferAllocateInfo.commandPool = context().commandPool();
      commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      commandBufferAllocateInfo.commandBufferCount = 1;

      VK_CHECK(vkAllocateCommandBuffers(
          device, &commandBufferAllocateInfo, &commandBuffer_));

      VkCommandBufferBeginInfo beginInfo{};
      beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      VK_CHECK(vkBeginCommandBuffer(commandBuffer_, &beginInfo));

      vkCmdBindPipeline(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
      vkCmdBindDescriptorSets(
          commandBuffer_,
          VK_PIPELINE_BIND_POINT_COMPUTE,
          pipelineLayout_,
          0,
          1,
          &descriptorSet,
          0,
          nullptr);
        */
    }
    
    pub fn add_memory_barrier(&mut self, 
        src_stage_mask:  VkPipelineStageFlags,
        src_access_mask: VkAccessFlags,
        dst_stage_mask:  VkPipelineStageFlags,
        dst_access_mask: VkAccessFlags)  {
        
        todo!();
        /*
            VkMemoryBarrier barrier{};
      barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      barrier.pNext = nullptr;
      barrier.srcAccessMask = srcAccessMask;
      barrier.dstAccessMask = dstAccessMask;
      vkCmdPipelineBarrier(
          commandBuffer_,
          srcStageMask,
          dstStageMask,
          0,
          1,
          &barrier,
          0,
          nullptr,
          0,
          nullptr);
        */
    }
    
    pub fn dispatch_command_buffer(&mut self, 
        group_countx: u32,
        group_county: u32,
        group_countz: u32)  {
        
        todo!();
        /*
            vkCmdDispatch(commandBuffer_, groupCountX, groupCountY, groupCountZ);
        */
    }
    
    pub fn end_command_buffer(&mut self)  {
        
        todo!();
        /*
            native::vulkan::endCommandBuffer(commandBuffer_);
        */
    }
    
    pub fn dispatch_command_buffer(&mut self, 
        gridx:           u32,
        gridy:           u32,
        gridz:           u32,
        work_group_size: WorkGroupSize)  {
        
        todo!();
        /*
            dispatchCommandBuffer(
          UP_DIV(gridX, workGroupSize.x),
          UP_DIV(gridY, workGroupSize.y),
          UP_DIV(gridZ, workGroupSize.z));
        */
    }
    
    pub fn submit_and_wait_command_buffer(&mut self)  {
        
        todo!();
        /*
            VkSubmitInfo submitInfo{};
      submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
      submitInfo.commandBufferCount = 1;
      submitInfo.pCommandBuffers = &commandBuffer_;

      VkFence fence{};
      VkFenceCreateInfo fenceCreateInfo{};
      fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
      fenceCreateInfo.flags = 0;
      VK_CHECK(vkCreateFence(context().device(), &fenceCreateInfo, NULL, &fence))

      VK_CHECK(vkQueueSubmit(context().queue(), 1, &submitInfo, fence));
      vkWaitForFences(context().device(), 1, &fence, VK_TRUE, kFenceTimeoutNanos);

      vkDestroyFence(context().device(), fence, NULL);
        */
    }
}

pub fn make_uniform_const_buffer(
        ptr:  *const c_void,
        size: VkDeviceSize) -> VBuffer {
    
    todo!();
        /*
            VBuffer constBuffer = VBuffer::makeUniformBuffer(size);
      constBuffer.copy_from_host_to_device(ptr, size);
      return constBuffer;
        */
}

impl Drop for ComputeUnitFactory {

    fn drop(&mut self) {
        todo!();
        /*
            vkDestroyPipelineCache(device_, pipelineCache_, nullptr /* allocator */);
        */
    }
}

impl ComputeUnitFactory {
    
    pub fn new(device: VkDevice) -> Self {
    
        todo!();
        /*


            : device_(device) 

      VkPipelineCacheCreateInfo createInfo{};
      createInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
      createInfo.pNext = nullptr;
      createInfo.flags = 0;
      createInfo.initialDataSize = 0;
      createInfo.pInitialData = nullptr;
      VK_CHECK(vkCreatePipelineCache(
          device_, &createInfo, nullptr /* allocator */, &pipelineCache_));
        */
    }
    
    pub fn get_cache_key(&mut self, 
        key:             *const u8,
        work_group_size: WorkGroupSize) -> String {
        
        todo!();
        /*
            stringstream ss;
      ss << key << ':' << workGroupSize.x << ':' << workGroupSize.y << ':'
         << workGroupSize.z;
      return ss.str();
        */
    }
    
    pub fn get(&mut self, 
        cache_key:  &String,
        factory_fn: fn() -> Arc<ComputeUnit>) -> &mut ComputeUnit {
        
        todo!();
        /*
            const auto it = computeUnits_.find(cacheKey);
      if (it != computeUnits_.end()) {
        return *(it->second.get());
      }
      auto computeUnit = factoryFn();
      computeUnits_.insert(make_pair(cacheKey, computeUnit));
      return *(computeUnit.get());
        */
    }

    #[cfg(USE_VULKAN_SHADERC_RUNTIME)]
    pub fn get(&mut self, 
        key:              *const u8,
        glsl_src:         *const u8,
        descr_set_layout: VkDescriptorSetLayout,
        work_group_size:  WorkGroupSize) -> &mut ComputeUnit {
        
        todo!();
        /*
            return get(
          getCacheKey(key, workGroupSize),
          [glslSrc,
           pipelineCache = pipelineCache_,
           descrSetLayout,
           workGroupSize]() {
            return make_shared<ComputeUnit>(
                glslSrc, pipelineCache, descrSetLayout, workGroupSize);
          });
        */
    }

    #[cfg(not(USE_VULKAN_SHADERC_RUNTIME))]
    pub fn get(&mut self, 
        key:              *const u8,
        code:             *const u32,
        code_size:        u32,
        descr_set_layout: VkDescriptorSetLayout,
        work_group_size:  WorkGroupSize) -> &mut ComputeUnit {
        
        todo!();
        /*
            return get(
          getCacheKey(key, workGroupSize),
          [code,
           codeSize,
           pipelineCache = pipelineCache_,
           descrSetLayout,
           workGroupSize]() {
            return make_shared<ComputeUnit>(
                code, codeSize, pipelineCache, descrSetLayout, workGroupSize);
          });
        */
    }
}

/// VBuffer <-> VImage
pub fn copy_buffer_to_image(
    buffer: &VBuffer,
    image:  &mut VImage)  {
    
    todo!();
        /*
            const auto device = context().device();

      VkDescriptorSetLayout descrSetLayout{};
      VkDescriptorSetLayoutBinding bindings[] = {
          descriptorSetLayoutBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE),
          descriptorSetLayoutBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
          descriptorSetLayoutBinding(2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)};
      createDescriptorSetLayout(
          device, bindings, 3 /* bindingsCount */, &descrSetLayout);

      VkDescriptorPool descrPool{};
      VkDescriptorPoolSize poolSizes[] = {{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
                                          {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
                                          {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}};
      createDescriptorPool(
          device, poolSizes, 3 /* poolSizeCount */, 1 /* maxSets */, &descrPool);

      VkDescriptorSet descrSet{};
      allocateDescriptorSet(device, descrPool, &descrSetLayout, &descrSet);

      image.bindStorageImage(descrSet, 0);
      buffer.bind(descrSet, 1);
      WorkGroupSize workGroupSize{8, 8, 1};

      auto& computeUnit = context().computeUnitFactory().get(
          GLSL_SPV(nchw_to_image), descrSetLayout, workGroupSize);
      computeUnit.createCommandBuffer(descrSet);

      image.addImageMemoryBarrierToGeneral(computeUnit.commandBuffer());
      buffer.addBufferMemoryBarrier(
          computeUnit.commandBuffer(), 0, buffer.sizeBytes());
      computeUnit.addMemoryBarrier(
          VK_PIPELINE_STAGE_HOST_BIT,
          VK_ACCESS_HOST_WRITE_BIT,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_ACCESS_SHADER_READ_BIT);
      computeUnit.dispatchCommandBuffer(
          image.w(), image.h(), image.d(), workGroupSize);
      computeUnit.endCommandBuffer();
      computeUnit.submitAndWaitCommandBuffer();

      vkDestroyDescriptorPool(device, descrPool, nullptr);
      vkDestroyDescriptorSetLayout(device, descrSetLayout, nullptr);
        */
}

pub fn copy_image_to_buffer(
    image:                              &VImage,
    buffer:                             &mut VBuffer,
    add_buffer_memory_barrier_for_host: bool)  {

    let add_buffer_memory_barrier_for_host: bool = add_buffer_memory_barrier_for_host.unwrap_or(false);

    todo!();
        /*
            const auto device = context().device();
      TORCH_INTERNAL_ASSERT(
          buffer.sizeBytes() >= image.capacityBytes(),
          "VulkanBuffer's capacity is less than VulkanImage capacity to copy from");

      VkDescriptorSetLayout descrSetLayout{};
      const VkDescriptorSetLayoutBinding bindings[] = {
          descriptorSetLayoutBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER),
          descriptorSetLayoutBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
          descriptorSetLayoutBinding(2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)};
      createDescriptorSetLayout(
          device, bindings, 3 /* bindingsCount */, &descrSetLayout);

      VkDescriptorPool descrPool{};
      const VkDescriptorPoolSize poolSizes[] = {
          {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
          {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
          {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}};
      createDescriptorPool(
          device, poolSizes, 3 /* poolSizeCount */, 1 /* maxSets */, &descrPool);

      VkDescriptorSet descrSet{};
      allocateDescriptorSet(device, descrPool, &descrSetLayout, &descrSet);

      image.bindShaderRead(descrSet, 0);
      buffer.bind(descrSet, 1);

      const WorkGroupSize workGroupSize{8, 8, 1};
      auto& computeUnit = context().computeUnitFactory().get(
          GLSL_SPV(image_to_nchw), descrSetLayout, workGroupSize);

      computeUnit.createCommandBuffer(descrSet);
      image.addImageMemoryBarrierToShaderRead(computeUnit.commandBuffer());
      computeUnit.dispatchCommandBuffer(
          image.w(), image.h(), image.d(), workGroupSize);

      if (addBufferMemoryBarrierForHost) {
        computeUnit.addMemoryBarrier(
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_HOST_BIT,
            VK_ACCESS_HOST_READ_BIT);
      }
      computeUnit.endCommandBuffer();
      computeUnit.submitAndWaitCommandBuffer();

      vkDestroyDescriptorPool(device, descrPool, nullptr);
      vkDestroyDescriptorSetLayout(device, descrSetLayout, nullptr);
        */
}

pub fn copy_buffer_to_buffer(
    src_buffer: &VBuffer,
    dst_buffer: &mut VBuffer,
    size:       VkDeviceSize,
    src_offset: VkDeviceSize,
    dst_offset: VkDeviceSize)  {

    let src_offset: VkDeviceSize = src_offset.unwrap_or(0);
    let dst_offset: VkDeviceSize = dst_offset.unwrap_or(0);
    
    todo!();
        /*
            auto device = context().device();
      VkCommandBuffer commandBuffer{};
      allocateCommandBuffer(device, &commandBuffer);
      beginCommandBuffer(commandBuffer);

      VkBufferCopy copyRegion{};
      copyRegion.srcOffset = srcOffset;
      copyRegion.dstOffset = dstOffset;
      copyRegion.size = size;
      vkCmdCopyBuffer(
          commandBuffer,
          srcBuffer.vkbuffer(),
          dstBuffer.vkbuffer(),
          1,
          &copyRegion);

      endCommandBuffer(commandBuffer);
      submitAndWaitCommandBuffer(device, context().queue(), commandBuffer);
        */
}

pub struct VulkanTensorImpl {
    sizes:   Vec<i64>,
    strides: Vec<i64>,
    numel:   i64,
    buffer:  RefCell<Box<VBuffer>>,
    image:   Box<VImage>,
}

impl VulkanTensorImpl {
    
    pub fn new(sizes: Vec<i64>) -> Self {
    
        todo!();
        /*


            : sizes_(move(sizes)),
            strides_(vector<i64>(sizes_.size())),
            numel_(multiply_integers(sizes_)) 

        TORCH_CHECK(
            initVulkanContextOnce(), "Vulkan Failed to create Vulkan Context");
        */
    }
    
    pub fn sizes(&self) -> Vec<i64> {
        
        todo!();
        /*
            return sizes_;
        */
    }
    
    pub fn strides(&self) -> Vec<i64> {
        
        todo!();
        /*
            return strides_;
        */
    }
    
    #[inline] pub fn dim(&self) -> i64 {
        
        todo!();
        /*
            return sizes_.size();
        */
    }
    
    #[inline] pub fn numel(&self) -> i64 {
        
        todo!();
        /*
            return numel_;
        */
    }
    
    #[inline] pub fn has_buffer(&self) -> bool {
        
        todo!();
        /*
            return static_cast<bool>(buffer_);
        */
    }
    
    #[inline] pub fn buffer(&mut self) -> *mut VBuffer {
        
        todo!();
        /*
            if (!has_buffer()) {
          buffer_ = make_unique<VBuffer>(buffer_size_for_sizes(sizes_));
        }
        return buffer_.get();
        */
    }
    
    pub fn buffer(&self) -> *const VBuffer {
        
        todo!();
        /*
            if (!has_buffer()) {
          buffer_ = make_unique<VBuffer>(buffer_size_for_sizes(sizes_));
        }
        return buffer_.get();
        */
    }
    
    #[inline] pub fn can_be_image(&self) -> bool {
        
        todo!();
        /*
            return dim() <= 4;
        */
    }
    
    #[inline] pub fn has_image(&self) -> bool {
        
        todo!();
        /*
            return static_cast<bool>(image_);
        */
    }
    
    #[inline] pub fn has_storage(&mut self) -> bool {
        
        todo!();
        /*
            return has_buffer();
        */
    }
    
    pub fn image_sizes_w_h_nc4(&mut self) -> ImageSizes {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(
            can_be_image(),
            "Vulkan: Only Tensors with dim <= 4 can be represented as Vulkam Image");
        auto d = dim();
        i64 _wd = 1;
        i64 _hd = 1;
        i64 _dd = 1;
        if (d == 4) {
          _wd = sizes_[3];
          _hd = sizes_[2];
          _dd = sizes_[1] * sizes_[0];
        } else if (d == 3) {
          _wd = sizes_[2];
          _hd = sizes_[1];
          _dd = sizes_[0];
        } else if (d == 2) {
          _wd = sizes_[1];
          _hd = sizes_[0];
        } else if (d == 1) {
          _wd = sizes_[0];
        }
        i32 wd = safe_downcast<i64>(_wd);
        i32 hd = safe_downcast<i64>(_hd);
        i32 dd = safe_downcast<i64>(_dd);
        return {{wd, hd, UP_DIV(dd, 4)}, {wd, hd, dd}};
        */
    }
    
    pub fn image(&mut self, image_sizes: Option<ImageSizes>) -> *mut VImage {
        let image_sizes: Option<ImageSizes> = image_sizes.unwrap_or(nullopt);

        todo!();
        /*
            if (image_) {
          return image_.get();
        }

        if (imageSizes.has_value()) {
          image_ = make_unique<VImage>(*imageSizes);
          return image_.get();
        }

        image_ = make_unique<VImage>(imageSizes_W_H_NC4());
        if (buffer_) {
          copy_buffer_to_image(*buffer_, *image_);
        }
        return image_.get();
        */
    }
    
    pub fn image(&self, image_sizes: Option<ImageSizes>) -> *const VImage {
        let image_sizes: Option<ImageSizes> = image_sizes.unwrap_or(nullopt);

        todo!();
        /*
            return const_cast<VulkanTensor::Impl*>(this)->image(imageSizes);
        */
    }
    
    pub fn buffer_size_for_sizes(&self, sizes: Vec<i64>) -> VkDeviceSize {
        
        todo!();
        /*
            const auto d = sizes.size();
        const auto numel = multiply_integers(sizes);
        VkDeviceSize bufferSize{sizeof(float) * numel};
        // alignment to be able to copy between image and buffer
        if (d == 4) {
          bufferSize =
              sizeof(float) * ALIGN_UP4(sizes[0] * sizes[1]) * sizes[2] * sizes[3];
        } else if (d == 3) {
          bufferSize = sizeof(float) * ALIGN_UP4(sizes[0]) * sizes[1] * sizes[2];
        } else if (d == 2) {
          bufferSize = sizeof(float) * 4 * sizes[0] * sizes[1];
        } else if (d == 1) {
          bufferSize = sizeof(float) * 4 * sizes[0];
        }
        return bufferSize;
        */
    }
    
    pub fn allocate_storage(&mut self)  {
        
        todo!();
        /*
            buffer_ = make_unique<VBuffer>(buffer_size_for_sizes(sizes_));
        */
    }
    
    pub fn set_data_from_host(&mut self, input_data: *const f32)  {
        
        todo!();
        /*
            buffer()->copy_from_host_to_device(
            (const void*)inputData, sizeof(float) * numel_);
        */
    }
    
    pub fn copy_data_to_host(&self, output_data: *mut f32)  {
        
        todo!();
        /*
            sync_image_to_buffer();
        buffer()->copy_from_device_to_host(outputData, sizeof(float) * numel_);
        */
    }
    
    pub fn sync_image_to_buffer(&self)  {
        
        todo!();
        /*
            if (has_image()) {
          copy_image_to_buffer(
              *image(),
              *(const_cast<VBuffer*>(buffer())),
              true /* memory barrier for host memory map */);
        }
        */
    }
}

impl VulkanTensor {
    
    pub fn impl_(&mut self) -> Arc<VulkanTensorImpl> {
        
        todo!();
        /*
            return impl_;
        */
    }
    
    pub fn impl_(&self) -> Arc<VulkanTensorImpl> {
        
        todo!();
        /*
            return impl_;
        */
    }
    
    pub fn sizes(&self) -> Vec<i64> {
        
        todo!();
        /*
            return impl()->sizes();
        */
    }
    
    pub fn sync_image_to_buffer(&self)  {
        
        todo!();
        /*
            return impl()->sync_image_to_buffer();
        */
    }
    
    pub fn strides(&self) -> Vec<i64> {
        
        todo!();
        /*
            return impl()->strides();
        */
    }
    
    pub fn dim(&self) -> i64 {
        
        todo!();
        /*
            return impl()->dim();
        */
    }
    
    pub fn numel(&self) -> i64 {
        
        todo!();
        /*
            return impl()->numel();
        */
    }
    
    pub fn has_storage(&self) -> bool {
        
        todo!();
        /*
            return impl()->has_buffer();
        */
    }
    
    pub fn allocate_storage(&mut self)  {
        
        todo!();
        /*
            impl()->allocate_storage();
        */
    }
    
    pub fn set_data_from_host(&mut self, input_data: *const f32)  {
        
        todo!();
        /*
            impl()->set_data_from_host(inputData);
        */
    }
    
    pub fn copy_data_to_host(&self, output_data: *mut f32)  {
        
        todo!();
        /*
            impl()->copy_data_to_host(outputData);
        */
    }
    
    pub fn has_buffer(&self) -> bool {
        
        todo!();
        /*
            return impl()->has_buffer();
        */
    }
    
    pub fn buffer(&mut self) -> *mut VBuffer {
        
        todo!();
        /*
            return impl()->buffer();
        */
    }
    
    pub fn buffer(&self) -> *const VBuffer {
        
        todo!();
        /*
            return impl()->buffer();
        */
    }
    
    pub fn can_be_image(&self) -> bool {
        
        todo!();
        /*
            return impl()->can_be_image();
        */
    }
    
    pub fn has_image(&self) -> bool {
        
        todo!();
        /*
            return impl()->has_image();
        */
    }
    
    pub fn image(&mut self, image_sizes: Option<ImageSizes>) -> *mut VImage {
        
        todo!();
        /*
            return impl()->image(imageSizes);
        */
    }
    
    pub fn image(&self, image_sizes: Option<ImageSizes>) -> *const VImage {
        
        todo!();
        /*
            return impl()->image(imageSizes);
        */
    }
    
    pub fn new(sizes: Vec<i64>) -> Self {
    
        todo!();
        /*


            : impl_(make_shared<Impl>(move(sizes)))
        */
    }
}

impl fmt::Display for ImageSize {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            s << "ImageSize{" << imageSize[0] << ", " << imageSize[1] << ", "
        << imageSize[2] << "}";
      return s;
        */
    }
}

impl fmt::Display for ImageSizes {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            s << "ImageSizes{imageSize:" << imageSizes.imageSize
        << ", dataSize:" << imageSizes.dataSize << "}";
      return s;
        */
    }
}

impl fmt::Display for WorkGroupSize {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            s << "WorkGroupSize{" << workGroupSize.x << " " << workGroupSize.y << " "
        << workGroupSize.z << "}";
      return s;
        */
    }
}
