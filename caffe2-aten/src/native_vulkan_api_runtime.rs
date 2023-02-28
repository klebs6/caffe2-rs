crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/api/Runtime.h]

pub enum RuntimeType {
    Debug,
    Release,
}

pub type RuntimeSelector = fn(_0: &Adapter) -> bool;

pub struct RuntimeDebug {
    instance: VkInstance,
}

/**
  | A Vulkan Runtime initializes a Vulkan
  | instance and decouples the concept
  | of Vulkan instance initialization
  | from intialization of, and subsequent
  | interactions with, Vulkan [physical
  | and logical] devices as a precursor
  | to multi-GPU support.
  | 
  | The Vulkan Runtime can be queried for
  | available
  | 
  | Adapters (i.e. physical devices) in
  | the system which in turn can be used for
  | creation of a Vulkan Context (i.e. logical
  | devices).
  | 
  | All Vulkan tensors in PyTorch are associated
  | with a Context to make tensor <-> device
  | affinity explicit.
  |
  */
pub struct Runtime {

    /**
      | Construction and destruction order
      | matters. Do not move members around.
      |
      */
    instance:              Handle<VkInstance, decltype(&VK_DELETER(Instance))>,
    debug_report_callback: Handle<VkDebugReportCallbackEXT,RuntimeDebug>,
}

impl Runtime {
    
    #[inline] pub fn instance(&self) -> VkInstance {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(instance_);
      return instance_.get();
        */
    }
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/api/Runtime.cpp]

mod configuration {
    use super::*;

    #[cfg(debug_assertions)]      pub const RUNTIME: RuntimeType = RuntimeTypeDebug;
    #[cfg(not(debug_assertions))] pub const RUNTIME: RuntimeType = RuntimeTypeRelease;
}

pub fn debug_report_callback_fn(
        flags:        vk::DebugReportFlagsEXT,
        object_type:  vk::DebugReportObjectTypeEXT,
        object:       u64,
        location:     usize,
        message_code: i32,
        layer_prefix: *const u8,
        message:      *const u8,
        user_data:    *mut ()) -> vk::Bool32 {
    
    todo!();
        /*
            stringstream stream;
      stream << layer_prefix << " " << message_code << " " << message << endl;
      const string log = stream.str();

      if (flags & VK_DEBUG_REPORT_ERROR_BIT_EXT) {
        LOG(ERROR) << log;
      } else if (flags & VK_DEBUG_REPORT_WARNING_BIT_EXT) {
        LOG(WARNING) << log;
      } else if (flags & VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT) {
        LOG(WARNING) << "Performance:" << log;
      } else if (flags & VK_DEBUG_REPORT_INFORMATION_BIT_EXT) {
        LOG(INFO) << log;
      } else if (flags & VK_DEBUG_REPORT_DEBUG_BIT_EXT) {
        LOG(INFO) << "Debug: " << log;
      }

      return VK_FALSE;
        */
}

pub fn create_instance(ty: RuntimeType) -> VkInstance {
    
    todo!();
        /*
            vector<const char*> enabled_instance_layers;
      vector<const char*> enabled_instance_extensions;

      if (Runtime::Type::Debug == type) {
        u32 instance_layers_count = 0;
        VK_CHECK(vkEnumerateInstanceLayerProperties(
            &instance_layers_count, nullptr));

        vector<VkLayerProperties> instance_layer_properties(
            instance_layers_count);

        VK_CHECK(vkEnumerateInstanceLayerProperties(
            &instance_layers_count,
            instance_layer_properties.data()));

        constexpr const char* const requested_instance_layers[]{
            // "VK_LAYER_LUNARG_api_dump",
            "VK_LAYER_KHRONOS_validation",
        };

        for (const auto& requested_instance_layer : requested_instance_layers) {
          for (const auto& layer : instance_layer_properties) {
            if (strcmp(requested_instance_layer, layer.layerName) == 0) {
              enabled_instance_layers.push_back(requested_instance_layer);
              break;
            }
          }
        }

        u32 instance_extension_count = 0;
        VK_CHECK(vkEnumerateInstanceExtensionProperties(
            nullptr, &instance_extension_count, nullptr));

        vector<VkExtensionProperties> instance_extension_properties(
            instance_extension_count);

        VK_CHECK(vkEnumerateInstanceExtensionProperties(
            nullptr, &instance_extension_count, instance_extension_properties.data()));

        constexpr const char* const requested_instance_extensions[]{
        #ifdef VK_EXT_debug_report
          VK_EXT_DEBUG_REPORT_EXTENSION_NAME,
        #endif
        };

        for (const auto& requested_instance_extension : requested_instance_extensions) {
          for (const auto& extension : instance_extension_properties) {
            if (strcmp(requested_instance_extension, extension.extensionName) == 0) {
              enabled_instance_extensions.push_back(requested_instance_extension);
              break;
            }
          }
        }
      }

      constexpr VkApplicationInfo application_info{
        VK_STRUCTURE_TYPE_APPLICATION_INFO,
        nullptr,
        "PyTorch",
        0,
        "PyTorch",
        0,
        VK_API_VERSION_1_0,
      };

      const VkInstanceCreateInfo instance_create_info{
        VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        nullptr,
        0u,
        &application_info,
        static_cast<u32>(enabled_instance_layers.size()),
        enabled_instance_layers.data(),
        static_cast<u32>(enabled_instance_extensions.size()),
        enabled_instance_extensions.data(),
      };

      VkInstance instance{};
      VK_CHECK(vkCreateInstance(&instance_create_info, nullptr, &instance));
      TORCH_CHECK(instance, "Invalid Vulkan instance!");

      return instance;
        */
}

pub fn create_debug_report_callback(
        instance: VkInstance,
        ty:       RuntimeType) -> VkDebugReportCallbackEXT {
    
    todo!();
        /*
            if (Runtime::Type::Debug != type) {
        return VkDebugReportCallbackEXT{};
      }

      const VkDebugReportCallbackCreateInfoEXT debugReportCallbackCreateInfo{
        VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT,
        nullptr,
        VK_DEBUG_REPORT_INFORMATION_BIT_EXT |
          VK_DEBUG_REPORT_WARNING_BIT_EXT |
          VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT |
          VK_DEBUG_REPORT_ERROR_BIT_EXT |
          VK_DEBUG_REPORT_DEBUG_BIT_EXT,
        debug_report_callback_fn,
        nullptr,
      };

      const auto vkCreateDebugReportCallbackEXT =
          (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(
              instance, "vkCreateDebugReportCallbackEXT");

      TORCH_CHECK(
          vkCreateDebugReportCallbackEXT,
          "Could not load vkCreateDebugReportCallbackEXT");

      VkDebugReportCallbackEXT debug_report_callback{};
      VK_CHECK(vkCreateDebugReportCallbackEXT(
          instance,
          &debugReportCallbackCreateInfo,
          nullptr,
          &debug_report_callback));

      TORCH_CHECK(
          debug_report_callback,
          "Invalid Vulkan debug report callback!");

      return debug_report_callback;
        */
}

pub fn acquire_physical_devices(instance: VkInstance) -> Vec<VkPhysicalDevice> {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          instance,
          "Invalid Vulkan instance!");

      u32 device_count = 0;
      VK_CHECK(vkEnumeratePhysicalDevices(instance, &device_count, nullptr));

      TORCH_CHECK(
          device_count > 0,
          "Vulkan: Could not find a device with Vulkan support!");

      vector<VkPhysicalDevice> devices(device_count);
      VK_CHECK(vkEnumeratePhysicalDevices(instance, &device_count, devices.data()));

      return devices;
        */
}

pub fn query_physical_device_properties(physical_device: VkPhysicalDevice) -> VkPhysicalDeviceProperties {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          physical_device,
          "Invalid Vulkan physical device!");

      VkPhysicalDeviceProperties physical_device_properties{};
      vkGetPhysicalDeviceProperties(
          physical_device,
          &physical_device_properties);

      return physical_device_properties;
        */
}

pub fn query_physical_device_memory_properties(physical_device: VkPhysicalDevice) -> VkPhysicalDeviceMemoryProperties {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          physical_device,
          "Invalid Vulkan physical device!");

      VkPhysicalDeviceMemoryProperties physical_device_memory_properties{};
      vkGetPhysicalDeviceMemoryProperties(
          physical_device,
          &physical_device_memory_properties);

      return physical_device_memory_properties;
        */
}

pub fn query_compute_queue_family_index(physical_device: VkPhysicalDevice) -> u32 {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          physical_device,
          "Invalid Vulkan physical device!");

      u32 queue_family_count = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(
          physical_device, &queue_family_count, nullptr);

      TORCH_CHECK(
          queue_family_count > 0,
          "Vulkan: Invalid number of queue families!");

      vector<VkQueueFamilyProperties>
          queue_families_properties(queue_family_count);

      vkGetPhysicalDeviceQueueFamilyProperties(
          physical_device,
          &queue_family_count,
          queue_families_properties.data());

      for (u32 i = 0; i < queue_families_properties.size(); ++i) {
        const VkQueueFamilyProperties& properties = queue_families_properties[i];
        if (properties.queueCount > 0 && (properties.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
          return i;
        }
      }

      TORCH_CHECK(
          false,
          "Vulkan: Could not find a queue family that supports compute operations!");
        */
}

impl RuntimeDebug {
    
    pub fn new(instance: VkInstance) -> Self {
    
        todo!();
        /*


            : instance_(instance) 
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
            instance,
            "Invalid Vulkan instance!");
        */
    }
    
    pub fn invoke(&self, debug_report_callback: VkDebugReportCallbackEXT)  {
        
        todo!();
        /*
            if (debug_report_callback) {
        const auto vkDestroyDebugReportCallbackEXT =
          (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(
              instance_, "vkDestroyDebugReportCallbackEXT");

          TORCH_CHECK(
              vkDestroyDebugReportCallbackEXT,
              "Could not load vkDestroyDebugReportCallbackEXT");

          vkDestroyDebugReportCallbackEXT(
              instance_, debug_report_callback, nullptr);
      }
        */
    }
}



impl Runtime {
    
    pub fn new(ty: Type) -> Self {
    
        todo!();
        /*


            : instance_(create_instance(type), &VK_DELETER(Instance)),
          debug_report_callback_(
              create_debug_report_callback(instance(), type),
              Debug(instance()))
        */
    }
    
    pub fn select(&mut self, selector: &Selector) -> Adapter {
        
        todo!();
        /*
            const vector<VkPhysicalDevice> physical_devices =
          acquire_physical_devices(instance());

      for (const VkPhysicalDevice physical_device : physical_devices) {
        const Adapter adapter{
          this,
          physical_device,
          query_physical_device_properties(physical_device),
          query_physical_device_memory_properties(physical_device),
          query_compute_queue_family_index(physical_device),
        };

        if (selector(adapter)) {
          return adapter;
        }
      }

      TORCH_CHECK(
          false,
          "Vulkan: no adapter was selected as part of device enumeration!");
        */
    }
}

pub fn runtime() -> *mut Runtime {
    
    todo!();
        /*
            static const unique_ptr<Runtime> runtime([]() -> Runtime* {
    #ifdef USE_VULKAN_WRAPPER
        if (!InitVulkan()) {
          TORCH_WARN("Vulkan: Failed to initialize Vulkan Wrapper!");
          return nullptr;
        }
    #endif

        try {
          return new Runtime(Configuration::kRuntime);
        }
        catch (const exception& e) {
          TORCH_WARN(
              "Vulkan: Failed to initialize runtime! Error: ",
              e.what());
        }
        catch (...) {
          TORCH_WARN(
              "Vulkan: Failed to initialize runtime! "
              "Error: Unknown");
        }

        return nullptr;
      }());

      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          runtime,
          "Invalid Vulkan runtime!");

      return runtime.get();
        */
}
