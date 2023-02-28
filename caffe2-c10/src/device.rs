crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/core/Device.h]

/**
  | An index representing a specific device; e.g.,
  | the 1 in GPU 1.
  |
  | A DeviceIndex is not independently meaningful
  | without knowing the DeviceType it is
  | associated; try to use Device rather than
  | DeviceIndex directly.
  |
  */
pub type DeviceIndex = i8;

/**
  | Represents a a compute device on which
  | a tensor is located.
  |
  | A device is uniquely identified by a type,
  | which specifies the type of machine it is
  | (e.g. CPU or Cuda GPU), and a device index or
  | ordinal, which identifies the specific compute
  | device when there is more than one of
  | a certain type.
  |
  | The device index is optional, and in its
  | defaulted state represents (abstractly) "the
  | current device".
  |
  | Further, there are two constraints on the
  | value of the device index, if one is
  | explicitly stored:
  |
  | 1. A negative index represents the current
  | device, a non-negative index represents
  | a specific, concrete device,
  |
  | 2. When the device type is CPU, the device
  | index must be zero.
  |
  */
#[derive(Default,PartialEq,Eq)]
pub struct Device {
    ty:    DeviceType,
    index: DeviceIndex, // default = -1
}

impl From<&str> for Device {

    /**
      | Constructs a `Device` from a string
      | description, for convenience.
      |
      | The string supplied must follow the
      | following schema:
      | `(cpu|cuda)[:<device-index>]`
      |
      | where `cpu` or `cuda` specifies the device
      | type, and `:<device-index>` optionally
      | specifies a device index.
      |
      */
    fn from(device_string: &str) -> Self {
    
        todo!();
        /*
        : device(Type::CPU),

            TORCH_CHECK(!device_string.empty(), "Device string must not be empty");

      // We assume gcc 5+, so we can use proper regex.
      static const regex regex("([a-zA-Z_]+)(?::([1-9]\\d*|0))?");
      smatch match;
      TORCH_CHECK(
          regex_match(device_string, match, regex),
          "Invalid device string: '",
          device_string,
          "'");
      type_ = parse_type(match[1].str());
      if (match[2].matched) {
        try {
          index_ = stoi(match[2].str());
        } catch (const exception&) {
          TORCH_CHECK(
              false,
              "Could not parse device index '",
              match[2].str(),
              "' in device string '",
              device_string,
              "'");
        }
      }
      validate();
        */
    }
}

impl Device {

    /**
      | Constructs a new `Device` from a `DeviceType`
      | and an optional device index.
      |
      */
    pub fn new(
        ty:    DeviceType,
        index: Option<DeviceIndex>) -> Self {

        let index: DeviceIndex = index.unwrap_or(-1);

        todo!();
        /*
        : ty(type),
        : index(index),

            validate();
        */
    }


    /// Sets the device index.
    pub fn set_index(&mut self, index: DeviceIndex)  {
        
        todo!();
        /*
            index_ = index;
        */
    }

    /// Returns the type of device this is.
    pub fn ty(&self) -> DeviceType {
        
        todo!();
        /*
            return type_;
        */
    }

    /// Returns the optional index.
    pub fn index(&self) -> DeviceIndex {
        
        todo!();
        /*
            return index_;
        */
    }

    /// Returns true if the device has
    /// a non-default index.
    ///
    pub fn has_index(&self) -> bool {
        
        todo!();
        /*
            return index_ != -1;
        */
    }

    /// Return true if the device is of Cuda type.
    ///
    pub fn is_cuda(&self) -> bool {
        
        todo!();
        /*
            return type_ == DeviceType::CUDA;
        */
    }

    /// Return true if the device is of HIP type.
    ///
    pub fn is_hip(&self) -> bool {
        
        todo!();
        /*
            return type_ == DeviceType::HIP;
        */
    }

    /// Return true if the device is of XPU type.
    ///
    pub fn is_xpu(&self) -> bool {
        
        todo!();
        /*
            return type_ == DeviceType::XPU;
        */
    }

    /// Return true if the device is of CPU type.
    pub fn is_cpu(&self) -> bool {
        
        todo!();
        /*
            return type_ == DeviceType::CPU;
        */
    }

    /// Return true if the device supports
    /// arbirtary strides.
    ///
    pub fn supports_as_strided(&self) -> bool {
        
        todo!();
        /*
            return type_ != DeviceType::XLA;
        */
    }

    pub fn validate(&mut self)  {
        
        todo!();
        /*
            // Removing these checks in release builds noticeably improves
        // performance in micro-benchmarks.
        // This is safe to do, because backends that use the DeviceIndex
        // have a later check when we actually try to switch to that device.
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
            index_ == -1 || index_ >= 0,
            "Device index must be -1 or non-negative, got ",
            (int)index_);
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
            !is_cpu() || index_ <= 0,
            "CPU device index must be -1 or zero, got ",
            (int)index_);
        */
    }
}

impl Hash for Device {

    fn hash<H>(&self, state: &mut H) where H: Hasher 
    {
        
        todo!();
        /*
            // Are you here because this static assert failed?  Make sure you ensure
        // that the bitmasking code below is updated accordingly!
        static_assert(sizeof(DeviceType) == 1, "DeviceType is not 8-bit");
        static_assert(sizeof(DeviceIndex) == 1, "DeviceIndex is not 8-bit");
        // Note [Hazard when concatenating signed integers]
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        // We must first convert to a same-sized unsigned type, before promoting to
        // the result type, to prevent sign extension when any of the values is -1.
        // If sign extension occurs, you'll clobber all of the values in the MSB
        // half of the resulting integer.
        //
        // Technically, by C/C++ integer promotion rules, we only need one of the
        // uint32_t casts to the result type, but we put in both for explicitness's
        // sake.
        uint32_t bits = static_cast<uint32_t>(static_cast<uint8_t>(d.type()))
                << 16 |
            static_cast<uint32_t>(static_cast<uint8_t>(d.index()));
        return hash<uint32_t>{}(bits);
        */
    }
}

//-------------------------------------------[.cpp/pytorch/c10/core/Device.cpp]

lazy_static!{
    /*
    // Check if compiler has working regex implementation
    //
    // Test below is adapted from https://stackoverflow.com/a/41186162
    #if defined(_MSVC_LANG) && _MSVC_LANG >= 201103L
    // Compiler has working regex. MSVC has erroneous __cplusplus.
    #elif __cplusplus >= 201103L &&                           \
        (!defined(__GLIBCXX__) || (__cplusplus >= 201402L) || \
         (defined(_GLIBCXX_REGEX_DFS_QUANTIFIERS_LIMIT) ||    \
          defined(_GLIBCXX_REGEX_STATE_LIMIT) ||              \
          (defined(_GLIBCXX_RELEASE) && _GLIBCXX_RELEASE > 4)))
    // Compiler has working regex.
    #else
    static_assert(false, "Compiler does not have proper regex support.");
    #endif
    */
}

pub fn parse_type(device_string: &String) -> DeviceType {
    
    todo!();
        /*
            static const array<
          pair<string, DeviceType>,
          static_cast<size_t>(DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES)>
          types = {{
              {"cpu", DeviceType::CPU},
              {"cuda", DeviceType::CUDA},
              {"xpu", DeviceType::XPU},
              {"mkldnn", DeviceType::MKLDNN},
              {"opengl", DeviceType::OPENGL},
              {"opencl", DeviceType::OPENCL},
              {"ideep", DeviceType::IDEEP},
              {"hip", DeviceType::HIP},
              {"fpga", DeviceType::FPGA},
              {"msnpu", DeviceType::MSNPU},
              {"xla", DeviceType::XLA},
              {"vulkan", DeviceType::Vulkan},
              {"mlc", DeviceType::MLC},
              {"meta", DeviceType::Meta},
              {"hpu", DeviceType::HPU},
          }};
      auto device = find_if(
          types.begin(),
          types.end(),
          [device_string](const pair<string, DeviceType>& p) {
            return p.first == device_string;
          });
      if (device != types.end()) {
        return device->second;
      }
      TORCH_CHECK(
          false,
          "Expected one of cpu, cuda, xpu, mkldnn, opengl, opencl, ideep, hip, msnpu, mlc, xla, vulkan, meta, hpu device type at start of device string: ",
          device_string);
        */
}

impl Device {
    
    /// Same string as returned from operator<<.
    ///
    pub fn str_(&self) -> String {
        
        todo!();
        /*
            string str = DeviceTypeName(type(), /* lower case */ true);
      if (has_index()) {
        str.push_back(':');
        str.append(to_string(index()));
      }
      return str;
        */
    }
}

impl fmt::Display for Device {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            stream << device.str();
      return stream;
        */
    }
}
