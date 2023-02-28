/*!
  | This is directly synchronized with
  | caffe2/proto/caffe2.proto, but doesn't require
  | me to figure out how to get Protobuf headers
  | into ATen/core (which would require a lot more
  | build system hacking.)
  |
  | If you modify me, keep me synchronized with
  | that file.
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/core/DeviceType.h]

#[repr(i8)]
#[derive(Default,PartialEq,Eq)]
pub enum DeviceType {

    #[default]
    CPU    = 0,

    Cuda   = 1, // Cuda.
    MKLDNN = 2, // Reserved for explicit MKLDNN
    OPENGL = 3, // OpenGL
    OPENCL = 4, // OpenCL
    IDEEP  = 5, // IDEEP.
    HIP    = 6, // AMD HIP
    FPGA   = 7, // FPGA
    MSNPU  = 8, // MSNPU
    XLA    = 9, // XLA / TPU
    Vulkan = 10, // Vulkan
    Metal  = 11, // Metal
    XPU    = 12, // XPU
    MLC    = 13, // ML Compute / Apple
    Meta   = 14, // Meta (tensors with no data)
    HPU    = 15, // HPU / HABANA
                 // NB: If you add more devices:
                 //  - Change the implementations of DeviceTypeName and isValidDeviceType
                 //    in DeviceType.cpp
                 //  - Change the number below
    COMPILE_TIME_MAX_DEVICE_TYPES = 16,
}

pub const K_CPU:    DeviceType = DeviceType::CPU;
pub const KCUDA:    DeviceType = DeviceType::Cuda;
pub const K_HIP:    DeviceType = DeviceType::HIP;
pub const KFPGA:    DeviceType = DeviceType::FPGA;
pub const KMSNPU:   DeviceType = DeviceType::MSNPU;
pub const K_XLA:    DeviceType = DeviceType::XLA;
pub const K_MLC:    DeviceType = DeviceType::MLC;
pub const K_META:   DeviceType = DeviceType::Meta;
pub const K_VULKAN: DeviceType = DeviceType::Vulkan;
pub const K_METAL:  DeviceType = DeviceType::Metal;
pub const K_XPU:    DeviceType = DeviceType::XPU;
pub const K_HPU:    DeviceType = DeviceType::HPU;

/// define explicit int constant
pub const COMPILE_TIME_MAX_DEVICE_TYPES: i32 = DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES as i32;

/**
  | Hey!  You seem to be adding a lot of new
  | DeviceTypes.
  |
  | The intent was for this constant to reflect
  | the actual number of DeviceTypes we support in
  | PyTorch; it's important that this number is
  | not too large as we use this to allocate stack
  | arrays in some places in our code.
  |
  | If you are indeed just adding the 17th device
  | type, feel free to change the check to 32; but
  | if you are adding some sort of extensible
  | device types registration, please be aware
  | that you are affecting code that this number
  | is small.  Try auditing uses of this constant.
  |
  */
const_assert!(COMPILE_TIME_MAX_DEVICE_TYPES <= 16);

impl Hash for DeviceType {

    fn hash<H>(&self, state: &mut H) where H: Hasher 
    {
        todo!();
        /*
            return hash<int>()(static_cast<int>(k));
        */
    }
}

//-------------------------------------------[.cpp/pytorch/c10/core/DeviceType.cpp]

pub fn device_type_name(
    d:          DeviceType,
    lower_case: Option<bool>) -> String {

    let lower_case: bool = lower_case.unwrap_or(false);
    
    todo!();
        /*
            switch (d) {
        // I considered instead using ctype::tolower to lower-case the strings
        // on the fly, but this seemed a bit much.
        case DeviceType::CPU:
          return lower_case ? "cpu" : "CPU";
        case DeviceType::CUDA:
          return lower_case ? "cuda" : "Cuda";
        case DeviceType::OPENGL:
          return lower_case ? "opengl" : "OPENGL";
        case DeviceType::OPENCL:
          return lower_case ? "opencl" : "OPENCL";
        case DeviceType::MKLDNN:
          return lower_case ? "mkldnn" : "MKLDNN";
        case DeviceType::IDEEP:
          return lower_case ? "ideep" : "IDEEP";
        case DeviceType::HIP:
          return lower_case ? "hip" : "HIP";
        case DeviceType::FPGA:
          return lower_case ? "fpga" : "FPGA";
        case DeviceType::MSNPU:
          return lower_case ? "msnpu" : "MSNPU";
        case DeviceType::XLA:
          return lower_case ? "xla" : "XLA";
        case DeviceType::MLC:
          return lower_case ? "mlc" : "MLC";
        case DeviceType::Vulkan:
          return lower_case ? "vulkan" : "VULKAN";
        case DeviceType::Metal:
          return lower_case ? "metal" : "METAL";
        case DeviceType::XPU:
          return lower_case ? "xpu" : "XPU";
        case DeviceType::Meta:
          return lower_case ? "meta" : "META";
        case DeviceType::HPU:
          return lower_case ? "hpu" : "HPU";
        default:
          TORCH_CHECK(
              false,
              "Unknown device: ",
              static_cast<int16_t>(d),
              ". If you have recently updated the caffe2.proto file to add a new "
              "device type, did you forget to update the DeviceTypeName() "
              "function to reflect such recent changes?");
          // The below code won't run but is needed to suppress some compiler
          // warnings.
          return "";
      }
        */
}

/**
  | NB: Per the C++ standard (e.g.,
  | https://stackoverflow.com/questions/18195312/what-happens-if-you-static-cast-invalid-value-to-enum-class)
  | as long as you cast from the same underlying
  | type, it is always valid to cast into an enum
  | class (even if the value would be invalid by
  | the enum.)
  |
  | Thus, the caller is allowed to cast a possibly
  | invalid int16_t to DeviceType and then pass it
  | to this function.
  |
  | (I considered making this function take an
  | int16_t directly, but that just seemed weird.)
  |
  */
pub fn is_valid_device_type(d: DeviceType) -> bool {
    
    todo!();
        /*
            switch (d) {
        case DeviceType::CPU:
        case DeviceType::CUDA:
        case DeviceType::OPENGL:
        case DeviceType::OPENCL:
        case DeviceType::MKLDNN:
        case DeviceType::IDEEP:
        case DeviceType::HIP:
        case DeviceType::FPGA:
        case DeviceType::MSNPU:
        case DeviceType::XLA:
        case DeviceType::MLC:
        case DeviceType::Vulkan:
        case DeviceType::Metal:
        case DeviceType::XPU:
        case DeviceType::Meta:
        case DeviceType::HPU:
          return true;
        default:
          return false;
      }
        */
}

impl fmt::Display for DeviceType {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            stream << DeviceTypeName(type, /* lower case */ true);
      return stream;
        */
    }
}
