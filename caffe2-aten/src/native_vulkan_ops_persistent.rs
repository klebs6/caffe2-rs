// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/ops/Persistent.h]
//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/ops/Persistent.cpp]

/**
  | This class is meant for allocation of resources
  | that will persist through the execution of the
  | program, or until they are explicitly free'd by
  | this code's clients, and its usage pattern is
  | in direct contrast with the primary resource
  | pool from which tensors draw from.
  |
  | Whereas the primary resource pool is purged in
  | its entirety at the end of each inference run,
  | the intended usage pattern for this class is
  | such that it delegates object lifetime
  | management to the users so resources can stick
  | around for as long as required.
  |
  | This is ideal for prepacked weights, or
  | scnearios where a precomputed or
  | once-transformed data can be stored and reused
  | in subsequent runs.
  |
  */
pub struct Persistent {
    pool: ResourcePool,
}

pub fn persistent() -> *mut Persistent {
    
    todo!();
        /*
            static const unique_ptr<Persistent> persistent(
        []() -> Persistent* {
          try {
            return new Persistent{
              api::Resource::Pool{
                api::context()->gpu(),
              },
            };
          }
          catch (const exception& e) {
            TORCH_WARN(
                "Vulkan: Failed to initialize persistent resource pool! Error: ",
                e.what());
          }
          catch (...) {
            TORCH_WARN(
                "Vulkan: Failed to initialize persistent resource pool! "
                "Error: Unknown");
          }

          return nullptr;
        }());

      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          persistent,
          "Vulkan: Invalid persistent pool!");

      return persistent.get();
        */
}
