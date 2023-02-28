crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/adaption.cpp]

/**
  | A string describing which function did checks
  | on its input arguments.
  |
  | TODO: Consider generalizing this into a call
  | stack.
  |
  */
pub type CheckedFrom = *const u8;

pub fn common_device_check_failure(
    common_device: &mut Option<Device>,
    tensor:        &Tensor,
    method_name:   CheckedFrom,
    arg_name:      CheckedFrom)  {

    todo!();
        /*
            TORCH_CHECK(false,
        "Expected all tensors to be on the same device, but "
        "found at least two devices, ", common_device.value(), " and ", tensor.device(), "! "
        "(when checking arugment for argument ", argName, " in method ", methodName, ")");
        */
}
