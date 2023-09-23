/*!
  | Are you here because you're wondering why
  | DeviceGuard(tensor) no longer works?  For code
  | organization reasons, we have temporarily(?)
  | removed this constructor from DeviceGuard.  The
  | new way to spell it is:
  |
  |    OptionalDeviceGuard guard(device_of(tensor));
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/DeviceGuard.h]

/**
  | Return the Device of a Tensor, if the
  | Tensor is defined.
  |
  */
#[inline] pub fn device_of(t: &Tensor) -> Option<Device> {
    
    todo!();
        /*
            if (t.defined()) {
        return make_optional(t.device());
      } else {
        return nullopt;
      }
        */
}

#[inline] pub fn device_of_maybe_tensor(t: &Option<Tensor>) -> Option<Device> {
    
    todo!();
        /*
            return t.has_value() ? device_of(t.value()) : nullopt;
        */
}

/**
  | Return the Device of a &[Tensor], if the list
  | is non-empty and the first Tensor is defined.
  | (This function implicitly assumes that all
  | tensors in the list have the same device.)
  */
#[inline] pub fn device_of_tensor_list(t: &[Tensor]) -> Option<Device> {
    
    todo!();
        /*
            if (!t.empty()) {
        return device_of(t.front());
      } else {
        return nullopt;
      }
        */
}
