crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/detail/CPUGuardImpl.cpp]

c10_register_guard_impl!{
    CPU, 
    NoOpDeviceGuardImpl<DeviceType::CPU>
}

