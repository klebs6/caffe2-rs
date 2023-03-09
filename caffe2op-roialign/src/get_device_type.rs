crate::ix!();

pub trait GetDeviceType {
    fn get_device_type() -> DeviceType;
}

impl GetDeviceType for CPUContext {
    fn get_device_type() -> DeviceType {
        todo!();
        /*
        DeviceType::CPU
        */
    }
}

impl GetDeviceType for CUDAContext {

    fn get_device_type() -> DeviceType {
        todo!();
        /*
        DeviceType::CUDA
        */
    }
}
