crate::ix!();

pub struct DeviceTypeRegisterer {
    
}

impl DeviceTypeRegisterer {
    
    pub fn new_with_device_type_and_registry_function<'a>(
        ty:   DeviceType, 
        func: RegistryFunction<'a>) -> Self {
    
        todo!();
        /*
            if (gDeviceTypeRegistry()->count(type)) {
          std::cerr << "Device type " << DeviceTypeName(type)
                    << "registered twice. This should not happen. Did you have "
                       "duplicated numbers assigned to different devices?";
          std::exit(1);
        }
        // Calling the registry function to get the actual registry pointer.
        gDeviceTypeRegistry()->emplace(type, func());
        */
    }
}

