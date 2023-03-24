crate::ix!();

#[inline] pub fn g_device_type_registry<'a>() 
-> *mut HashMap<DeviceType, *mut OperatorRegistry<'a>> 
{
    todo!();
    /*
        static std::map<DeviceType, OperatorRegistry*> g_device_type_registry;
      return &g_device_type_registry;
    */
}

