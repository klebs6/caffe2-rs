crate::ix!();

/**
  | User can set the preferred engines as
  | a list of engine names, in descending
  | order of preference.
  |
  */
pub type EnginePrefType = Vec<String>;

/// {device_type -> {operator_name -> EnginePrefType}}
pub type PerOpEnginePrefType = HashMap<DeviceType,HashMap<String,EnginePrefType>>;

/// {device_type -> EnginePrefType}
pub type GlobalEnginePrefType = HashMap<DeviceType,EnginePrefType>;

#[inline] pub fn g_per_op_engine_pref<'a>() -> &'a mut PerOpEnginePrefType {
    
    todo!();
    /*
        static auto* g_per_op_engine_pref_ = new PerOpEnginePrefType();
      return *g_per_op_engine_pref_;
    */
}

#[inline] pub fn g_global_engine_pref<'a>() -> &'a mut GlobalEnginePrefType {
    
    todo!();
    /*
        static auto* g_global_engine_pref_ =
          new GlobalEnginePrefType{{CUDA, {"CUDNN"}}, {HIP, {"MIOPEN"}}};
      return *g_global_engine_pref_;
    */
}

#[inline] pub fn set_per_op_engine_pref(
    per_op_engine_pref: &PerOpEnginePrefType)
{
    todo!();
    /*
        for (const auto& device_pref_pair : per_op_engine_pref) {
        const auto& device_type = device_pref_pair.first;
        CAFFE_ENFORCE(
            gDeviceTypeRegistry()->count(device_type),
            "Device type ",
            device_type,
            " not registered.");
        auto* registry = gDeviceTypeRegistry()->at(device_type);

        for (const auto& op_pref_pair : device_pref_pair.second) {
          const auto& op_type = op_pref_pair.first;
          CAFFE_ENFORCE(
              registry->Has(op_type),
              "Operator type ",
              op_type,
              " not registered in ",
              device_type,
              " registry.");
        }
      }
      g_per_op_engine_pref() = per_op_engine_pref;
    */
}


#[inline] pub fn set_global_engine_pref(
    global_engine_pref: &GlobalEnginePrefType)
{
    todo!();
    /*
        for (const auto& device_pref_pair : global_engine_pref) {
        const auto& device_type = device_pref_pair.first;
        CAFFE_ENFORCE(
            gDeviceTypeRegistry()->count(device_type),
            "Device type ",
            device_type,
            " not registered.");
      }
      g_global_engine_pref() = global_engine_pref;
    */
}

#[inline] pub fn set_engine_pref(
    per_op_engine_pref: &PerOpEnginePrefType,
    global_engine_pref: &GlobalEnginePrefType)
{
    
    todo!();
    /*
        SetPerOpEnginePref(per_op_engine_pref);
      SetGlobalEnginePref(global_engine_pref);
    */
}

#[inline] pub fn set_op_engine_pref(
    op_type: &String,
    op_pref: &HashMap<DeviceType, EnginePrefType>)  
{
    todo!();
    /*
        for (const auto& device_pref_pair : op_pref) {
        const auto& device_type_proto = device_pref_pair.first;
        const auto& device_type =
            ProtoToType(static_cast<DeviceTypeProto>(device_type_proto));
        CAFFE_ENFORCE(
            gDeviceTypeRegistry()->count(device_type),
            "Device type ",
            device_type,
            " not registered.");
        CAFFE_ENFORCE(
            gDeviceTypeRegistry()->at(device_type)->Has(op_type),
            "Operator type ",
            op_type,
            " not registered in ",
            device_type,
            " registry.");
        g_per_op_engine_pref()[device_type][op_type] = device_pref_pair.second;
      }
    */
}


