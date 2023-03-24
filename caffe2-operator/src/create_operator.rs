crate::ix!();

#[inline] pub fn try_create_operator<Context>(
    key:          &String,
    operator_def: &OperatorDef,
    ws:           *mut Workspace) -> Box<OperatorStorage> 
{
    
    todo!();
    /*
        const auto& type_proto = operator_def.device_option().device_type();
      const auto& type = ProtoToType(static_cast<DeviceTypeProto>(type_proto));
      CAFFE_ENFORCE(
          gDeviceTypeRegistry()->count(type),
          "Device type ",
          type,
          " not registered.");
      OperatorRegistry* registry = gDeviceTypeRegistry()->at(type);
      VLOG(1) << "Creating operator with device type " << type;
      try {
        return registry->Create(key, operator_def, ws);
      } catch (const UnsupportedOperatorFeature& err) {
        LOG(WARNING) << "Operator " << operator_def.type()
                     << " does not support the requested feature. Msg: "
                     << err.what()
                     << ". Proto is: " << ProtoDebugString(operator_def);
        return nullptr;
      }
    */
}

#[inline] pub fn create_operator<Context>(
    operator_def: &OperatorDef,
    ws: *mut Workspace) -> Box<OperatorStorage> 
{
    todo!();
    /*
        static StaticLinkingProtector g_protector;
      const auto& op_type = operator_def.type();
      const auto& device_type_proto = operator_def.device_option().device_type();
      const auto& device_type =
          ProtoToType(static_cast<DeviceTypeProto>(device_type_proto));

    #ifndef CAFFE2_NO_OPERATOR_SCHEMA
      // first, check with OpSchema if the operator is legal.
      auto* schema = OpSchemaRegistry::Schema(op_type);
      if (schema) {
        CAFFE_ENFORCE(
            schema->Verify(operator_def),
            "Operator def did not pass schema checking: ",
            ProtoDebugString(operator_def));
      } else {
        // We would like to recommend every op to register its schema, so if there
        // is not one, we print a LOG_ERROR. But we will still allow the operator
        // to be constructed.
        LOG(ERROR) << "Cannot find operator schema for " << op_type
                   << ". Will skip schema checking.";
      }
    #endif

      // second try engines specified in the operator_def and preferred engines
      std::vector<std::string> engines{};
      if (operator_def.engine().size()) {
        const auto op_def_engines = split(',', operator_def.engine());
        engines.insert(engines.end(), op_def_engines.begin(), op_def_engines.end());
      }
      if (!FLAGS_caffe2_disable_implicit_engine_preference &&
          g_per_op_engine_pref().count(device_type) &&
          g_per_op_engine_pref()[device_type].count(op_type)) {
        const auto& preferred_engines =
            g_per_op_engine_pref()[device_type][op_type];
        VLOG(2) << "Inserting per-op engine preference: " << preferred_engines;
        engines.insert(
            engines.end(), preferred_engines.begin(), preferred_engines.end());
      }
      if (!FLAGS_caffe2_disable_implicit_engine_preference &&
          g_global_engine_pref().count(device_type)) {
        const auto& preferred_engines = g_global_engine_pref()[device_type];
        VLOG(2) << "Inserting global engine preference: " << preferred_engines;
        engines.insert(
            engines.end(), preferred_engines.begin(), preferred_engines.end());
      }
      for (const auto& engine : engines) {
        const std::string key = OpRegistryKey(op_type, engine);
        VLOG(1) << "Trying to create operator " << op_type << " with engine "
                << engine;
        auto op = TryCreateOperator(key, operator_def, ws);
        if (op) {
          if (engine.size() <=
              (unsigned)FLAGS_caffe2_operator_max_engine_name_length) {
            op->annotate_engine(engine);
          } else {
            op->annotate_engine(
                engine.substr(0, FLAGS_caffe2_operator_max_engine_name_length));
          }
          return op;
        } else {
          // If the above fails, we will just return the normal case with the
          // default implementation.
          VLOG(1) << "Engine " << engine << " is not available for operator "
                  << op_type << ".";
        }
      }
      if (operator_def.engine().size() && !VLOG_IS_ON(1)) {
        static int log_occurrences = 0;
        if (log_occurrences <= 64) {
          ++log_occurrences;
          LOG(INFO) << "Engine " << operator_def.engine()
                    << " is not available for operator " << op_type << ".";
        }
      }
      VLOG(1) << "Using default implementation.";

      // Lastly, if the engine does not work here, try using the default engine.
      auto op = TryCreateOperator(op_type, operator_def, ws);
      CAFFE_ENFORCE(
          op,
          "Cannot create operator of type '",
          op_type,
          "' on the device '",
          DeviceTypeName(device_type),
          "'. Verify that implementation for the corresponding device exist. It "
          "might also happen if the binary is not linked with the operator "
          "implementation code. If Python frontend is used it might happen if "
          "dyndep.InitOpsLibrary call is missing. Operator def: ",
          ProtoDebugString(operator_def));
      return op;
    */
}

/**
  | Creates an operator with the given operator
  | definition.
  |
  | Throws on error and never returns nullptr
  */
#[inline] pub fn create_operator_with_net_position<Context>(
    operator_def: &OperatorDef,
    ws:           *mut Workspace,
    net_position: Option<i32>) -> Box<OperatorStorage> 
{
    let net_position: i32 = net_position.unwrap_or(kNoNetPositionSet);

    todo!();
    /*
        try {
        auto op = _CreateOperator(operator_def, ws);
        op->set_net_position(net_position);
        return op;
      } catch (...) {
        if (net_position != 0) {
          VLOG(1) << "Operator constructor with net position " << net_position
                  << " failed";
          ws->last_failed_op_net_position = net_position;
        } else {
          VLOG(1) << "Failed operator constructor doesn't have an id set";
        }
        throw;
      }
    */
}
