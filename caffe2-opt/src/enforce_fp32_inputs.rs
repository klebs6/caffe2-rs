crate::ix!();

#[inline] pub fn enforce_fp_32inputs_to_fp16(
    weights:     &HashSet<String>,
    pred_net:    *mut NetDef,
    shape_hints: *mut ShapeInfoMap)  {
    
    todo!();
    /*
        std::unordered_map<std::string, ShapeInfo> user_input_map;
      for (const auto& i : pred_net->external_input()) {
        if (weights.count(i)) {
          continue;
        }
        auto it = shape_hints->find(i);
        if (it == shape_hints->end() ||
            it->second.shape.data_type() != TensorProto_DataType_FLOAT) {
          continue;
        }
        auto& shape_info = it->second;
        user_input_map[i] = shape_info;
        shape_info.shape.set_data_type(TensorProto_DataType_FLOAT16);
      }

      if (user_input_map.empty()) {
        return;
      }

      std::vector<OperatorDef> ops;
      for (const auto& op : pred_net->op()) {
        ops.emplace_back(op);
      }
      pred_net->clear_op();
      int current_pos = ops.size();

      const char kBridgeTensorSuffix[] = "_to_float_bridge";
      std::vector<OperatorDef> converts;
      for (const auto& elem : user_input_map) {
        const auto& name = elem.first;
        const auto& shape_info = elem.second;
        std::string new_name = name + kBridgeTensorSuffix;
        shape_hints->emplace(new_name, shape_info);
        converts.emplace_back(CreateOperatorDef(
            "HalfToFloat",
            "",
            {name},
            {new_name},
            {MakeArgument<int>(kNetPos, current_pos++)}));
      }
      for (const auto& op : converts) {
        pred_net->add_op()->CopyFrom(op);
      }

      for (auto& op : ops) {
        for (auto& input : *op.mutable_input()) {
          if (user_input_map.count(input)) {
            input += kBridgeTensorSuffix;
          }
        }
      }

      for (const auto& op : ops) {
        pred_net->add_op()->CopyFrom(op);
      }
    */
}

