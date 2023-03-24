crate::ix!();

#[inline] pub fn merge_fp_32inputs_and_convert_to_fp16(
    batch_size:  usize,
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
        const auto it = shape_hints->find(i);
        // Heuristic: the input has to be of float type, 2-dimensional and the first
        // dimension has to be of batch size
        if (it == shape_hints->end() ||
            it->second.shape.data_type() != TensorProto_DataType_FLOAT) {
          continue;
        }
        auto shape_info = it->second;
        if (shape_info.shape.dims_size() != 2 ||
            shape_info.shape.dims(0) != batch_size) {
          continue;
        }
        shape_info.shape.set_data_type(TensorProto_DataType_FLOAT16);

        user_input_map[i] = shape_info;
      }

      if (user_input_map.empty()) {
        return;
      }
      std::unordered_map<std::string, std::vector<std::string>>
          user_inputs_by_partition;
      std::unordered_map<std::string, std::unordered_set<std::string>>
          user_input_set_by_partition;
      for (const auto& op : pred_net->op()) {
        for (const auto& i : op.input()) {
          if (user_input_map.find(i) != user_input_map.end()) {
            const auto& partition = op.device_option().node_name().empty()
                ? "default"
                : op.device_option().node_name();
            if (user_input_set_by_partition[partition].find(i) ==
                user_input_set_by_partition[partition].end()) {
              user_inputs_by_partition[partition].emplace_back(i);
              user_input_set_by_partition[partition].insert(i);
            }
          }
        }
      }

      std::vector<OperatorDef> ops;
      for (const auto& op : pred_net->op()) {
        ops.emplace_back(op);
      }
      pred_net->clear_op();
      int current_pos = ops.size();

      for (const auto& elem : user_inputs_by_partition) {
        const auto& partition = elem.first;
        const auto& user_inputs = elem.second;
        const auto& user_input_set = user_input_set_by_partition[partition];

        OperatorDef op1;
        op1.set_type("Concat");
        for (const auto& i : user_inputs) {
          op1.add_input(i);
        }
        op1.add_output(partition + "_fp32_input_concated");
        op1.add_output(partition + "_fp32_input_concated_split_info");
        auto shape_info = user_input_map[user_inputs.front()];
        int total = 0;
        for (const auto& u : user_inputs) {
          total += user_input_map[u].shape.dims(1);
        }
        shape_info.shape.set_dims(1, total);
        AddArgument("axis", 1, &op1);
        AddArgument(kNetPos, current_pos++, &op1);
        pred_net->add_op()->CopyFrom(op1);

        // TODO: a possible optimization is to fuse the fp16 conversion into Concat
        OperatorDef op2;
        op2.set_type("FloatToHalf");
        op2.add_input(partition + "_fp32_input_concated");
        op2.add_output(partition + "_fp16_input_concated");
        AddArgument("clip", 1, &op2);
        AddArgument(kNetPos, current_pos++, &op2);
        shape_hints->emplace(partition + "_fp16_input_concated", shape_info);
        pred_net->add_op()->CopyFrom(op2);

        OperatorDef op3;
        op3.set_type("Split");
        op3.add_input(partition + "_fp16_input_concated");
        op3.mutable_device_option()->set_node_name(partition);

        std::vector<OperatorDef> converts;
        for (const auto& i : user_inputs) {
          std::string new_name = partition + "_" + i + "_split_fp16";
          op3.add_output(new_name);
          shape_hints->emplace(new_name, user_input_map[i]);
          converts.emplace_back(CreateOperatorDef(
              "HalfToFloat",
              "",
              {partition + "_" + i + "_split_fp16"},
              {partition + "_" + i + "_split"},
              {MakeArgument<int>(kNetPos, current_pos++)}));
          converts.back().mutable_device_option()->set_node_name(partition);

          auto converted_shape = user_input_map[i];
          converted_shape.shape.set_data_type(TensorProto_DataType_FLOAT);
          shape_hints->emplace(partition + "_" + i + "_split", converted_shape);
        }
        AddArgument("axis", 1, &op3);
        AddArgument(kNetPos, current_pos++, &op3);
        auto* arg = op3.add_arg();
        arg->set_name("split");
        for (const auto& u : user_inputs) {
          arg->add_ints(user_input_map[u].shape.dims(1));
        }
        pred_net->add_op()->CopyFrom(op3);
        for (const auto& op : converts) {
          pred_net->add_op()->CopyFrom(op);
        }

        for (auto& op : ops) {
          if ((!op.device_option().node_name().empty() &&
               op.device_option().node_name() == partition) ||
              (op.device_option().node_name().empty() && partition == "default")) {
            for (auto& i : *op.mutable_input()) {
              if (user_input_set.count(i)) {
                i = partition + "_" + i + "_split";
              }
            }
          }
        }
      }

      for (const auto& op : ops) {
        pred_net->add_op()->CopyFrom(op);
      }
    */
}

