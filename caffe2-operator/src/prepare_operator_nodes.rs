crate::ix!();

#[inline] pub fn prepare_operator_nodes(net_def: &Arc<NetDef>, ws: *mut Workspace) -> Vec<OperatorNode> {
    
    todo!();
    /*
        std::vector<OperatorNode> operator_nodes(net_def->op_size());
      std::map<string, int> blob_creator;
      std::map<string, std::set<int>> blob_readers;
      bool net_def_has_device_option = net_def->has_device_option();
      // Initialize the operators
      for (int idx = 0; idx < net_def->op_size(); ++idx) {
        const OperatorDef& op_def = net_def->op(idx);
        VLOG(1) << "Creating operator #" << idx << ": " << op_def.name() << ": "
                << op_def.type();
        if (net_def_has_device_option) {
          OperatorDef temp_def(op_def);

          DeviceOption temp_dev(net_def->device_option());
          temp_dev.MergeFrom(op_def.device_option());

          temp_def.mutable_device_option()->CopyFrom(temp_dev);
          operator_nodes[idx].operator_ = CreateOperator(temp_def, ws, idx);
        } else {
          auto op = CreateOperator(op_def, ws, idx);
          op->set_debug_def(
              std::shared_ptr<const OperatorDef>{net_def, &(net_def->op(idx))});
          operator_nodes[idx].operator_ = std::move(op);
        }
        // Check the inputs, and set up parents if necessary. This addressese the
        // read after write case.
        auto checkInputs =
            [&](const google::protobuf::RepeatedPtrField<std::string>& inputs) {
              for (const string& input : inputs) {
                if (blob_creator.count(input) == 0) {
                  VLOG(1) << "Input " << input << " not produced by this net. "
                          << "Assuming it is pre-existing.";
                } else {
                  int parent = blob_creator[input];
                  VLOG(1) << "op dependency (RaW " << input << "): " << parent
                          << "->" << idx;
                  operator_nodes[idx].parents_.push_back(parent);
                  operator_nodes[parent].children_.push_back(idx);
                }
                // Add the current idx to the readers of this input.
                blob_readers[input].insert(idx);
              }
            };
        checkInputs(op_def.input());
        checkInputs(op_def.control_input());

        // Check the outputs.
        for (const string& output : op_def.output()) {
          if (blob_creator.count(output) != 0) {
            // This addresses the write after write case - we will assume that all
            // writes are inherently sequential.
            int waw_parent = blob_creator[output];
            VLOG(1) << "op dependency (WaW " << output << "): " << waw_parent
                    << "->" << idx;
            operator_nodes[idx].parents_.push_back(waw_parent);
            operator_nodes[waw_parent].children_.push_back(idx);
          }
          // This addresses the write after read case - we will assume that writes
          // should only occur after all previous reads are finished.
          for (const int war_parent : blob_readers[output]) {
            VLOG(1) << "op dependency (WaR " << output << "): " << war_parent
                    << "->" << idx;
            operator_nodes[idx].parents_.push_back(war_parent);
            operator_nodes[war_parent].children_.push_back(idx);
          }
          // Renew the creator of the output name.
          blob_creator[output] = idx;
          // The write would create an implicit barrier that all earlier readers of
          // this output is now parents of the current op, and future writes would
          // not need to depend on these earlier readers. Thus, we can clear up the
          // blob readers.
          blob_readers[output].clear();
        }
      }

      // Now, make sure that the parent list and the children list do not contain
      // duplicated items.
      for (int i = 0; i < (int)operator_nodes.size(); ++i) {
        auto& node = operator_nodes[i];
        // Sort, remove duplicates, and delete self dependency.
        auto& p = node.parents_;
        std::sort(p.begin(), p.end());
        p.erase(std::unique(p.begin(), p.end()), p.end());
        p.erase(std::remove(p.begin(), p.end(), i), p.end());
        // Do the same for the children vector.
        auto& c = node.children_;
        std::sort(c.begin(), c.end());
        c.erase(std::unique(c.begin(), c.end()), c.end());
        c.erase(std::remove(c.begin(), c.end(), i), c.end());
      }

      return operator_nodes;
    */
}

