crate::ix!();

#[inline] pub fn fetch_inputs_to_if_ops_subnet(net: *mut NetDef)  {
    
    todo!();
    /*
        NetDef clone(*net);
      clone.clear_op();
      for (auto& op : net->op()) {
        if (op.type() == "If" || op.type() == "AsyncIf") {
          OperatorDef new_op(op);
          ArgumentHelper helper(op);
          std::set<std::string> subnet_inputs, subnet_outputs;
          if (helper.HasSingleArgumentOfType<NetDef>("then_net")) {
            auto then_net = helper.GetSingleArgument<NetDef>("then_net", NetDef());
            for (const auto& nested_op : then_net.op()) {
              collectInputsAndOutputs(nested_op, &subnet_inputs, &subnet_outputs);
            }
          }
          if (helper.HasSingleArgumentOfType<NetDef>("else_net")) {
            auto else_net = helper.GetSingleArgument<NetDef>("else_net", NetDef());
            for (const auto& nested_op : else_net.op()) {
              collectInputsAndOutputs(nested_op, &subnet_inputs, &subnet_outputs);
            }
          }
          for (const std::string& blob : subnet_inputs) {
            if (subnet_outputs.count(blob) == 0) {
              new_op.add_input(blob);
            }
          }
          clone.add_op()->CopyFrom(new_op);
        } else {
          clone.add_op()->CopyFrom(op);
        }
      }
      net->Swap(&clone);
    */
}
