crate::ix!();

#[inline] pub fn add_apply_link_ops(
    links:          &Vec<Link>,
    timestep:       String,
    device_option:  &DeviceOption,
    netdef:         *mut NetDef)  
{
    todo!();
    /*
        std::vector<OperatorDef> ops;
      for (auto& link : links) {
        OperatorDef opdef;
        opdef.set_type("rnn_internal_apply_link");
        opdef.add_input(timestep);
        opdef.add_input(link.external);
        opdef.add_output(link.internal);
        opdef.add_output(link.external);
        opdef.mutable_device_option()->CopyFrom(device_option);

        Argument* offset_arg = opdef.add_arg();
        offset_arg->set_name("offset");
        offset_arg->set_i(link.offset);

        Argument* window_arg = opdef.add_arg();
        window_arg->set_name("window");
        window_arg->set_i(link.window);

        // Find out if the linked blob is used first as an output: then we need
        // to add control_input to that op
        for (auto& op : *netdef->mutable_op()) {
          if (HasInput(op, link.internal)) {
            // First appears as an input, no need to do antyhing
            continue;
          }
          if (HasOutput(op, link.internal)) {
            op.add_control_input(link.internal);
            break;
          }
        }

        ops.push_back(opdef);

        netdef->add_external_input(link.internal);
        netdef->add_external_input(link.external);
      }

      detail::PrependOps(ops, netdef);
    */
}
