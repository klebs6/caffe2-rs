crate::ix!();

#[inline] pub fn check_net(
    net:          &mut NetDef,
    expected_net: &mut NetDef)  
{

    todo!();
    /*
        CHECK_EQ(net.op().size(), expected_net.op().size());
      for (int i = 0; i < net.op().size(); i++) {
        auto& op1 = net.op(i);
        auto& op2 = expected_net.op(i);
        CHECK_EQ(op1.type(), op2.type());
        CHECK_EQ(op1.input_size(), op2.input_size());
        CHECK_EQ(op1.output_size(), op2.output_size());
        for (int j = 0; j < op1.input_size(); j++) {
          CHECK_EQ(op1.input(j), op2.input(j));
        }
        for (int j = 0; j < op1.output_size(); j++) {
          CHECK_EQ(op1.output(j), op2.output(j));
        }
        CHECK_EQ(
            op1.device_option().device_type(), op2.device_option().device_type());
        ArgumentHelper helper1(op1);
        ArgumentHelper helper2(op2);
        for (auto& arg : op1.arg()) {
          const auto& name = arg.name();
          if (name == "net_pos") {
            continue;
          }
          CHECK(helper2.HasArgument(name))
              << "Argument " << name << " doesn't exist";
          const auto arg1 = helper1.GetSingleArgument<int>(name, 0);
          const auto arg2 = helper2.GetSingleArgument<int>(name, 0);
          CHECK_EQ(arg1, arg2);
        }
      }
    */
}

