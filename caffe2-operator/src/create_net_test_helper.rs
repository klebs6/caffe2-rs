crate::ix!();

#[inline] pub fn create_net_test_helper(
    ws:     *mut Workspace,
    input:  &Vec<String>,
    output: &Vec<String>) -> Box<NetBase> 
{
    todo!();
    /*
        NetDef net_def;
      {
        auto& op = *(net_def.add_op());
        op.set_type("NetTestDummy");
        op.add_input("in");
        op.add_output("hidden");
      }
      {
        auto& op = *(net_def.add_op());
        op.set_type("NetTestDummy");
        op.add_input("hidden");
        op.add_output("out");
      }

      for (const auto& name : input) {
        net_def.add_external_input(name);
      }
      for (const auto& name : output) {
        net_def.add_external_output(name);
      }
      return CreateNet(net_def, ws);
    */
}

