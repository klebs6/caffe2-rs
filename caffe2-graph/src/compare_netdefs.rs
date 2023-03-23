crate::ix!();

/**
  | Checks if two netdefs are in terms of
  | type, input, and output.
  |
  */
#[inline] pub fn compare_netdefs(
    net_a: &NetDef,
    net_b: &NetDef)  
{
    
    todo!();
    /*
        EXPECT_EQ(net_a.op_size(), net_b.op_size());
      for (int i = 0; i < net_a.op_size(); i++) {
        EXPECT_EQ(net_a.op(i).type(), net_b.op(i).type());
        EXPECT_EQ(net_a.op(i).input_size(), net_b.op(i).input_size());
        for (int j = 0; j < net_a.op(i).input_size(); j++) {
          EXPECT_EQ(net_a.op(i).input(j), net_b.op(i).input(j));
        }
        EXPECT_EQ(net_a.op(i).output_size(), net_b.op(i).output_size());
        for (int j = 0; j < net_a.op(i).output_size(); j++) {
          EXPECT_EQ(net_a.op(i).output(j), net_b.op(i).output(j));
        }
      }
    */
}
