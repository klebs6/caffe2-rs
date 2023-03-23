crate::ix!();

/**
  | Adds an operator def to a netdef.
  |
  | Returns the ptr, if you want to add anything
  | extra (such as device_option)
  */
#[inline] pub fn add_op(
    netdef_ptr: *mut NetDef,
    op_type:    String,
    inputs:     Vec<String>,
    outputs:    Vec<String>) -> *mut OperatorDef 
{

    todo!();
    /*
        CHECK(netdef_ptr);
      auto& netdef = *netdef_ptr;
      auto op_ptr = netdef.add_op();
      auto& op = *op_ptr;
      op.set_type(op_type);
      for (const string& inp : inputs) {
        op.add_input(inp);
      }
      for (const string& outp : outputs) {
        op.add_output(outp);
      }
      return op_ptr;
    */
}
