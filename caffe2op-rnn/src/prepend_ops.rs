crate::ix!();

#[inline] pub fn prepend_ops(
    ops: Vec<OperatorDef>,
    netdef: *mut NetDef)  
{
    todo!();
    /*
        for (auto& o : netdef->op()) {
        ops.push_back(o);
      }
      netdef->mutable_op()->Clear();
      for (auto& o : ops) {
        auto* ao = netdef->add_op();
        ao->CopyFrom(o);
      }
    */
}
