crate::ix!();

#[inline] pub fn push_op_to_front(op: &mut OperatorDef, net: *mut NetDef)  {
    
    todo!();
    /*
        *net->add_op() = op;
      google::protobuf::RepeatedPtrField<caffe2::OperatorDef>* op_list(
          net->mutable_op());
      // Reverse iterate, swapping new element in front each time
      for (int i(net->op_size() - 1); i > 0; --i) {
        op_list->SwapElements(i, i - 1);
      }
    */
}
