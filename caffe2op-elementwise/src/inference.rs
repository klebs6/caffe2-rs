crate::ix!();

#[inline] pub fn elementwise_op_shape_inference(
    def:   &OperatorDef,
    input: &Vec<TensorShape>) -> Vec<TensorShape> 
{
    todo!();
    /*
        std::vector<TensorShape> out(1);
      out[0].set_data_type(in[0].data_type());
      ArgumentHelper helper(def);
      const bool broadcast = helper.GetSingleArgument<bool>("broadcast", false);
      if (broadcast) {
        out[0].mutable_dims()->CopyFrom(in[0].dims());
      } else {
        const std::vector<int> A_dims(in[0].dims().begin(), in[0].dims().end());
        const std::vector<int> B_dims(in[1].dims().begin(), in[1].dims().end());
        const std::vector<int> C_dims =
            elementwise_ops_utils::ComputeBinaryBroadcastForwardDims(
                A_dims, B_dims);
        for (const int dim : C_dims) {
          out[0].add_dims(dim);
        }
      }
      return out;
    */
}

#[inline] pub fn elementwise_gradient_op_shape_inference(
    def:   &OperatorDef,
    input: &Vec<TensorShape>) -> Vec<TensorShape> 
{
    todo!();
    /*
        std::vector<TensorShape> out;
      out.push_back(in.at(1));
      out.push_back(in.at(2));
      return out;
    */
}
