crate::ix!();

#[inline] pub fn cost_inference_for_sum(
    def:   &OperatorDef,
    input: &Vec<TensorShape>) -> OpSchemaCost {
    
    todo!();
    /*
        struct OpSchema::Cost cost = PointwiseCostInference<1>(def, in);
      cost.flops *= (in.size() - 1);
      cost.params_bytes = 0;
      return cost;
    */
}

#[inline] pub fn weighted_sum_shape_inference(
    unused: &OperatorDef,
    input:  &Vec<TensorShape>) -> Vec<TensorShape> {
    
    todo!();
    /*
        vector<TensorShape> out(1);
      out[0] = in[0];
      return out;
    */
}

#[inline] pub fn cost_inference_for_weighted_sum(
    unused: &OperatorDef,
    input:  &Vec<TensorShape>) -> OpSchemaCost 
{
    todo!();
    /*
        CAFFE_ENFORCE_EQ(
          in.size() % 2, 0, "WeightedSum requires an even number of inputs");
      struct OpSchema::Cost c;

      const auto& X0 = in[0];
      const auto& nElem = nElemFromDim(X0);
      const auto& nInputs = in.size();
      c.flops = (nInputs - 1) * nElem;
      c.bytes_read = (nInputs / 2) * (nElem + 1) * sizeof(X0.data_type());
      c.bytes_written = nElem * sizeof(X0.data_type());
      c.params_bytes = (nInputs / 2) * sizeof(X0.data_type());
      return c;
    */
}
