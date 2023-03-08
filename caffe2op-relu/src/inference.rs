crate::ix!();

#[inline] pub fn cost_inference_for_relu(
    def:   &OperatorDef, 
    input: &Vec<TensorShape>) -> OpSchemaCost {
    
    todo!();
    /*
        struct OpSchema::Cost cost = PointwiseCostInference<0>(def, in);
      cost.params_bytes = 0;
      return cost;
    */
}

#[inline] pub fn cost_inference_for_relun(
    def:   &OperatorDef, 
    input: &Vec<TensorShape>) -> OpSchemaCost {
    
    todo!();
    /*
        struct OpSchema::Cost cost = PointwiseCostInference<2>(def, in);
      cost.params_bytes = 0;
      return cost;
    */
}
