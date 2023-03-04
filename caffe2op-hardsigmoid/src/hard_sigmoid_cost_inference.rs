crate::ix!();

#[inline] pub fn cost_inference_for_hard_sigmoid(
    def: &OperatorDef,
    input: &Vec<TensorShape>) -> OpSchemaCost 
{
    todo!();
    /*
        struct OpSchema::Cost cost = PointwiseCostInference<4>(def, in);
      cost.params_bytes = 0;
      return cost;
    */
}
