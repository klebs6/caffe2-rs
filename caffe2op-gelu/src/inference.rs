crate::ix!();

#[inline] pub fn cost_inference_for_gelu(
    def: &OperatorDef,
    input: &Vec<TensorShape>) -> OpSchemaCost 
{
    todo!();
    /*
        struct OpSchema::Cost cost = PointwiseCostInference<2>(def, in);
      cost.params_bytes = 0;
      return cost;
    */
}

cost_inference_function!{Gelu, CostInferenceForGelu}
