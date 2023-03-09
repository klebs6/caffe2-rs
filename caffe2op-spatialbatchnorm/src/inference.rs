crate::ix!();

#[inline] pub fn cost_inference_for_spatialBN(
    def: &OperatorDef, 
    input: &Vec<TensorShape>) -> OpSchemaCost 
{
    todo!();
    /*
        struct OpSchema::Cost cost = PointwiseCostInference<4>(def, in);
      ArgumentHelper helper(def);
      auto order =
          StringToStorageOrder(helper.GetSingleArgument<string>("order", "NCHW"));
      const TensorShape X = in[0];
      const int C =
          (order == StorageOrder::NCHW ? X.dims(1) : X.dims(X.dims_size() - 1));
      cost.params_bytes = 2 * C * sizeof(float);
      return cost;
    */
}
