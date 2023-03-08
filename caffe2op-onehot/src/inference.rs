crate::ix!();

#[inline] pub fn tensor_inference_for_batch_one_hot(
    def:   &OperatorDef, 
    input: &Vec<TensorShape>) -> Vec<TensorShape> 
{
    todo!();
    /*
        std::vector<int64_t> output_dims(2);
      output_dims[0] = in[0].dims(0); // N
      output_dims[1] = in[2].dims(0); // vals.size()
      return vector<TensorShape>{
          CreateTensorShape(vector<int64_t>{output_dims}, in[0].data_type())};
    */
}

#[inline] pub fn tensor_inference_for_bucket_batch_one_hot(
    def:   &OperatorDef, 
    input: &Vec<TensorShape>) -> Vec<TensorShape> {
    
    todo!();
    /*
        std::vector<int64_t> output_dims(2);
      output_dims[0] = in[0].dims(0); // N
      output_dims[1] = in[1].dims(0) + in[2].dims(0); // vals.size() + length.size()
      return vector<TensorShape>{
          CreateTensorShape(vector<int64_t>{output_dims}, in[0].data_type())};
    */
}

#[inline] pub fn cost_inference_for_batch_one_hot(
    def:   &OperatorDef, 
    input: &Vec<TensorShape>) -> OpSchemaCost {
    
    todo!();
    /*
        CAFFE_ENFORCE_EQ(in.size(), 3, "BatchOneHot requires three inputs");
      struct OpSchema::Cost c;
      const TensorShape output = TensorInferenceForBatchOneHot(def, in)[0];

      const auto& data = in[0];
      const auto& length = in[1];
      const auto& values = in[2];

      uint64_t nBytesData = nElemFromDim(data) * sizeof(data.data_type());
      uint64_t nBytesLength = nElemFromDim(length) * sizeof(length.data_type());
      uint64_t nBytesValues = nElemFromDim(values) * sizeof(values.data_type());
      c.flops = 0;
      c.bytes_read = nBytesData + nBytesLength + nBytesValues;
      c.bytes_written = nElemFromDim(output) * sizeof(output.data_type());
      c.params_bytes = 0;
      return c;
    */
}
