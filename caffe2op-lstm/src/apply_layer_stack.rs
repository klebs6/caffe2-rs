crate::ix!();

#[inline] pub fn apply_layer_stack<HiddenType, WeightType>(
    layer:      &dyn Layer<HiddenType,WeightType, OutputType = LayerOutput::<Tensor, HiddenType>>,
    input:      &Tensor,
    hiddens:    &Vec<HiddenType>,
    weights:    &Vec<WeightType>,
    num_layers: i64) -> LayerOutput<Tensor,Vec<HiddenType>> 
{
    todo!();
    /*
        CAFFE_ENFORCE(
          num_layers == hiddens.size(),
          "Expected more hidden states in stacked_rnn");
      CAFFE_ENFORCE(
          num_layers == weights.size(), "Expected more weights in stacked_rnn");

      auto layer_input = input.UnsafeSharedInstance();
      auto hidden_it = hiddens.begin();
      auto weight_it = weights.begin();
      std::vector<HiddenType> final_hiddens(num_layers);
      for (int64_t l = 0; l < num_layers; ++l) {
        auto layer_output = layer(layer_input, *(hidden_it++), *(weight_it++));
        final_hiddens.at(l) = std::move(layer_output.final_hidden);
        layer_input = std::move(layer_output.outputs);
      }
      return {layer_input, final_hiddens};
    */
}
