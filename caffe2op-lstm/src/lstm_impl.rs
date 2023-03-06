crate::ix!();

#[inline] pub fn lstm_impl(
    input:          &Tensor,
    params:         &Vec<CellParams>,
    hx:             &Tensor,
    cx:             &Tensor,
    num_layers:     i64,
    bidirectional:  bool,
    context:        *mut CPUContext) -> (Tensor,Tensor,Tensor) 
{
    todo!();
    /*
        using stack_output = LayerOutput<Tensor, std::vector<TensorTuple>>;
      auto layer_hx = unbind(hx, 0, context);
      auto layer_cx = unbind(cx, 0, context);
      int64_t total_layers = layer_hx.size();
      std::vector<std::tuple<Tensor, Tensor>> hiddens;
      hiddens.reserve(total_layers);
      for (int64_t i = 0; i < total_layers; ++i) {
        hiddens.emplace_back(std::move(layer_hx[i]), std::move(layer_cx[i]));
      }
      LSTMCell cell(context);
      std::shared_ptr<stack_output> stack_output_ptr;
      if (bidirectional) {
        auto bidir_result = apply_layer_stack(
            FullBidirectionalLSTMLayer{cell, context},
            input,
            pair_vec(hiddens),
            pair_vec(params),
            num_layers);
        stack_output_ptr.reset(new stack_output(
            bidir_result.outputs,
            unpair_vec(std::move(bidir_result.final_hidden))));
      } else {
        auto result = apply_layer_stack(
            FullLSTMLayer{cell, context}, input, hiddens, params, num_layers);
        stack_output_ptr = std::make_shared<stack_output>(std::move(result));
      }

      std::vector<Tensor> hy, cy;
      hy.reserve(total_layers);
      cy.reserve(total_layers);
      for (auto& hidden : stack_output_ptr->final_hidden) {
        hy.push_back(std::move(std::get<0>(hidden)));
        cy.push_back(std::move(std::get<1>(hidden)));
      }
      return std::make_tuple(
          std::move(stack_output_ptr->outputs),
          stack(hy, 0, context),
          stack(cy, 0, context));
    */
}
