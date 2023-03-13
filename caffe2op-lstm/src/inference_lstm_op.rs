crate::ix!();

pub struct InferenceLSTMOp<Context> {
    storage:       OperatorStorage,
    context:       Context,
    num_layers:    i64,
    bidirectional: bool,
    has_biases:    bool,
    batch_first:   bool,
}

impl<Context> InferenceLSTMOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator(std::forward<Args>(args)...),
            num_layers_(this->template GetSingleArgument<int64_t>("num_layers", 1)),
            bidirectional_(
                this->template GetSingleArgument<bool>("bidirectional", false)),
            has_biases_(this->template GetSingleArgument<bool>("has_biases", true)),
            batch_first_(
                this->template GetSingleArgument<bool>("batch_first", false))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& _input = Input(0);
      auto& hidden_0 = Input(1);
      auto& hidden_1 = Input(2);
      std::vector<Tensor> params;
      for (int i = 3; i < InputSize(); i++) {
        params.push_back(Input(i).UnsafeSharedInstance());
      }
      auto input = batch_first_ ? transpose(_input, 0, 1, &context_)
                                : _input.UnsafeSharedInstance();

      auto cell_params = gather_params(params, has_biases_, &context_);
      auto results = _lstm_impl(
          input,
          cell_params,
          hidden_0,
          hidden_1,
          num_layers_,
          bidirectional_,
          &context_);

      auto output = copy_ctor(std::get<0>(results));
      if (batch_first_) {
        output = transpose(output, 0, 1, &context_);
      }
      SetOutputTensor(0, copy_ctor(output));
      SetOutputTensor(1, copy_ctor(std::get<1>(results)));
      SetOutputTensor(2, copy_ctor(std::get<2>(results)));
      return true;
        */
    }
}

register_cpu_operator!{InferenceLSTM, InferenceLSTMOp}

num_inputs!{InferenceLSTM, (1,INT_MAX)}

num_outputs!{InferenceLSTM, 3}

outputs!{InferenceLSTM, 
    0 => ("output", "the output of the last layer of lstm"),
    1 => ("hidden", "hidden state at t = seq_len"),
    2 => ("cell",   "cell state at t = seq_len")
}

args!{InferenceLSTM, 
    0 => ("num_layers",     "(*long*): number of layers in the lstm stack"),
    1 => ("has_biases",     "(*bool*): whether the cells have biases or not"),
    2 => ("batch_first",    "(*bool*): whether the batch is at dim 0"),
    3 => ("bidirectional",  "(*bool*): if bidirectional")
}

no_gradient!{InferenceLSTM}
