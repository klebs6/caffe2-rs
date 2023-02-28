crate::ix!();

use crate::{
    OperatorStorage,
    Tensor,
    CPUContext
};

pub type TensorTuple = (Tensor, Tensor);

#[derive(Clone)]
pub struct CellParams {
    w_ih:    Tensor,
    w_hh:    Tensor,
    b_ih:    Option<Tensor>,
    b_hh:    Option<Tensor>,
    context: *mut CPUContext,
}

impl CellParams {
    
    pub fn new(
        _w_ih: &Tensor,
        _w_hh: &Tensor,
        _b_ih: &Tensor,
        _b_hh: &Tensor,
        _context: *mut CPUContext) -> Self 
    {
        todo!();
        /*
            initParams(_w_ih, _w_hh, _b_ih, _b_hh, _context);
        */
    }
    
    #[inline] pub fn init_params(
        &mut self, 
        _w_ih: &Tensor,
        _w_hh: &Tensor,
        _b_ih: &Tensor,
        _b_hh: &Tensor,
        _context: *mut CPUContext)
    {
        todo!();
        /*
            w_ih = copy_ctor(_w_ih);
        w_hh = copy_ctor(_w_hh);
        b_ih = copy_ctor(_b_ih);
        b_hh = copy_ctor(_b_hh);
        context = _context;
        */
    }
    
    #[inline] pub fn linear_ih(&self, input: &Tensor) -> Tensor {
        
        todo!();
        /*
            return linear(input, w_ih, b_ih, context);
        */
    }
    
    #[inline] pub fn linear_hh(&self, h: &Tensor) -> Tensor {
        
        todo!();
        /*
            return linear(h, w_hh, b_hh, context);
        */
    }
}

pub struct LSTMCell {
    context: *mut CPUContext,
}

impl LSTMCell {
    
    pub fn new(context: *mut CPUContext) -> Self {
        todo!();
        /*
            : context_(context)
        */
    }

    pub fn invoke(&self, input: &Tensor, hidden: &TensorTuple, params: &CellParams) -> TensorTuple {
        todo!();
        /*
        const auto& hx = std::get<0>(hidden);
        const auto& cx = std::get<1>(hidden);
        auto linear_ih = params.linear_ih(input);
        auto linear_hh = params.linear_hh(hx);
        auto gates = add(linear_ih, linear_hh, context_);
        auto chunked_gates = chunk(gates, 4, 1, context_);
        auto ingate = sigmoid(chunked_gates[0]);
        auto forgetgate = sigmoid(chunked_gates[1]);
        auto cellgate = tanh(chunked_gates[2], context_);
        auto outgate = sigmoid(chunked_gates[3]);

        auto cy =
            add(mul(forgetgate, cx, context_),
                mul(ingate, cellgate, context_),
                context_);
        auto hy = mul(outgate, tanh(cy, context_), context_);
        return std::make_tuple(std::move(hy), std::move(cy));
        */
    }
}

pub struct LayerOutput<OutputType, HiddenType> {
    outputs:      OutputType,
    final_hidden: HiddenType,
}

impl<OutputType,HiddenType> LayerOutput<OutputType, HiddenType> {
    
    pub fn new(_outputs: &OutputType, _hidden: &HiddenType) -> Self {
        todo!();
        /*
            outputs = copy_ctor(_outputs);
        final_hidden = copy_ctor(_hidden);
        */
    }
}

pub trait Layer<HiddenType, ParamType> {
    type OutputType = LayerOutput::<Tensor, HiddenType>;
    fn invoke(&self, input: &Tensor, input_hidden: &HiddenType, params: &ParamType) -> Self::OutputType;
}

pub struct FullLSTMLayer {
    cell:    LSTMCell,
    context: *mut CPUContext,
}

impl Layer<TensorTuple, CellParams> for FullLSTMLayer {

    fn invoke(&self,
        inputs: &Tensor,
        input_hidden: &(Tensor, Tensor),
        params: &CellParams) 
        -> LayerOutput<Tensor, TensorTuple> 
    {
        todo!();
        /*
        auto unstacked_output =
            (*this)(unbind(inputs, 0, context_), input_hidden, params);
        return {stack(unstacked_output.outputs, 0, context_),
                unstacked_output.final_hidden};
        */
    }
}

impl FullLSTMLayer {
    
    pub fn new(cell: &mut LSTMCell, context: *mut CPUContext) -> Self {
        todo!();
        /*
            : cell_(cell), context_(context)
        */
    }

    pub fn invoke_vec(&self, 
        step_inputs:  &Vec<Tensor>, 
        input_hidden: (Tensor, Tensor), 
        params:       &CellParams) 
        -> LayerOutput<Vec<Tensor>, TensorTuple> 
    {
        todo!();
        /*
        std::vector<Tensor> step_outputs;
        auto hidden = copy_ctor(input_hidden);

        for (size_t i = 0; i < step_inputs.size(); i++) {
          hidden = cell_(step_inputs[i], hidden, params);
          step_outputs.push_back(copy_ctor(std::get<0>(hidden)));
        }

        return {step_outputs, hidden};
        */
    }
}

pub trait FullBidirectionalLSTMLayerTypes {
    type BidirHiddenType = (TensorTuple, TensorTuple);
    type ParamType       = (CellParams, CellParams);
    type OutputType      = LayerOutput<Tensor, Self::BidirHiddenType>;
}

pub struct FullBidirectionalLSTMLayer {
    layer:   FullLSTMLayer,
    context: *mut CPUContext,
}

impl Layer<(TensorTuple, TensorTuple), (CellParams, CellParams)> for FullBidirectionalLSTMLayer {

    #[inline] fn invoke(
        &self, 
        input: &Tensor,
        input_hidden: &<Self as FullBidirectionalLSTMLayerTypes>::BidirHiddenType,
        params: &<Self as FullBidirectionalLSTMLayerTypes>::ParamType) 
        -> <Self as FullBidirectionalLSTMLayerTypes>::OutputType 
    {

        todo!();
        /*
            std::vector<Tensor> outputs;
        auto step_inputs = unbind(input, 0, context_);
        auto fw_result = layer_(step_inputs, input_hidden.first, params.first);
        auto fw_output = stack(fw_result.outputs, 0, context_);
        outputs.push_back(copy_ctor(fw_output));
        auto rev_step_inputs = reverse(std::move(step_inputs));
        auto rev_result =
            layer_(rev_step_inputs, input_hidden.second, params.second);
        std::reverse(rev_result.outputs.begin(), rev_result.outputs.end());
        auto rev_output = stack(rev_result.outputs, 0, context_);
        outputs.push_back(copy_ctor(rev_output));
        return {cat(outputs, fw_output.dim() - 1, context_),
                std::make_pair(
                    std::move(fw_result.final_hidden),
                    std::move(rev_result.final_hidden))};
        */
    }
}

impl FullBidirectionalLSTMLayerTypes for FullBidirectionalLSTMLayer { }

impl FullBidirectionalLSTMLayer {
    
    pub fn new(cell: &mut LSTMCell, context: *mut CPUContext) -> Self {
        todo!();
        /*
            : layer_(cell, context), context_(context)
        */
    }
    
    #[inline] pub fn reverse(&self, x: Vec<Tensor>) -> Vec<Tensor> {
        
        todo!();
        /*
            std::reverse(x.begin(), x.end());
        return std::move(x);
        */
    }
}

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

/**
  | Parses a flat list of parameter tensors
  | into a list of CellParams
  |
  */
#[inline] pub fn gather_params(
    params:     &Vec<Tensor>,
    has_biases: bool,
    context:    *mut CPUContext) -> Vec<CellParams> 
{
    todo!();
    /*
        Tensor undefined;
      std::vector<CellParams> result;
      if (has_biases) {
        CAFFE_ENFORCE_EQ(
            params.size() % 4, 0, "got an incorrect number of LSTM parameters");
        for (size_t i = 0; i < params.size(); i += 4) {
          result.emplace_back(
              params[i], params[i + 1], params[i + 2], params[i + 3], context);
        }
      } else {
        CAFFE_ENFORCE_EQ(
            params.size() % 2, 0, "got an incorrect number of LSTM parameters");
        for (size_t i = 0; i < params.size(); i += 2) {
          result.emplace_back(
              params[i], params[i + 1], undefined, undefined, context);
        }
      }
      return result;
    */
}

pub struct InferenceLSTMOp<Context> {
    storage: OperatorStorage,
    context: Context,
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
