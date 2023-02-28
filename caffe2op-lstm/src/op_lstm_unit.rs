crate::ix!();

use crate::{
    GradientMakerBase,
    OperatorStorage,
    Workspace,
    OperatorDef,
};

#[inline] pub fn lSTMUnit<T, Context>(
    n:           i32,
    d:           i32,
    t:           i32,
    h_prev:      *const T,
    c_prev:      *const T,
    x:           *const T,
    seq_lengths: *const i32,
    drop_states: bool,
    c:           *mut T,
    h:           *mut T,
    forget_bias: f32,
    context:     *mut Context)
{
    todo!();
    /*
        LstmUnitCpu<T>(
          N, D, t, H_prev, C_prev, X, seqLengths, drop_states, C, H, forget_bias);
    */
}

#[inline] pub fn lstm_unit_gradient<T, Context>(
    n:             i32,
    d:             i32,
    t:             i32,
    c_prev:        *const T,
    x:             *const T,
    seq_lengths:   *const i32,
    c:             *const T,
    h:             *const T,
    c_diff:        *const T,
    h_diff:        *const T,
    drop_states:   bool,
    h_prev_diff:   *mut T,
    c_prev_diff:   *mut T,
    x_diff:        *mut T,
    forget_bias:   f32,
    context:       *mut Context) 
{
    todo!();
    /*
        LstmUnitGradientCpu<T>(
          N,
          D,
          t,
          C_prev,
          X,
          seqLengths,
          C,
          H,
          C_diff,
          H_diff,
          drop_states,
          H_prev_diff,
          C_prev_diff,
          X_diff,
          forget_bias);
    */
}

/**
  | LSTMUnit computes the activations
  | of a standard LSTM (without peephole
  | connections), in a sequence-length
  | aware fashion.
  | 
  | Concretely, given the (fused) inputs
  | X (TxNxD), the previous cell state (NxD),
  | and the sequence lengths (N), computes
  | the LSTM activations, avoiding computation
  | if the input is invalid (as in, the value
  | at X{t][n] >= seqLengths[n].
  |
  */
pub struct LSTMUnitOp<Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,

    forget_bias:      f32,
    sequence_lengths: bool,
    drop_states:      bool,
}

register_cpu_operator!{LSTMUnit, LSTMUnitOp<CPUContext>}

num_inputs!{LSTMUnit, (4,5)}

num_outputs!{LSTMUnit, 2}

args!{LSTMUnit, 
    0 => ("forget_bias", "Bias term to add in while calculating forget gate"),
    1 => ("sequence_lengths", "When false, the sequence lengths input is left out, 
        and all following inputs are shifted left by one.")
}

/**
  | additional input tags are determined
  | dynamically based on whether sequence_lengths
  | is present.
  |
  */
input_tags!{
    LSTMUnitOp {
        HiddenTM1,
        CellTM1,
        Gates,
        SeqLengths
    }
}

output_tags!{
    LSTMUnitOp {
        HiddenT,
        CellT
    }
}

impl<Context> LSTMUnitOp<Context> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator<Context>(operator_def, ws),
            forget_bias_(static_cast<float>(
                this->template GetSingleArgument<float>("forget_bias", 0.0))),
            sequence_lengths_(
                this->template GetSingleArgument<bool>("sequence_lengths", true)),
            drop_states_(
                this->template GetSingleArgument<bool>("drop_states", false))
        */
    }

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            // handle potentially-missing sequence lengths input
            const size_t TIMESTEP = SEQ_LENGTHS + (sequence_lengths_ ? 1 : 0);

            // Extract N
            const auto N = Input(CELL_T_M_1).size(1);

            // Gates: 1xNxG
            const auto G = Input(GATES).size(2);
            const auto D = Input(CELL_T_M_1).size(2);

            CAFFE_ENFORCE_EQ(4 * D, G);
            const auto* H_prev = Input(HIDDEN_T_M_1).template data<T>();
            const auto* C_prev = Input(CELL_T_M_1).template data<T>();
            const auto* X = Input(GATES).template data<T>();

            const int32_t* seqLengths = nullptr;
            if (sequence_lengths_) {
              CAFFE_ENFORCE_EQ(Input(SEQ_LENGTHS).numel(), N);
              seqLengths = Input(SEQ_LENGTHS).template data<int32_t>();
            }

            const auto t = static_cast<OperatorStorage*>(this)
                               ->Input<Tensor>(TIMESTEP, CPU)
                               .template data<int32_t>()[0];
            Output(CELL_T)->ResizeLike(Input(CELL_T_M_1));
            auto* C = Output(CELL_T)->template mutable_data<T>();
            Output(HIDDEN_T)->ResizeLike(Input(CELL_T_M_1));
            auto* H = Output(HIDDEN_T)->template mutable_data<T>();
            detail::LSTMUnit<T, Context>(
                N,
                D,
                t,
                H_prev,
                C_prev,
                X,
                seqLengths,
                drop_states_,
                C,
                H,
                forget_bias_,
                &context_);
            return true;
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DoRunWithType<float>();
        */
    }
}

///-------------------------------------
pub struct LSTMUnitGradientOp<Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,

    forget_bias:      f32,
    sequence_lengths: bool,
    drop_states:      bool,
}

register_cpu_operator!{LSTMUnitGradient, LSTMUnitGradientOp<CPUContext>}

num_inputs!{LSTMUnitGradient, (8,9)}

num_outputs!{LSTMUnitGradient, 3}

args!{LSTMUnitGradient, 
    0 => ("sequence_lengths", "When false, the sequence lengths input is left out, 
        and all following inputs are shifted left by one.")
}

/**
  | additional input tags are determined
  | dynamically based on whether sequence_lengths
  | is present.
  |
  */
input_tags!{
    LSTMUnitGradientOp {
        HiddenTM1,
        CellTM1,
        Gates,
        SeqLengths
    }
}

output_tags!{
    LSTMUnitGradientOp {
        HiddenTM1Grad,
        CellTM1Grad,
        GatesGrad
    }
}

impl<Context> LSTMUnitGradientOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            forget_bias_(static_cast<float>(
                this->template GetSingleArgument<float>("forget_bias", 0.0))),
            sequence_lengths_(
                this->template GetSingleArgument<bool>("sequence_lengths", true)),
            drop_states_(
                this->template GetSingleArgument<bool>("drop_states", false))
        */
    }

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            // handle potentially-missing sequence lengths input
            const size_t inputOffset = SEQ_LENGTHS + (sequence_lengths_ ? 1 : 0);
            const size_t TIMESTEP = inputOffset;
            const size_t HIDDEN_T = inputOffset + 1;
            const size_t CELL_T = inputOffset + 2;
            const size_t HIDDEN_T_GRAD = inputOffset + 3;
            const size_t CELL_T_GRAD = inputOffset + 4;

            // Extract N
            const auto N = Input(CELL_T_M_1).size(1);

            // Gates: 1xNxG
            const auto G = Input(GATES).size(2);
            const auto D = Input(CELL_T_M_1).size(2);

            CAFFE_ENFORCE_EQ(4 * D, G);
            const auto* C_prev = Input(CELL_T_M_1).template data<T>();
            const auto* X = Input(GATES).template data<T>();
            const auto t = static_cast<OperatorStorage*>(this)
                               ->Input<Tensor>(TIMESTEP, CPU)
                               .template data<int32_t>()[0];
            const auto* C = Input(CELL_T).template data<T>();
            const auto* H = Input(HIDDEN_T).template data<T>();
            const auto* C_diff = Input(CELL_T_GRAD).template data<T>();
            const auto* H_diff = Input(HIDDEN_T_GRAD).template data<T>();

            const int32_t* seqLengths = nullptr;
            if (sequence_lengths_) {
              CAFFE_ENFORCE_EQ(Input(SEQ_LENGTHS).numel(), N);
              seqLengths = Input(SEQ_LENGTHS).template data<int32_t>();
            }

            Output(HIDDEN_T_M_1_GRAD)->ResizeLike(Input(HIDDEN_T_M_1));
            auto* H_prev_diff = Output(HIDDEN_T_M_1_GRAD)->template mutable_data<T>();
            Output(CELL_T_M_1_GRAD)->ResizeLike(Input(CELL_T_M_1));
            auto* C_prev_diff = Output(CELL_T_M_1_GRAD)->template mutable_data<T>();
            Output(GATES_GRAD)->ResizeLike(Input(GATES));
            auto* X_diff = Output(GATES_GRAD)->template mutable_data<T>();

            detail::LSTMUnitGradient<T, Context>(
                N,
                D,
                t,
                C_prev,
                X,
                seqLengths,
                C,
                H,
                C_diff,
                H_diff,
                drop_states_,
                H_prev_diff,
                C_prev_diff,
                X_diff,
                forget_bias_,
                &context_);
            return true;
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DoRunWithType<float>();
        */
    }
}

pub struct GetLSTMUnitGradient;

impl GetGradientDefs for GetLSTMUnitGradient {
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            if (GetFlagArgument(def_, "sequence_lengths", true)) {
          return SingleGradientDef(
              "LSTMUnitGradient",
              "",
              vector<string>{
                  I(0), I(1), I(2), I(3), I(4), O(0), O(1), GO(0), GO(1)},
              vector<string>{GI(0), GI(1), GI(2)});
        } else {
          return SingleGradientDef(
              "LSTMUnitGradient",
              "",
              vector<string>{I(0), I(1), I(2), I(3), O(0), O(1), GO(0), GO(1)},
              vector<string>{GI(0), GI(1), GI(2)});
        }
        */
    }
}

register_gradient!{LSTMUnit, GetLSTMUnitGradient}
