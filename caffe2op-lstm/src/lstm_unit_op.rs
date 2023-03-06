crate::ix!();

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
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct LSTMUnitOp<Context> {
    storage:          OperatorStorage,
    context:          Context,
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
