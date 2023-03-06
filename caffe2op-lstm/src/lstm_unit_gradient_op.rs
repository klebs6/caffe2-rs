crate::ix!();

///-------------------------------------
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct LSTMUnitGradientOp<Context> {
    storage:          OperatorStorage,
    context:          Context,
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
