crate::ix!();

///----------------------------------------
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct GRUUnitGradientOp<T, Context> {

    storage:          OperatorStorage,
    context:          Context,

    drop_states:      bool,
    sequence_lengths: bool,

    phantom:          PhantomData<T>,
}

num_inputs!{GRUUnitGradient, (5,6)}

num_outputs!{GRUUnitGradient, 2}

args!{GRUUnitGradient, 
    0 => ("sequence_lengths", "When false, the sequence lengths input is left out, and all following inputs are shifted left by one.")
}

impl<T,Context> GRUUnitGradientOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            drop_states_(
                this->template GetSingleArgument<bool>("drop_states", false)),
            sequence_lengths_(
                this->template GetSingleArgument<bool>("sequence_lengths", true))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            // handle potentially-missing sequence lengths input
        const size_t inputOffset = SEQ_LENGTHS + (sequence_lengths_ ? 1 : 0);
        const size_t TIMESTEP = inputOffset;
        const size_t HIDDEN_T = inputOffset + 1;
        const size_t HIDDEN_T_GRAD = inputOffset + 2;

        // Extract N
        const auto N = Input(HIDDEN_T_M_1).size(1);

        // Gates: 1xNxG
        const auto G = Input(GATES).size(2);
        const auto D = Input(HIDDEN_T_M_1).size(2);

        CAFFE_ENFORCE_EQ(3 * D, G);
        const auto* H_prev = Input(HIDDEN_T_M_1).template data<T>();
        const auto* X = Input(GATES).template data<T>();
        const auto t = static_cast<OperatorStorage*>(this)
                           ->Input<Tensor>(TIMESTEP, CPU)
                           .template data<int32_t>()[0];
        const auto* H = Input(HIDDEN_T).template data<T>();
        const auto* H_diff = Input(HIDDEN_T_GRAD).template data<T>();

        const int32_t* seqLengths = nullptr;
        if (sequence_lengths_) {
          CAFFE_ENFORCE_EQ(Input(SEQ_LENGTHS).numel(), N);
          seqLengths = Input(SEQ_LENGTHS).template data<int32_t>();
        }

        Output(HIDDEN_T_M_1_GRAD)->ResizeLike(Input(HIDDEN_T_M_1));
        auto* H_prev_diff = Output(HIDDEN_T_M_1_GRAD)->template mutable_data<T>();
        Output(GATES_GRAD)->ResizeLike(Input(GATES));
        auto* X_diff = Output(GATES_GRAD)->template mutable_data<T>();

        detail::GRUUnitGradient<T, Context>(
            N,
            D,
            t,
            H_prev,
            X,
            seqLengths,
            H,
            H_diff,
            drop_states_,
            H_prev_diff,
            X_diff,
            &context_);
        return true;
        */
    }
}

input_tags!{
    GRUUnitGradientOp {
        HiddenTM1,
        Gates,
        SeqLengths
    }
}

output_tags!{
    GRUUnitGradientOp {
        HiddenTM1Grad,
        GatesGrad
    }
}

register_cpu_operator!{GRUUnit, GRUUnitOp<f32,CPUContext>}

register_cpu_operator!{GRUUnitGradient, GRUUnitGradientOp<f32, CPUContext> }

#[inline] fn gru_unit_gradient<T, Context>(
    n:           i32,
    d:           i32,
    t:           i32,
    h_prev:      *const T,
    x:           *const T,
    seq_lengths: *const i32,
    h:           *const T,
    h_diff:      *const T,
    drop_states: bool,
    h_prev_diff: *mut T,
    x_diff:      *mut T,
    context:     *mut Context) 
{
    todo!();
    /*
        for (int n = 0; n < N; ++n) {
        const bool valid = seqLengths == nullptr || t < seqLengths[n];

        for (int d = 0; d < D; ++d) {
          T* h_prev_diff = H_prev_diff + d;
          T* reset_diff = X_diff + 0 * D + d;
          T* update_diff = X_diff + 1 * D + d;
          T* output_diff = X_diff + 2 * D + d;

          if (!valid) {
            if (drop_states) {
              *h_prev_diff = 0;
            } else {
              *h_prev_diff = H_diff[d];
            }
            *reset_diff = 0;
            *update_diff = 0;
            *output_diff = 0;
          } else {
            // Calculate Gate Outputs
            const T u = sigmoid(X[1 * D + d]);
            const T o = host_tanh(X[2 * D + d]);

            *h_prev_diff = H_diff[d] * u;
            *reset_diff = 0; // 0 contribution to gradient from this operation
            *update_diff = (H_diff[d] * H_prev[d] - H_diff[d] * o) * u * (1.0f - u);
            *output_diff = H_diff[d] * (1.0f - u) * (1.0f - o * o);
          }
        }

        H_prev += D;
        X += 3 * D;
        H += D;
        H_diff += D;
        X_diff += 3 * D;
        H_prev_diff += D;
      }
    */
}
