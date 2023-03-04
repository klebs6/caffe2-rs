crate::ix!();

/**
  | GRUUnit computes the activations of
  | a standard GRU, in a sequence-length
  | aware fashion.
  | 
  | Concretely, given the (fused) inputs
  | X (TxNxD), the previous hidden state
  | (NxD), and the sequence lengths (N),
  | computes the GRU activations, avoiding
  | computation if the input is invalid
  | (as in, the value at X[t][n] >= seqLengths[n].
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct GRUUnitOp<T, Context> {

    storage:          OperatorStorage,
    context:          Context,

    drop_states:      bool,
    sequence_lengths: bool,

    phantom:          PhantomData<T>,
}

num_inputs!{GRUUnit, (3,4)}

num_outputs!{GRUUnit, 1}

outputs!{GRUUnit, 
    0 => ("hidden", "The new GRU hidden state calculated by this op.")
}

args!{GRUUnit, 
    0 => ("drop_states", "Bool to determine if hidden state is zeroes or passed along for timesteps past the given sequence_length."),
    1 => ("sequence_lengths", "When false, the sequence lengths input is left out, and all following inputs are shifted left by one.")
}

/**
  | additional input tags are determined
  | dynamically based on whether sequence_lengths
  | is present.
  |
  */
input_tags!{
    GRUUnitOp {
        HiddenTM1,
        Gates,
        SeqLengths
    }
}

output_tags!{
    GRUUnitOp {
        HiddenT
    }
}

impl<T, Context> GRUUnitOp<T, Context> {
    
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
        const size_t TIMESTEP = SEQ_LENGTHS + (sequence_lengths_ ? 1 : 0);

        // Extract N
        const auto N = Input(HIDDEN_T_M_1).size(1);

        // Gates: 1xNxG
        const auto G = Input(GATES).size(2);
        const auto D = Input(HIDDEN_T_M_1).size(2);

        CAFFE_ENFORCE_EQ(3 * D, G);
        const auto* H_prev = Input(HIDDEN_T_M_1).template data<T>();
        const auto* X = Input(GATES).template data<T>();

        const int32_t* seqLengths = nullptr;
        if (sequence_lengths_) {
          CAFFE_ENFORCE_EQ(Input(SEQ_LENGTHS).numel(), N);
          seqLengths = Input(SEQ_LENGTHS).template data<int32_t>();
        }

        const auto t = static_cast<OperatorStorage*>(this)
                           ->Input<Tensor>(TIMESTEP, CPU)
                           .template data<int32_t>()[0];
        Output(HIDDEN_T)->ResizeLike(Input(HIDDEN_T_M_1));
        auto* H = Output(HIDDEN_T)->template mutable_data<T>();

        detail::GRUUnit<T, Context>(
            N, D, t, H_prev, X, seqLengths, drop_states_, H, &context_);
        return true;
        */
    }
}

#[inline] fn gru_unit<T, Context>(
    n:            i32,
    d:            i32,
    t:            i32,
    h_prev:       *const T,
    x:            *const T,
    seq_lengths:  *const i32,
    drop_states:  bool,
    h:            *mut T,
    context:      *mut Context) 
{
    todo!();
    /*
        for (int n = 0; n < N; ++n) {
        const bool valid = seqLengths == nullptr || t < seqLengths[n];

        for (int d = 0; d < D; ++d) {
          if (!valid) {
            if (drop_states) {
              H[d] = 0;
            } else {
              H[d] = H_prev[d];
            }
          } else {
            const T update = X[1 * D + d];
            const T output = X[2 * D + d];
            T sigmoid_update = sigmoid(update);
            H[d] = H_prev[d] * sigmoid_update +
                host_tanh(output) * (1.0f - sigmoid_update);
          }
        }

        H_prev += D;
        X += 3 * D;
        H += D;
      }
    */
}
