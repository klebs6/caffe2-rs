crate::ix!();

/**
  | Super special-case operator. Used
  | to pad a tensor to mimic pytorch's pad_packed_sequence.
  | 
  | Given an input tensor INPUT of size NxBxM
  | and an input tensor LENS of size B, where
  | 
  | N = maximum sequence length
  | 
  | B = batch size
  | 
  | M = hidden size
  | 
  | set each element of INPUT to zero if it
  | is is past the end of the corresponding
  | sequence (i.e. if LENS[j] > i for an index
  | (i,j,k)).
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct VariableLengthSequencePaddingOp<T, Context> {
    storage: OperatorStorage,
    context: Context,
    phantom: PhantomData<T>,
}

register_cpu_operator!{
    VariableLengthSequencePadding, 
    VariableLengthSequencePaddingOp<f32, CPUContext>
}

num_inputs!{VariableLengthSequencePadding, 2}

num_outputs!{VariableLengthSequencePadding, 1}

allow_inplace!{VariableLengthSequencePadding, vec![(0, 0)]}

input_tags!{
    VariableLengthSequencePaddingOp {
        Input,
        SeqLengths
    }
}

output_tags!{
    VariableLengthSequencePaddingOp {
        Output
    }
}

impl<T,Context> VariableLengthSequencePaddingOp<T,Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto N = Input(INPUT).size(0);
        const auto B = Input(INPUT).size(1);
        const auto M = Input(INPUT).size(2);

        auto X = Output(OUTPUT)->template mutable_data<T>();

        auto seqLengths = Input(SEQ_LENGTHS).template data<int32_t>();

        detail::VariableLengthSequencePadding<T, Context>(
            N, B, M, X, seqLengths, 0, &context_);
        return true;
        */
    }
}

#[inline] pub fn variable_length_sequence_padding<T, Context>(
    n:            i32,
    b:            i32,
    m:            i32,
    x:            *mut T,
    seq_lengths:  *const i32,
    pad_value:    T,
    context:      *mut Context) 
{
    todo!();
    /*
        for (int j = 0; j < B; j++) {
        for (int i = seqLengths[j]; i < N; i++) {
          EigenVectorArrayMap<T>(X + B * M * i + M * j, M).setConstant(padValue);
        }
      }
    */
}
