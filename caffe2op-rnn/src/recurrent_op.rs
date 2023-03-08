crate::ix!();

/**
  | Recurrent wraps the Cudnn R5 RNN implementation.
  | See the Cudnn R5 documentation for more
  | information.
  | 
  | In general, the implementation takes
  | an input (TxNxD) tensor, the hidden
  | state input (NxD), the cell input (NxD),
  | and a weight tensor (effectively an
  | opaque blob, where the size and layout
  | is dictated by Cudnn).
  | 
  | The outputs are the output (again, TxNxD),
  | the final hidden/cell states (NxD).
  | These can be reset (at sequence boundaries
  | across minibatches) by multiplying
  | by zero.
  | 
  | The Cudnn arguments (hidden_size,
  | bidirectional, num_layers, rnn_mode,
  | input_mode) are passed directly through
  | to Cudnn.
  |
  */
#[USE_RECURRENT_BASE_FUNCTIONS]
pub struct RecurrentOp<T> {
    base:    RecurrentBaseOp<T>,
    phantom: PhantomData<T>,
}

register_cudnn_operator!{Recurrent, RecurrentOp<float>}

num_inputs!{Recurrent, 4}

num_outputs!{Recurrent, 5}

impl<T> RecurrentOp<T> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : RecurrentBaseOp<T>(std::forward<Args>(args)...)
        */
    }
}

input_tags!{
    RecurrentOp {
        Input,
        HiddenInput,
        CellInput,
        Weight
    }
}

output_tags!{
    RecurrentOp {
        Output,
        HiddenOutput,
        CellOutput,
        RnnScratch,
        DropoutStates
    }
}

#[derive(PartialEq,Eq)]
pub enum RecurrentParamOpMode { 
    SET_PARAM,
    GET_PARAM
}

impl<T> RecurrentOp<T> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const int seqLength = Input(INPUT).dim32(0);
      if (Input(INPUT).sizes() != cachedInputDims_) {
        initialize(
            Input(INPUT),
            Output(DROPOUT_STATES),
            Output(OUTPUT),
            Output(HIDDEN_OUTPUT),
            Output(CELL_OUTPUT));
        cachedInputDims_ = Input(INPUT).sizes().vec();
      }

      // Validation checks
      size_t weightsSize;
      CUDNN_ENFORCE(cudnnGetRNNParamsSize(
          cudnn_wrapper_.inline_cudnn_handle(),
          rnnDesc_,
          xDesc_->descs()[0],
          &weightsSize,
          cudnnTypeWrapper<T>::type));
      CAFFE_ENFORCE_EQ(Input(WEIGHT).nbytes(), weightsSize);

      // Training reserve size
      CUDNN_ENFORCE(cudnnGetRNNTrainingReserveSize(
          cudnn_wrapper_.inline_cudnn_handle(),
          rnnDesc_,
          seqLength,
          xDesc_->descs(),
          &reserveNbytes_));
      Output(RNN_SCRATCH)
          ->Resize(std::vector<int>{static_cast<int>(
              reserveNbytes_ / 4)}); // sizeof(T) - workaround clang bug
      Output(RNN_SCRATCH)->template mutable_data<T>();

      auto InputData = [this](int i) { return this->Input(i).template data<T>(); };
      auto OutputData = [this](int i) {
        return this->Output(i)->template mutable_data<T>();
      };

      if (OperatorStorage::GetSingleArgument<int>(OpSchema::Arg_IsTest, 0)) {
        cudnn_wrapper_.with_cudnn_state(0, [&](CudnnState* state) {
          CUDNN_ENFORCE(cudnnRNNForwardInference(
              state->cudnn_handle(),
              rnnDesc_,
              seqLength,
              xDesc_->descs(),
              InputData(INPUT), //.template data<T>(),
              hxDesc_,
              InputData(HIDDEN_INPUT), //.template data<T>(),
              cxDesc_,
              InputData(CELL_INPUT), //.template data<T>(),
              wDesc_,
              InputData(WEIGHT), //.template data<T>(),
              yDesc_->descs(),
              OutputData(OUTPUT), //->template mutable_data<T>(),
              hyDesc_,
              OutputData(HIDDEN_OUTPUT), //->template mutable_data<T>(),
              cyDesc_,
              OutputData(CELL_OUTPUT), //->template mutable_data<T>(),
              state->workspace().get(cudnnWsNbytes_),
              cudnnWsNbytes_));
        });
      } else {
        cudnn_wrapper_.with_cudnn_state(0, [&](CudnnState* state) {
          CUDNN_ENFORCE(cudnnRNNForwardTraining(
              state->cudnn_handle(),
              rnnDesc_,
              seqLength,
              xDesc_->descs(),
              InputData(INPUT), //.template data<T>(),
              hxDesc_,
              InputData(HIDDEN_INPUT), //.template data<T>(),
              cxDesc_,
              InputData(CELL_INPUT), //.template data<T>(),
              wDesc_,
              InputData(WEIGHT), //.template data<T>(),
              yDesc_->descs(),
              OutputData(OUTPUT), //->template mutable_data<T>(),
              hyDesc_,
              OutputData(HIDDEN_OUTPUT), //->template mutable_data<T>(),
              cyDesc_,
              OutputData(CELL_OUTPUT), //->template mutable_data<T>(),
              state->workspace().get(cudnnWsNbytes_),
              cudnnWsNbytes_,
              OutputData(RNN_SCRATCH), //->template mutable_data<T>(),
              reserveNbytes_));
        });
      }

      return true;
        */
    }
}
