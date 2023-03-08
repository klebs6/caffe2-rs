crate::ix!();

#[USE_RECURRENT_BASE_FUNCTIONS]
pub struct RecurrentGradientOp<T> {
    base:    RecurrentBaseOp<T>,
    phantom: PhantomData<T>,
}

register_cudnn_operator!{
    RecurrentGradient, 
    RecurrentGradientOp<f32>
}

num_inputs!{RecurrentGradient, 7}

num_outputs!{RecurrentGradient, 6}

allow_inplace!{RecurrentGradient, vec![(4, 5)]}

input_tags!{
    RecurrentGradientOp
    {
        Input,
        HiddenInput,
        CellInput,
        Weight,
        RnnScratch,
        Output,
        GradOutput,
        GradHiddenOutput,
        GradCellOutput
    }
}

output_tags!{
    RecurrentGradientOp
    {
        GradInput,
        GradHiddenInput,
        GradCellInput,
        GradWeight,
        DropoutStates,
        RnnScratchOut
    }
}

impl<T> RecurrentGradientOp<T> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : RecurrentBaseOp<T>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const int seqLength = Input(INPUT).dim32(0);
      if (Input(INPUT).sizes() != cachedInputDims_) {
        initialize(Input(INPUT), Output(DROPOUT_STATES));
        cachedInputDims_ = Input(INPUT).sizes().vec();
      }
      CUDNN_ENFORCE(cudnnGetRNNTrainingReserveSize(
          cudnn_wrapper_.inline_cudnn_handle(),
          rnnDesc_,
          seqLength,
          xDesc_->descs(),
          &reserveNbytes_));
      CAFFE_ENFORCE_EQ(reserveNbytes_, Input(RNN_SCRATCH).nbytes());
      Output(GRAD_INPUT)->ResizeLike(Input(INPUT));
      Output(GRAD_HIDDEN_INPUT)->ResizeLike(Input(HIDDEN_INPUT));
      Output(GRAD_CELL_INPUT)->ResizeLike(Input(CELL_INPUT));

      Output(GRAD_WEIGHT)->ResizeLike(Input(WEIGHT));
      math::Set<T, CUDAContext>(
          Output(GRAD_WEIGHT)->numel(),
          0.0,
          Output(GRAD_WEIGHT)->template mutable_data<T>(),
          &context_);

    #if CUDNN_VERSION_MIN(6,0,0)
      auto * reserve = Output(RNN_SCRATCH_OUT)->template mutable_data<T>();
    #else
      const auto * reserve = Output(RNN_SCRATCH_OUT)->template data<T>();
    #endif
      auto InputData = [this](int i) { return this->Input(i).template data<T>(); };
      auto OutputData = [this](int i) {
        return this->Output(i)->template mutable_data<T>();
      };

      cudnn_wrapper_.with_cudnn_state(0, [&](CudnnState* state) {
        CUDNN_ENFORCE(cudnnRNNBackwardData(
            state->cudnn_handle(),
            rnnDesc_,
            seqLength,
            yDesc_->descs(),
            InputData(OUTPUT), // Input(OUTPUT).template data<T>(),
            yDesc_->descs(),
            InputData(GRAD_OUTPUT), // Input(GRAD_OUTPUT).template data<T>(),
            hyDesc_,
            // Note: like CNTK, ignore these gradient inputs. t16675365 to
            // reconsider.
            nullptr,
            cyDesc_,
            nullptr,
            wDesc_,
            InputData(WEIGHT), // Input(WEIGHT).template data<T>(),
            hxDesc_,
            InputData(HIDDEN_INPUT), // Input(HIDDEN_INPUT).template data<T>(),
            cxDesc_,
            InputData(CELL_INPUT),
            xDesc_->descs(),
            OutputData(GRAD_INPUT),
            hxDesc_,
            OutputData(GRAD_HIDDEN_INPUT),
            cxDesc_,
            OutputData(GRAD_CELL_INPUT),
            state->workspace().get(cudnnWsNbytes_),
            cudnnWsNbytes_,
            reserve,
            reserveNbytes_));
        CUDNN_ENFORCE(cudnnRNNBackwardWeights(
            state->cudnn_handle(),
            rnnDesc_,
            seqLength,
            xDesc_->descs(),
            InputData(INPUT), // Input(INPUT).template data<T>(),
            hxDesc_,
            InputData(HIDDEN_INPUT), // Input(HIDDEN_INPUT).template data<T>(),
            yDesc_->descs(),
            InputData(OUTPUT), // Input(OUTPUT).template data<T>(),
            state->workspace().get(cudnnWsNbytes_),
            cudnnWsNbytes_,
            wDesc_,
            OutputData(
                GRAD_WEIGHT), // Output(GRAD_WEIGHT)->template mutable_data<T>(),
            reserve,
            reserveNbytes_));
      });

      return true;
        */
    }
}
