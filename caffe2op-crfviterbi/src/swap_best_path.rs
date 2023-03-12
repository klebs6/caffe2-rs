crate::ix!();

/**
  | Given a sequence of indices and a matrix,
  | enforce that these indices have the
  | best columnwise scores score
  |
  */
pub struct SwapBestPathOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

register_cpu_operator!{SwapBestPath, SwapBestPathOp}

num_inputs!{SwapBestPath, 2}

num_outputs!{SwapBestPath, 1}

inputs!{SwapBestPath, 
    0 => ("predictions", "N*D predictions matrix"),
    1 => ("bestPath", "N*1 vector holds the best path indices ")
}

outputs!{SwapBestPath, 
    0 => ("new_predictions", "N*D updated predictions matrix")
}

no_gradient!{SwapBestPath}

impl<Context> SwapBestPathOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& data = Input(0);
        auto& newBestIdicies = Input(1);

        CAFFE_ENFORCE(
            data.dim() == 2 && newBestIdicies.dim() == 1,
            "predictions should be a 2D matrix and  bestPath should be 1D vector");

        CAFFE_ENFORCE(
            data.size(0) == newBestIdicies.size(0),
            "predictions and bestPath dimensions not matching");

        auto* updatedData = Output(0, data.sizes(), at::dtype<float>());
        float* outData = updatedData->template mutable_data<float>();
        context_.CopyItemsSameDevice(
            data.dtype(), data.numel(), data.template data<float>(), outData);

        Tensor bestScores(CPU);
        bestScores.ResizeLike(newBestIdicies);
        Tensor oldBestIndices(CPU);
        oldBestIndices.ResizeLike(newBestIdicies);

        ColwiseMaxAndArg(
            data.template data<float>(),
            data.size(0),
            data.size(1),
            bestScores.template mutable_data<float>(),
            oldBestIndices.template mutable_data<int32_t>());

        auto block_size = data.numel() / data.size(0);

        const int32_t* oldBestIdx = oldBestIndices.template data<int32_t>();
        const int32_t* newIdx = newBestIdicies.template data<int32_t>();

        for (auto i = 0; i < data.dim32(0); i++) {
          std::swap(
              outData[i * block_size + newIdx[i]],
              outData[i * block_size + oldBestIdx[i]]);
        }
        return true;
        */
    }
}
