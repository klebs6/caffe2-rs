crate::ix!();

/**
  | Retrieves blobs from scratch workspaces
  | (which contain intermediate recurrent
  | network computation for each timestep)
  | and puts them in the global workspace
  | under CPUContext.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct RecurrentNetworkBlobFetcherOp<Context> {
    storage: OperatorStorage,
    context: Context,
    prefix:  String,
    ws:      *mut Workspace,
}

num_inputs!{RecurrentNetworkBlobFetcher, 1}

num_outputs!{RecurrentNetworkBlobFetcher, 1}

inputs!{RecurrentNetworkBlobFetcher, 
    0 => ("ScratchWorkspaceBlob", "Name of scratch workspace blob returned by recurrent network.")
}

outputs!{RecurrentNetworkBlobFetcher, 
    0 => ("blob_names", "1D tensor of strings containing extracted blob names.")
}

args!{RecurrentNetworkBlobFetcher, 
    0 => ("prefix", "Prefix string to prepend extracted blobs.")
}

impl<Context> RecurrentNetworkBlobFetcherOp<Context> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator<Context>(operator_def, ws) 

        prefix_ = this->template GetSingleArgument<std::string>("prefix", "rnn");
        ws_ = ws;
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const detail::ScratchWorkspaces& scratch =
            this->template Input<detail::ScratchWorkspaces>(0);
        const std::vector<std::shared_ptr<Workspace>>& stepWorkspaces =
            scratch.stepWorkspaces;

        std::vector<std::string> blob_names_vector = {};

        for (int64_t i = 0; i < stepWorkspaces.size(); i++) {
          Workspace* currentStepWorkspace = stepWorkspaces[i].get();
          std::vector<std::string> blob_names = currentStepWorkspace->LocalBlobs();

          for (auto& blob_name : blob_names) {
            const Blob* currentBlob = currentStepWorkspace->GetBlob(blob_name);
            const auto& currentTensor = currentBlob->Get<Tensor>();

            std::string newBlobName =
                prefix_ + std::string("_") + blob_name + c10::to_string(i);
            blob_names_vector.push_back(newBlobName);

            BlobGetMutableTensor(ws_->CreateBlob(newBlobName), CPU)
                ->ResizeLike(currentTensor);
            auto type = Context::GetDeviceType();
            auto* newTensor = BlobGetMutableTensor(ws_->GetBlob(newBlobName), type);
            newTensor->CopyFrom(currentTensor);
          }
        }

        auto* output =
          Output(0, {static_cast<int64_t>(blob_names_vector.size())}, at::dtype<std::string>());
        std::copy(
            blob_names_vector.begin(),
            blob_names_vector.end(),
            output->template mutable_data<std::string>());

        return true;
        */
    }
}

register_cpu_operator!{
    RecurrentNetworkBlobFetcher,
    RecurrentNetworkBlobFetcherOp<CPUContext>
}

should_not_do_gradient!{RecurrentNetworkBlobFetcher}

register_cuda_operator!{
    RecurrentNetworkBlobFetcher,
    RecurrentNetworkBlobFetcherOp<CUDAContext>
}
