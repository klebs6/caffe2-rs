crate::ix!();

/**
  | Return a 1D tensor of strings containing
  | the names of each blob in the active workspace.
  |
  */
pub struct GetAllBlobNamesOp {
    storage:        OperatorStorage,
    context:        CPUContext,
    include_shared: bool,
    ws:             *mut Workspace,
}

num_inputs!{GetAllBlobNames, 0}

num_outputs!{GetAllBlobNames, 1}

outputs!{GetAllBlobNames, 
    0 => ("blob_names", "1D tensor of strings containing blob names.")
}

args!{GetAllBlobNames, 
    0 => ("include_shared", "(bool, default true) Whether to include blobs inherited from parent workspaces.")
}

register_cpu_operator!{GetAllBlobNames, GetAllBlobNamesOp}

should_not_do_gradient!{GetAllBlobNamesOp}

impl GetAllBlobNamesOp {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator<CPUContext>(operator_def, ws),
            include_shared_(GetSingleArgument<int>("include_shared", true)),
            ws_(ws)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
        const auto& blobs = include_shared_ ? ws_->Blobs() : ws_->LocalBlobs();
        auto* out = Output(0, {static_cast<int64_t>(blobs.size())}, at::dtype<std::string>());
        std::copy(
            blobs.begin(), blobs.end(), out->template mutable_data<std::string>());
        return true;
        */
    }
}
