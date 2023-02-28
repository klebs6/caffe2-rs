crate::ix!();

pub type Predictor_TensorList = Vec<TensorCPU>;
pub type Predictor_TensorMap  = HashMap<String,TensorCPU>;

pub struct Predictor {
    config:  PredictorConfig,
}

impl Predictor {
    
    #[inline] pub fn def(&self) -> &NetDef {
        
        todo!();
        /*
            return *config_.predict_net;
        */
    }
    
    #[inline] pub fn ws(&mut self) -> *mut Workspace {
        
        todo!();
        /*
            return config_.ws.get();
        */
    }
    
    #[inline] pub fn input_names(&self) -> &Vec<String> {
        
        todo!();
        /*
            return config_.input_names;
        */
    }
    
    #[inline] pub fn output_names(&self) -> &Vec<String> {
        
        todo!();
        /*
            return config_.output_names;
        */
    }
}

#[inline] pub fn enforce_is_tensor(ws: *mut Workspace, name: &String)  {
    
    todo!();
    /*
        auto blob = ws->GetBlob(name);
      CAFFE_ENFORCE(blob, "Blob does not exist: ", name);
      CAFFE_ENFORCE(
          BlobIsTensorType(*blob, CPU), "Blob is not a CPU Tensor: ", name);
    */
}

#[inline] pub fn get_blob(ws: *mut Workspace, name: &String) -> *mut Blob {
    
    todo!();
    /*
        enforceIsTensor(ws, name);
      auto* blob = ws->GetBlob(name);
      CAFFE_ENFORCE(blob, "Blob: ", name, " does not exist");
      return blob;
    */
}

#[inline] pub fn get_tensor(ws: *mut Workspace, name: &String) -> &Tensor {
    
    todo!();
    /*
        return *BlobGetMutableTensor(getBlob(ws, name), CPU);
    */
}

impl From<PredictorConfig> for Predictor {
    fn from(x: PredictorConfig) -> Self {
        todo!();
        /*
            : config_(std::move(config)) 

      const auto& initialized_vec = config_.ws->Blobs();
      const std::unordered_set<std::string> initialized{initialized_vec.begin(),
                                                        initialized_vec.end()};
      for (const auto& name : config_.predict_net->external_input()) {
        if (!initialized.count(name)) {
          auto* blob = config_.ws->CreateBlob(name);
          BlobGetMutableTensor(blob, CPU);
        }
      }
      CAFFE_ENFORCE(config_.ws->CreateNet(config_.predict_net));
        */
    }
}

impl Predictor {
    
    pub fn new(
        init_net:     &NetDef,
        run_net:      &NetDef,
        parent:       *mut Workspace,
        run_init:     Option<bool>,
        optimization: Option<i32>) -> Self 
    {
        let run_init = run_init.unwrap_or(true);
        let optimization = optimization.unwrap_or(1);
    
        todo!();
        /*
            : Predictor(makePredictorConfig(
              init_net,
              run_net,
              parent,
              run_init,
              optimization))
        */
    }
    
    /**
     | Executes `run_net` on the inputs.
     |
     | The first `inputs.size()` inputs from run_net::external_inputs
     | are shared with the data in `inputs`.
     |
     | Precondition:
     |   inputs.size() <= run_net_.external_inputs.size()
     |
     | Postcondition:
     |   outputs->size() == run_net.external_inputs.size()
     |
     | NOTE: output is a part of thread local workspace
     | and is only valid until the next predictor execution.
     |
     | Returns true on success
     */
    #[inline] pub fn invoke(&mut self, 
        inputs:  &Predictor_TensorList, 
        outputs: *mut Predictor_TensorList) -> bool {
        
        todo!();
        /*
            CAFFE_ENFORCE(
          inputs.size() <=
          static_cast<unsigned>(config_.predict_net->external_input_size()));
      for (size_t i = 0; i < inputs.size(); ++i) {
        // This is evil and shares the same underlying tensor
        BlobSetTensor(
            getBlob(config_.ws.get(), config_.predict_net->external_input(i)),
            inputs[i].UnsafeSharedInstance());
      }

      if (!config_.ws->RunNet(config_.predict_net->name())) {
        return false;
      }
      outputs->clear();
      for (size_t i = 0; i < config_.predict_net->external_output_size(); ++i) {
        outputs->emplace_back(
            getTensor(config_.ws.get(), config_.predict_net->external_output(i))
                .UnsafeSharedInstance());
      }
      return true;
        */
    }
    
    #[inline] pub fn run_map_workspace(&mut self, inputs: &Predictor_TensorMap) -> bool {
        
        todo!();
        /*
            if (!config_.input_names.empty()) {
        CAFFE_ENFORCE_EQ(inputs.size(), input_names().size());
      }
      for (auto& input : inputs) {
        if (!input_names().empty()) {
          CAFFE_ENFORCE(
              std::find(input_names().begin(), input_names().end(), input.first) !=
                  input_names().end(),
              "Input can't be found: ",
              input.first);
        }
        // This is evil and shares the same underlying tensor
        BlobSetTensor(
            getBlob(config_.ws.get(), input.first),
            input.second.UnsafeSharedInstance());
      }

      return config_.ws->RunNet(config_.predict_net->name());
        */
    }
    
    /**
      | Similar to run, but consumes a map of
      | name to tensor as input
      |
      */
    #[inline] pub fn invoke_with_tensor_list_outputs(
        &mut self, 
        inputs: &Predictor_TensorMap, 
        outputs: *mut Predictor_TensorList) -> bool {
        
        todo!();
        /*
            if (!run_map_workspace(inputs)) {
        return false;
      }
      outputs->clear();
      for (size_t i = 0; i < config_.predict_net->external_output_size(); ++i) {
        outputs->push_back(
            getTensor(config_.ws.get(), config_.predict_net->external_output(i))
                .UnsafeSharedInstance());
      }
      return true;
        */
    }
    
    /**
      | Similar to the other run fns, except
      | inputs and outputs are both maps of string
      | name to tensor.
      |
      */
    #[inline] pub fn invoke_with_tensor_map_outputs(
        &mut self, 
        inputs: &Predictor_TensorMap, 
        outputs: *mut Predictor_TensorMap) -> bool {
        
        todo!();
        /*
            if (!run_map_workspace(inputs)) {
        return false;
      }

      for (const std::string& outputName : output_names()) {
        outputs->emplace(
            outputName,
            getTensor(config_.ws.get(), outputName).UnsafeSharedInstance());
      }
      return true;
        */
    }
}
