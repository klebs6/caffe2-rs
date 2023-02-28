crate::ix!();

/**
  | Parameters for a Predictor provided
  | by name.
  | 
  | They are stored as shared_ptr to accommodate
  | parameter sharing
  |
  */
pub type PredictorParameters = HashMap<String, Arc<Blob>>;

/**
  | Stores parameters nessasary for creating
  | a PredictorInterface object.
  |
  */
pub struct PredictorConfig {

    /**
      | A map of parameter name to Tensor object.
      | Predictor is supposed to guarantee
      | constness of all these Tensor objects.
      |
      */
    parameters:           Arc<PredictorParameters>,
    predict_net:          Arc<NetDef>,

    /**
      | Input names of a model. User will have
      | to provide all of the inputs for inference
      |
      */
    input_names:          Vec<String>,

    /**
      | Output names of a model. All outputs
      | will be returned as results of inference
      |
      */
    output_names:         Vec<String>,

    /**
      | Parameter names of a model. Should be
      | a subset of parameters map passed in.
      |
      | We provide a separate set of parameter
      | names here as whole parameter set passed
      | in by a user might contain extra tensors
      | used by other models
      */
    parameter_names:      Vec<String>,

    /**
      | TODO We still save ws is because of the
      | current design of workspace and tensor.
      |
      | Once tensor support intrusive_ptr, we'll
      | get rid of this and use parameters to
      | construct Workspace
      */
    ws:                   Arc<Workspace>,
}

#[inline] pub fn make_workspace(parameters: Arc<PredictorParameters>) -> Workspace {
    
    todo!();
    /*
    
    */
}

/**
  | We don't use the getNet() from
  | predictor_utils.cc here because that file has
  | additional dependencies that we want to avoid
  | bringing in, to keep the binary size as small
  | as possible.
  */
#[inline] pub fn get_net<'a>(
    def:  &'a MetaNetDef,
    name: &String) -> &'a NetDef 
{
    todo!();
    /*
        for (const auto& n : def.nets()) {
        if (n.key() == name) {
          return n.value();
        }
      }
      CAFFE_THROW("Net not found: ", name);
    */
}

pub fn get_blobs<'a>(def: &'a MetaNetDef, name: &String) -> &'a RepeatedPtrField<String> {

    todo!();
    /*
      for (const auto& b : def.blobs()) {
        if (b.key() == name) {
          return b.value();
        }
      }
      CAFFE_THROW("Blob not found: ", name);
    */
}

#[inline] pub fn make_predictor_config(
    def:      &MetaNetDef,
    parent:   *mut Workspace,
    run_init: Option<bool>) -> PredictorConfig 
{
    let run_init = run_init.unwrap_or(true);
    todo!();
    /*
        const auto& init_net =
          getNet(def, PredictorConsts::default_instance().global_init_net_type());
      const auto& run_net =
          getNet(def, PredictorConsts::default_instance().predict_net_type());
      auto config = makePredictorConfig(init_net, run_net, parent, run_init);
      const auto& inputs =
          getBlobs(def, PredictorConsts::default_instance().inputs_blob_type());
      for (const auto& input : inputs) {
        config.input_names.emplace_back(input);
      }

      const auto& outputs =
          getBlobs(def, PredictorConsts::default_instance().outputs_blob_type());
      for (const auto& output : outputs) {
        config.output_names.emplace_back(output);
      }
      return config;
    */
}

#[inline] pub fn make_predictor_config_with_init_net_and_run_net(
    init_net:      &NetDef,
    run_net:       &NetDef,
    parent:        *mut Workspace,
    run_init:      Option<bool>,
    optimization:  Option<i32>) -> PredictorConfig 
{
    let run_init     = run_init.unwrap_or(true);
    let optimization = optimization.unwrap_or(1);

    todo!();
    /*
        PredictorConfig config;
      config.ws = std::make_shared<Workspace>(parent);
      auto& ws = *config.ws;
      config.predict_net = std::make_shared<NetDef>(run_net);
      if (run_init) {
        CAFFE_ENFORCE(ws.RunNetOnce(init_net));
      }
    #ifdef C10_MOBILE
      GlobalInit();
    #endif
      if (optimization &&
          !ArgumentHelper::HasArgument(*config.predict_net, "disable_nomnigraph")) {
    #ifdef CAFFE2_OPTIMIZER
        try {
          *config.predict_net =
              opt::optimize(*config.predict_net, &ws, optimization);
        } catch (const std::exception& e) {
          LOG(WARNING) << "Optimization pass failed: " << e.what();
        }
    #else
        static std::atomic<bool> warningEmitted{};
        // Emit the log only once.
        if (!warningEmitted.exchange(true)) {
          LOG(WARNING) << "Caffe2 is compiled without optimization passes.";
        }
    #endif
      }
      return config;
    */
}
