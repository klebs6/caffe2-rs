crate::ix!();

use crate::NetDef;

/**
  | This struct stores information about
  | the inference graph which defines underlying
  | math of BlackBoxPredictor. Other parts of
  | it such as various threading optimizations
  | don't belong here.
  |
  */
pub struct InferenceGraph {

    predict_init_net_def:  Box<NetDef>,

    /**
      | shared_ptr allows to share NetDef with its
      | operators on each of the threads without
      | memory replication. Note that
      | predict_init_net_def_ could be stored by
      | value as its operators are discarded
      | immidiatly after use (via RunNetOnce)
      */
    predict_net_def:              Arc<NetDef>,

    input_names:                  Vec<String>,
    output_names:                 Vec<String>,
    parameter_names:              Vec<String>,
    predictor_net_ssa_rewritten:  bool, // default = false
}
