crate::ix!();

/**
  | Given a net, with primiary inputs and
  | outputs defined in its external_inputs/outputs,
  | and given the set of weights and extra
  | weights (created during conversion
  | to ONNX if exists), we check whether
  | some of the weights are used in the net,
  | and if so, we put it in the initialize_list
  | and add it to the external_inputs too.
  | 
  | -----------
  | @param net
  | 
  | [in] c2 net (cutoff from a bigger net)
  | ----------
  | @param weights_in_ws
  | 
  | [in] all the weights in the workspace
  | 
  | conversion \param initialization_list
  | [out] weights that needs<nonblock-prefix-expr-nostruct>
  | to be offload to backend
  | ----------
  | @param total_inputs_vec
  | 
  | [out] total #inputs of the net that doesn't
  | have a producer
  |
  */
#[inline] pub fn get_weights_and_inputs(
    net:                 &NetDef,
    weights_in_ws:       &HashSet<String>,
    extra_weights:       &Vec<String>,
    initialization_list: *mut HashSet<String>,
    total_inputs_vec:    *mut Vec<String>)  {
    
    todo!();
    /*
        std::unordered_set<std::string> total_inputs;

      // extra weights is definitely extra weights/inputs
      for (const auto& extra_weight : extra_weights) {
        if (total_inputs.emplace(extra_weight).second) {
          total_inputs_vec->emplace_back(extra_weight);
        }
        initialization_list->emplace(extra_weight);
      }

      // Boundary inputs that should not be weights
      std::unordered_set<std::string> boundary_inputs;
      for (const auto& i : net.external_input()) {
        boundary_inputs.emplace(i);
      }

      for (const auto& op : net.op()) {
        for (const auto& input : op.input()) {
          bool not_seen = total_inputs.emplace(input).second;
          if (!not_seen) {
            continue;
          }
          if (weights_in_ws.count(input)) {
            // We add weights as inputs too
            total_inputs_vec->emplace_back(input);
            initialization_list->emplace(input);
            VLOG(2) << "Add weights: " << input;
          } else if (boundary_inputs.count(input)) {
            VLOG(2) << "Adding boundary input: " << input;
            total_inputs_vec->emplace_back(input);
          }
        }
      }
    */
}
