crate::ix!();

#[inline] pub fn compute_blob_recycling_for_dag(
    net:                   &NetDef,
    heads:                 &Vec<String>,
    op_indices:            &Vec<i32>,
    shareable_blob_names:  &HashSet<String>,
    namescope:             &String,
    dont_share_blob_names: &HashSet<String>,
    blob_shapes:           &HashMap<String, Vec<i32>>) -> NetDef 
{
    todo!();
    /*
        ComputeBlobRecyclingForDag memonger(net.op_size());
      return memonger.OptimizeNet(
          net,
          heads,
          op_indices,
          shareable_blob_names,
          namescope,
          dont_share_blob_names,
          blob_shapes);
    */
}

pub struct ComputeBlobRecyclingForDag {

    tokens_counter:           i32,//1
    floats_saved:             i32,//0

    /// blob_name -> Op edges.
    blob_to_ops:              HashMap<String, Vec<i32>>,

    /// Current Op in degree.
    blob_input_count:         HashMap<String, i32>,

    /// Op in degree.
    op_inputs:                Vec<i32>,

    /// Current Op visit counts.
    op_visited_count:         Vec<i32>,

    share_counts:             HashMap<String, i32>,
    blob_sizes:               HashMap<String, i32>,

    req_tokens:               HashMap<String,HashSet<i32>>,

    op_token_deposit:         Vec<HashSet<i32>>,
    optim_op_outputs:         HashSet<String>,

    mapping:                  HashMap<String, String>,
    blob_device:              HashMap<String, DeviceOption>,

    /// The set of output blobs we already processed.
    processed_output_blobs:   HashSet<String>,
    op_visited:               Vec<bool>,
}

impl ComputeBlobRecyclingForDag {
    
    #[inline] pub fn process_op(&mut self, 
        net:                    &NetDef,
        shareable_blob_names:   &HashSet<String>,
        namescope:              &String,
        dont_share_blob_names:  &HashSet<String>,
        blob_shapes:            &HashMap<String, Vec<i32>>,
        op_index:               i32,
        free_blobs:             *mut Vec<(i32, String)>,
        tokens:                 *mut HashSet<i32>)  
    {
        todo!();
        /*
            // The tokens we have now is the union of current tokens operator is holding
        // and tokens pushed from parents.
        tokens->insert(
            op_token_deposit_[op_index].begin(), op_token_deposit_[op_index].end());
        op_token_deposit_[op_index].clear();
        CAFFE_ENFORCE(!op_visited_[op_index]);
        op_visited_[op_index] = true;

        const OperatorDef& current_op = net.op(op_index);

        // The set of freed input blobs by processing current op.
        std::vector<std::pair<int, string>> new_free_blobs;
        std::unordered_set<string> new_free_blobs_set;

        // Now update blob tokens.
        for (const auto& input : current_op.input()) {
          const auto& actual_blob = get_blob_or_mapped_blob(input);
          req_tokens_[actual_blob].insert(tokens->begin(), tokens->end());
          if (actual_blob != input) {
            req_tokens_[input].insert(tokens->begin(), tokens->end());
          }
        }
        for (const auto& output : current_op.output()) {
          const auto& actual_blob = get_blob_or_mapped_blob(output);
          req_tokens_[actual_blob].insert(tokens->begin(), tokens->end());
          if (actual_blob != output) {
            req_tokens_[output].insert(tokens->begin(), tokens->end());
          }
        }

        // Increment blob count and check if we can free input blobs.
        for (const auto& input : current_op.input()) {
          if (has_key(shareable_blob_names, input)) {
            blob_input_count_[input]++;
            if (blob_input_count_[input] == (int)blob_to_ops_[input].size()) {
              const string& actual_blob = get_blob_or_mapped_blob(input);
              if (!has_key(dont_share_blob_names, actual_blob)) {
                new_free_blobs.emplace_back(
                    -share_counts_[actual_blob], actual_blob);
                new_free_blobs_set.insert(actual_blob);
              }
            }
          }
        }

        // Check if we can recycle free blobs and use it as output blob.
        for (const auto& output : current_op.output()) {
          if (has_key(shareable_blob_names, output) &&
              !has_key(processed_output_blobs_, output) &&
              !has_key(new_free_blobs_set, output)) {
            const string freed_blob = get_free_blob(
                output, blob_shapes, tokens, free_blobs, blob_device_[output]);
            if (freed_blob != "") {
              req_tokens_[freed_blob].insert(tokens->begin(), tokens->end());
              share_counts_[freed_blob]++;
              mapping_[output] = freed_blob;
            }
            processed_output_blobs_.insert(output);
          }
        }

        // Insert new freed blobs.
        std::unordered_set<string> free_blob_set;
        for (const auto& free_blob : *free_blobs) {
          free_blob_set.insert(free_blob.second);
        }
        for (const auto& new_free_blob : new_free_blobs) {
          if (!has_key(free_blob_set, new_free_blob.second)) {
            free_blobs->push_back(new_free_blob);
            if (blob_shapes.size() > 0) {
              if (!has_key(blob_sizes_, new_free_blob.second)) {
                blob_sizes_.insert(
                    {new_free_blob.second,
                     infer_blob_size(new_free_blob.second, blob_shapes)});
              }
            }
            std::push_heap(
                free_blobs->begin(),
                free_blobs->end(),
                std::greater<std::pair<int, string>>());
          }
        }

        int num_branches = 0;
        for (const auto& output : current_op.output()) {
          num_branches += blob_to_ops_[output].size();
        }

        for (const auto& output : current_op.output()) {
          for (const auto& input_op_index : blob_to_ops_[output]) {
            op_visited_count_[input_op_index]++;
            if (op_visited_count_[input_op_index] == op_inputs_[input_op_index]) {
              std::unordered_set<int> new_tokens;
              new_tokens.insert(tokens->begin(), tokens->end());
              if (num_branches > 1) {
                new_tokens.insert(tokens_counter_++);
              }
              process_op(
                  net,
                  shareable_blob_names,
                  namescope,
                  dont_share_blob_names,
                  blob_shapes,
                  input_op_index,
                  free_blobs,
                  &new_tokens);
            } else {
              if (!op_visited_[input_op_index]) {
                op_token_deposit_[input_op_index].insert(
                    tokens->begin(), tokens->end());
              }
            }
          }
        }
        */
    }
    
    #[inline] pub fn infer_blob_size(
        &mut self, 
        blob_name: &String,
        blob_shapes: &HashMap<String, Vec<i32>>) -> i32 
    {
        todo!();
        /*
            const auto& blob_shapes_iter = blob_shapes.find(blob_name);
        if (blob_shapes_iter == blob_shapes.end()) {
          return 0;
        }
        int size = 1;
        for (size_t i = 0; i < blob_shapes_iter->second.size(); ++i) {
          size *= blob_shapes_iter->second[i];
        }
        return size;
        */
    }
    
    #[inline] pub fn get_blob_or_mapped_blob(&mut self, blob_name: &String) -> String {
        
        todo!();
        /*
            auto mapped_blob = mapping_.find(blob_name);
        if (mapped_blob == mapping_.end()) {
          return blob_name;
        } else {
          return mapped_blob->second;
        }
        */
    }
    
    /// Returns true if the op that generates that blob acquires all tokens.
    #[inline] pub fn can_use_blob(
        &mut self, 
        blob_name:     &String,
        tokens:        *mut HashSet<i32>,
        device_option: &DeviceOption) -> bool 
    {
        todo!();
        /*
            const DeviceOption& blob_device = blob_device_[blob_name];
        if (device_option.device_type() != blob_device.device_type() ||
            device_option.device_id() != blob_device.device_id()) {
          return false;
        }
        for (const int token : req_tokens_[blob_name]) {
          if (tokens->find(token) == tokens->end()) {
            return false;
          }
        }
        return true;
        */
    }

    /// Returns the name of the blob that we are going to map blob_name into.
    #[inline] pub fn get_free_blob(
        &mut self, 
        blob_name:   &String,
        blob_shapes: &HashMap<String, Vec<i32>>,
        tokens:      *mut HashSet<i32>,
        free_blobs:  *mut Vec<(i32, String)>,
        device:      &DeviceOption) -> String 
    {

        todo!();
        /*
            string freed_blob = "";
        if (blob_shapes.size() == 0) {
          std::vector<std::pair<int, string>> cant_use_blobs;
          while (free_blobs->size() > 0) {
            std::pop_heap(
                free_blobs->begin(),
                free_blobs->end(),
                std::greater<std::pair<int, string>>());
            const auto cand_free_blob = free_blobs->back();
            free_blobs->pop_back();
            if (can_use_blob(cand_free_blob.second, tokens, device)) {
              freed_blob = cand_free_blob.second;
              break;
            } else {
              cant_use_blobs.push_back(cand_free_blob);
            }
          }
          for (const auto& cant_use_blob : cant_use_blobs) {
            free_blobs->push_back(cant_use_blob);
            std::push_heap(
                free_blobs->begin(),
                free_blobs->end(),
                std::greater<std::pair<int, string>>());
          }
        } else {
          // Heuristic to choose the largest blob to fit output thats
          // slightly less than blob_size.
          const int blob_size = infer_blob_size(blob_name, blob_shapes);
          int best_size = -1;
          int free_blob_index = -1;
          for (size_t i = 0; i < free_blobs->size(); ++i) {
            const string& cb_name = (*free_blobs)[i].second;
            if (can_use_blob(cb_name, tokens, device)) {
              const int cand_bz = blob_sizes_[cb_name];
              CAFFE_ENFORCE(blob_sizes_.find(cb_name) != blob_sizes_.end());
              if (cand_bz >= best_size) {
                if (best_size < blob_size || best_size >= cand_bz) {
                  best_size = cand_bz;
                  free_blob_index = i;
                }
              }
            }
          }
          if (free_blob_index != -1) {
            floats_saved_ += best_size;
            freed_blob = (*free_blobs)[free_blob_index].second;
            free_blobs->erase(free_blobs->begin() + free_blob_index);
          }
        }
        return freed_blob;
        */
    }
    
    pub fn new(size: i32) -> Self {
        todo!();
        /*
            : op_inputs_(size),
            op_visited_count_(size),
            op_token_deposit_(size),
            op_visited_(size, false)
        */
    }

     #[inline] pub fn has_key_in_map<K, V>(
         &mut self, 
         in_map: &HashMap<K, V>, 
         key: &K) -> bool 
     {
         todo!();
         /*
            return in_map.find(key) != in_map.end();
         */
     }

     #[inline] pub fn has_key_in_set<K>(
         &mut self, 
         in_set: &HashSet<K>, 
         key: &K) -> bool 
     {
         todo!();
         /*
            return in_set.find(key) != in_set.end();
         */
     }

    pub fn optimize_net(
        &mut self,
        net:                    &NetDef,
        heads:                  &Vec<String>,
        op_indices:             &Vec<i32>,
        shareable_blob_names:   &HashSet<String>,
        namescope:              &String,
        dont_share_blob_names:  &HashSet<String>,
        blob_shapes:            &HashMap<String, Vec<i32>>) -> NetDef 
    {
        todo!();
          /*
            // Memonger modifies the graph. Do an early schema check here to make sure
            // the operators are valid
            run_schema_check(net);
            // Construct the set of input blobs.
            std::unordered_set<string> heads_blobs_set(heads.begin(), heads.end());

            // Construct the set of output blobs we want to optimize.
            // Blobs not eligible for sharing are filtered out
            for (const int op_index : op_indices) {
              for (const auto& output : net.op(op_index).output()) {
                if (has_key(shareable_blob_names, output) && !has_key(dont_share_blob_names, output)) {
                  optim_op_outputs_.insert(output);
                }
              }
            }

            // Compute operators in degree (op_inputs_) and initialize how many ops are
            // sharing input blobs (share_counts_).
            // Note: We have to handle the cases where output blobs are shared.
            std::unordered_map<string, int> blob_seen;
            for (const int op_index : op_indices) {
              for (const auto& input : net.op(op_index).input()) {
                if (has_key(shareable_blob_names, input) ||
                    has_key(heads_blobs_set, input)) {
                  if (has_key(optim_op_outputs_, input)) {
                    CAFFE_ENFORCE(
                        blob_seen.find(input) != blob_seen.end(),
                        "Input ",
                        input,
                        " was not output by an op before");
                    op_inputs_[op_index] += blob_seen[input];
                  } else {
                    share_counts_[input] = 1;
                  }
                  blob_to_ops_[input].push_back(op_index);
                }
              }
              for (const auto& output : net.op(op_index).output()) {
                blob_seen[output] += 1;
                blob_device_[output] = net.op(op_index).device_option();
                // Exception for CopyGPUToCPU that has
                // cuda device option but whose inputs/outputs are on CPU
                if (net.op(op_index).type() == "CopyGPUToCPU") {
                  blob_device_[output].set_device_type(0);
                  blob_device_[output].set_device_id(0);
                }
              }
            }

            // The main recursive call. Here we do start DFS in the operator graph
            // from the input blobs. Note that the input ordering does not indicate
            // operator graph ordering. To avoid traversing children operators first,
            // traversal begins from root ops and then recursively children ops are
            // visited.
            for (const auto& input_blob : heads) {
              for (const int op_index : blob_to_ops_[input_blob]) {
                if (!op_visited_[op_index] && !op_inputs_[op_index]) {
                  vector<std::pair<int, string>> free_blobs;
                  std::unordered_set<int> tokens{tokens_counter_++};
                  process_op(
                      net,
                      shareable_blob_names,
                      namescope,
                      dont_share_blob_names,
                      blob_shapes,
                      op_index,
                      &free_blobs,
                      &tokens);
                }
              }
            }

            // Rename mapped blobs.
            std::unordered_map<string, string> renamed;
            int name_idx = 0;
            std::unordered_set<string> mapped_blobs_set;
            for (const auto& mapped_blob : mapping_) {
              mapped_blobs_set.insert(mapped_blob.second);
              if (has_key(optim_op_outputs_, mapped_blob.second)) {
                if (renamed.find(mapped_blob.second) == renamed.end()) {
                  renamed.insert(
                      {mapped_blob.second,
                       namescope + "__m" + c10::to_string(name_idx++) + "_shared"});
                }
              } else {
                renamed.insert({mapped_blob.second, mapped_blob.second});
              }
            }

            // Recursively rename mapped_blobs.
            mapping_.insert(renamed.begin(), renamed.end());
            bool had_changes = true;
            while (had_changes) {
              had_changes = false;
              for (const auto mapped_blob : mapping_) {
                if (has_key(renamed, mapped_blob.second) &&
                    renamed[mapped_blob.second] != mapped_blob.second) {
                  renamed[mapped_blob.first] = renamed[mapped_blob.second];
                  mapping_[mapped_blob.first] = renamed[mapped_blob.first];
                }
              }
            }

            NetDef optimized_net = apply_assignments(net);
            LOG(INFO) << "Remapping " << mapping_.size() << " using "
                      << mapped_blobs_set.size() << " shared blobs.";
            if (floats_saved_ > 0) {
              LOG(INFO) << "Memonger saved approximately : "
                        << (floats_saved_ * 4.0 / 1024.0 / 1024.0) << " MB.";
            }

            return optimized_net;

          */
    }
    
    #[inline] pub fn apply_assignments(&mut self, net: &NetDef) -> NetDef {
        
        todo!();
        /*
            NetDef optimized_net = net;
        // Rename optimized_net blobs.
        for (int i = 0; i < optimized_net.op_size(); ++i) {
          // Special handling for RNNs, which have internal nets that
          // can refer to memongered blobs
          if (optimized_net.op(i).type().find("RecurrentNetwork") == 0) {
            apply_recurrent_blob_assignments(optimized_net.mutable_op(i));
          }

          // Special handling for AsyncIf ops, where internal nets can
          // refer to memongered blobs
          if (optimized_net.op(i).type() == "AsyncIf") {
            apply_asyncif_blob_assignments(optimized_net.mutable_op(i));
          }

          for (int j = 0; j < optimized_net.op(i).input_size(); ++j) {
            const string& input_name =
                get_blob_or_mapped_blob(optimized_net.op(i).input(j));
            optimized_net.mutable_op(i)->set_input(j, input_name);
          }

          for (int j = 0; j < optimized_net.op(i).output_size(); ++j) {
            auto output_name =
                get_blob_or_mapped_blob(optimized_net.op(i).output(j));
            optimized_net.mutable_op(i)->set_output(j, output_name);
          }
        }
        return optimized_net;
        */
    }
    
    #[inline] pub fn apply_recurrent_blob_assignments(&mut self, op: *mut OperatorDef)  {
        
        todo!();
        /*
            // Recursively map stepnets in RecurrentNetworks, and
        // attach a mapping table
        for (int i = 0; i < op->arg_size(); i++) {
          Argument* arg = op->mutable_arg(i);
          const string& name = arg->name();
          if (name == "step_net" || name == "backward_step_net") {
            if (arg->has_n()) {
              NetDef* step_net_ref = arg->mutable_n();
              CAFFE_ENFORCE(
                  !arg->has_s(),
                  "Invalid definition for ",
                  name,
                  ". Only one of NetDef and string should be present");
              NetDef optimized_net = apply_assignments(*step_net_ref);
              step_net_ref->CopyFrom(optimized_net);
            } else {
              NetDef step_net;
              CAFFE_ENFORCE(
                  TextFormat::ParseFromString(
                      arg->s(), &step_net),
                  "Could not parse step net:",
                  name);
              step_net = apply_assignments(step_net);
              arg->set_s(ProtoDebugString(step_net));
            }
          }
        }

        // Store renamings
        vector<string> inputs_outputs(op->input().begin(), op->input().end());
        inputs_outputs.insert(
            inputs_outputs.end(), op->output().begin(), op->output().end());

        for (auto& b : inputs_outputs) {
          string mapped = get_blob_or_mapped_blob(b);
          if (b != mapped) {
            Argument* map_arg = op->add_arg();
            map_arg->set_name(b + ".rename");
            map_arg->set_s(mapped);
          }
        }
        */
    }
    
    #[inline] pub fn apply_asyncif_blob_assignments(&mut self, op: *mut OperatorDef)  {
        
        todo!();
        /*
            for (int i = 0; i < op->arg_size(); i++) {
          Argument* arg = op->mutable_arg(i);
          const string& name = arg->name();
          if (name == "then_net" || name == "else_net") {
            NetDef* step_net_ref = arg->mutable_n();
            NetDef optimized_net = apply_assignments(*step_net_ref);

            // update external inputs and outputs mappings as well
            // for this internal net
            std::vector<string> optim_external_inputs;
            for (auto& blob_name : optimized_net.external_input()) {
              optim_external_inputs.push_back(get_blob_or_mapped_blob(blob_name));
            }
            optimized_net.mutable_external_input()->Clear();
            for (const auto& blob_name : optim_external_inputs) {
              optimized_net.add_external_input(blob_name);
            }

            std::vector<string> optim_external_outputs;
            for (auto& blob_name : optimized_net.external_output()) {
              optim_external_outputs.push_back(get_blob_or_mapped_blob(blob_name));
            }
            optimized_net.mutable_external_output()->Clear();
            for (const auto& blob_name : optim_external_outputs) {
              optimized_net.add_external_output(blob_name);
            }

            step_net_ref->CopyFrom(optimized_net);
          }
        }
        */
    }
}
