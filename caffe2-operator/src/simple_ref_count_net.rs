crate::ix!();

/**
  | SimpleRefcountNet is an implementation
  | that adds an additional abstraction
  | on top of SimpleRefCountNet: it tracks
  | all the tensors and for those that are
  | considered internal/temporary, delete
  | them once their refcount go to zero.
  | 
  | In the context of a simple static run,
  | this can be carried out during construction
  | time: we will do a pass through the network
  | and track what blobs we need to do reset
  | on, after the execution of every op.
  | 
  | To identify which blob is considered
  | temporary, we employ the following
  | strategy: any blob that is
  | 
  | (1) consumed but not produced by ops
  | in the net, or
  | 
  | (2) produced but not consumed by ops
  | in the net, or
  | 
  | (3) is marked as external_output in
  | the protobuf will NOT be considered
  | temporary.
  | 
  | In the long run, we should design proper
  | functional interfaces so that nets
  | are less imperative and more functional.
  | 
  | Also, for now, SimpleRefCountNet should
  | only be used for benchmarking purposes
  | and not product use, since it is not going
  | to provide better performance gain,
  | and is implicitly incompatible with
  | the contract that earlier Nets expose
  | - that all intermediate blobs are visible
  | to the users.
  |
  */
pub struct SimpleRefCountNet {
    base: SimpleNet,

    /**
      | The list of blobs to delete when each
      | operator finishes its run.
      |
      | This will be populated during construction
      | time.
      */
    delete_list:  Vec<Vec<*mut Blob>>,
}

impl SimpleRefCountNet {

    pub fn new(net_def: &Arc<NetDef>, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : SimpleNet(net_def, ws) 

      VLOG(1) << "Constructing SimpleRefCountNet " << net_def->name();
      // Construct the "to delete" list.
      delete_list_.resize(net_def->op_size());

      std::map<string, int> last_consumed_at;
      std::set<string> created_by_me;
      // For each operator
      for (int idx = 0; idx < net_def->op_size(); ++idx) {
        const auto& op_def = net_def->op(idx);
        for (const string& in_name : op_def.input()) {
          last_consumed_at[in_name] = idx;
        }
        for (const string& out_name : op_def.output()) {
          created_by_me.insert(out_name);
        }
      }
      // We do not delete any operator that is not produced by the net, and
      // any operator that is marked as external_output. Any blob that is not
      // consumed won't be in the last_consumed_at map, so we don't need to
      // do anything special.
      for (auto& kv : last_consumed_at) {
        if (!created_by_me.count(kv.first)) {
          kv.second = -1;
        }
      }
      for (const string& name : net_def->external_output()) {
        last_consumed_at[name] = -1;
      }
      // Set up the delete list.
      for (auto& kv : last_consumed_at) {
        if (kv.second > 0) {
          delete_list_[kv.second].push_back(ws->GetBlob(kv.first));
          VLOG(1) << "NetSimpleRefCountNet: will delete " << kv.first
                  << " at operator #" << kv.second;
        }
      }
        */
    }
    
    #[inline] pub fn run(&mut self) -> bool {
        
        todo!();
        /*
            StartAllObservers();
      VLOG(1) << "Running net " << name_;
      for (auto op_id = 0U; op_id < operators_.size(); ++op_id) {
        auto& op = operators_[op_id];
        VLOG(1) << "Running operator " << op->debug_def().name() << "("
                << op->debug_def().type() << ").";
        bool res = op->Run();
        if (!res) {
          LOG(ERROR) << "Operator failed: " << ProtoDebugString(op->debug_def());
          return false;
        }
        for (Blob* blob : delete_list_[op_id]) {
          blob->Reset();
        }
      }
      StopAllObservers();
      return true;
        */
    }
}

register_net!{simple_refcount, SimpleRefCountNet}
