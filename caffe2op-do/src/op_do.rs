crate::ix!();

use crate::{
    OperatorStorage,
    Workspace,
    OperatorDef,
    NetDef
};

/**
  | 'Do' control operator, executes a subnet
  | in a separate workspace.
  | 
  | Last blobs in the input and output lists
  | should be the same blob created with
  | 
  | CreateScope op. Arguments 'inner_blobs'
  | and 'outer_blobs_idx' provide a mapping
  | between selected inner blob names and
  | corresponding outer blob indices.
  |
  */
pub struct DoOp<Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage:               OperatorStorage,
    context:               Context,

    blob_bindings:         HashMap<String,String>,
    forwarded_inner_blobs: HashSet<String>,
    is_gradient_op:        bool,
    copy_external_blobs:   bool,
    reuse_workspace:       bool,
    net_def:               NetDef,
    parent_ws:             *mut Workspace,
}

register_cuda_operator!{Do, DoOp<CUDAContext>}

register_cpu_operator!{Do, DoOp<CPUContext>}

num_inputs!{Do, (1,INT_MAX)}

num_outputs!{Do, (1,INT_MAX)}

args!{Do, 
    0 => ("net",               "Subnet with blob bindings"),
    1 => ("inner_blobs",       "List of inner net blob names to bind to outer workspace"),
    2 => ("outer_blobs_idx",   "Indices of corresponding outer workspace blobs, in order: operator inputs, operator outputs (skipping workspace blobs)"),
    3 => ("saved_fwd_blobs",   "List of blobs from the forward Do operator workspace needed in backward pass, used in gradient Do operator"),
    4 => ("reuse_workspace",   "Whether to reuse workspace or create a new one in a given scope")
}

allow_inplace!{Do, 
    |input: i32, output: i32| -> bool {
        true
    }
}

impl<Context> DoOp<Context> {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator<Context>(operator_def, ws), parent_ws_(ws) 

        CAFFE_ENFORCE(
            this->template HasSingleArgumentOfType<NetDef>("net"),
            "net must be specified in Do operator");
        net_def_ = this->template GetSingleArgument<NetDef>("net", NetDef());
        is_gradient_op_ = operator_def.is_gradient_op();
        copy_external_blobs_ =
            this->template GetSingleArgument<bool>("copy_external_blobs", false);
        reuse_workspace_ =
            this->template GetSingleArgument<bool>("reuse_workspace", false);
        CAFFE_ENFORCE(
            !(is_gradient_op_ && reuse_workspace_),
            "Gradient Do op requires use of stacked workspaces");
        CAFFE_ENFORCE(
            !(copy_external_blobs_ && reuse_workspace_),
            "Reuse workspace and copy external blobs simultaneously in Do op");

        const auto& inner_blobs =
            this->template GetRepeatedArgument<std::string>("inner_blobs");
        const auto& outer_blobs_idx =
            this->template GetRepeatedArgument<int>("outer_blobs_idx");
        CAFFE_ENFORCE_EQ(
            inner_blobs.size(),
            outer_blobs_idx.size(),
            "Invalid blob bindings: different inner/outer blobs lengths");

        const auto& outer_blob_names = checkAndGetOuterNames(operator_def);
        std::unordered_set<std::string> used_outer_names;
        for (size_t blob_idx = 0; blob_idx < inner_blobs.size(); ++blob_idx) {
          CAFFE_ENFORCE(
              !blob_bindings_.count(inner_blobs[blob_idx]),
              "Invalid blob bindings: redefinition of inner blob " +
                  inner_blobs[blob_idx]);
          CAFFE_ENFORCE(
              outer_blobs_idx[blob_idx] >= 0 &&
                  outer_blobs_idx[blob_idx] < outer_blob_names.size(),
              "Invalid blob bindings: outer blob index (" +
                  c10::to_string(outer_blobs_idx[blob_idx]) + ", inner name: " +
                  inner_blobs[blob_idx] + ") is out of bounds [0, " +
                  c10::to_string(outer_blob_names.size() - 1) + "]");
          const auto& outer_name = outer_blob_names[outer_blobs_idx[blob_idx]];
          CAFFE_ENFORCE(
              !used_outer_names.count(outer_name),
              "Reusage of outer name: " + outer_name);
          used_outer_names.insert(outer_name);
          blob_bindings_[inner_blobs[blob_idx]] = outer_name;
          forwarded_inner_blobs_.insert(inner_blobs[blob_idx]);
        }
        std::unordered_set<std::string> all_outer_names(
            outer_blob_names.begin(), outer_blob_names.end());
        CAFFE_ENFORCE_EQ(
            used_outer_names.size(),
            all_outer_names.size(),
            "Not all outer names are used in blob bindings");
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto* ws_stack =
            this->template Output<detail::WorkspaceStack>(OutputSize() - 1);
        std::shared_ptr<Workspace> net_workspace;
        if (is_gradient_op_) {
          net_workspace =
              ws_stack->popGradientWorkspace(parent_ws_, blob_bindings_);
        } else {
          if (reuse_workspace_ && !ws_stack->empty()) {
            net_workspace =
                ws_stack->reuseLastForwardWorkspace(parent_ws_, blob_bindings_);
          } else {
            net_workspace =
                ws_stack->pushForwardWorkspace(parent_ws_, blob_bindings_);
          }
        }
        CAFFE_ENFORCE(net_workspace, "Failed to initialize Do op workspace");

        // TODO(iliacher): figure how to reuse existing net with a new workspace
        auto* net = net_workspace->GetNet(net_def_.name());
        if (!net) {
          net = net_workspace->CreateNet(net_def_, true);
        }
        CAFFE_ENFORCE(net, "Failed to initialize subnet");
        auto success = net->Run();
        if (!is_gradient_op_ && copy_external_blobs_) {
          net_workspace->template CopyForwardedTensors<Context>(
              forwarded_inner_blobs_);
        }
        return success;
        */
    }

    /**
      | returns vector of input blob names
      | followed by output blob names in operator
      | definition order; ensures that input
      | (output) names are unique, checks number
      | of input (output) blobs
      */
    #[inline] pub fn check_and_get_outer_names(&self, operator_def: &OperatorDef) -> Vec<String> {
        
        todo!();
        /*
            auto input_names = getInputBlobNames(operator_def);
        CAFFE_ENFORCE(!input_names.empty(), "Expected at least one input blob");
        std::string input_ws_blob = input_names.back(); // copy
        // removing blob that holds pointer op workspace
        input_names.pop_back();

        std::unordered_set<std::string> all_input_names(
            input_names.begin(), input_names.end());
        CAFFE_ENFORCE_EQ(
            input_names.size(), all_input_names.size(), "Duplicate input blobs");

        auto output_names = getOutputBlobNames(operator_def);
        CAFFE_ENFORCE(!output_names.empty(), "Expected at least one output blob");
        const auto& output_ws_blob = output_names.back();
        CAFFE_ENFORCE_EQ(
            input_ws_blob,
            output_ws_blob,
            "Expected same input/output workspace blob");
        // remove blob that holds pointer to op workspace
        output_names.pop_back();

        std::unordered_set<std::string> all_output_names(
            output_names.begin(), output_names.end());
        CAFFE_ENFORCE_EQ(
            output_names.size(), all_output_names.size(), "Duplicate output blobs");

        std::vector<std::string> outer_blob_names;
        outer_blob_names.reserve(input_names.size() + output_names.size());
        outer_blob_names.insert(
            outer_blob_names.end(), input_names.begin(), input_names.end());
        outer_blob_names.insert(
            outer_blob_names.end(), output_names.begin(), output_names.end());
        return outer_blob_names;
        */
    }
    
    #[inline] pub fn get_input_blob_names(&self, operator_def: &OperatorDef) -> Vec<String> {
        
        todo!();
        /*
            std::vector<std::string> names;
        names.reserve(operator_def.input_size());
        for (auto idx = 0; idx < operator_def.input_size(); ++idx) {
          names.push_back(operator_def.input(idx));
        }
        return names;
        */
    }
    
    #[inline] pub fn get_output_blob_names(&self, operator_def: &OperatorDef) -> Vec<String> {
        
        todo!();
        /*
            std::vector<std::string> names;
        names.reserve(operator_def.output_size());
        for (auto idx = 0; idx < operator_def.output_size(); ++idx) {
          names.push_back(operator_def.output(idx));
        }
        return names;
        */
    }
}
