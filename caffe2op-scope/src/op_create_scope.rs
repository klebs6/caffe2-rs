crate::ix!();

use crate::{
    OperatorStorage,
    CPUContext,
    Workspace,
};

/**
  | Keeps track of forward and backward
  | gradient workspaces in stack, reuses
  | previously created workspaces, non-thread
  | safe
  |
  */
pub struct WorkspaceStack {

    blob_bindings:      HashMap<String,String>,
    grad_blob_bindings: HashMap<String,String>,
    parent_ws:          *mut Workspace,
    top:                i32,
    workspaces:         Vec<Arc<Workspace>>,
}

impl WorkspaceStack {
    
    pub fn new() -> Self {
        todo!();
        /*
            : parent_ws_(nullptr), top_(-1)
        */
    }
    
    #[inline] pub fn push_forward_workspace(
        &mut self, 
        parent_ws: *mut Workspace) -> Arc<Workspace> 
    {
        todo!();
        /*
            return pushForwardWorkspace(
            parent_ws, std::unordered_map<std::string, std::string>());
        */
    }
    
    #[inline] pub fn push_forward_workspace_with_bindings(
        &mut self, 
        parent_ws:     *mut Workspace, 
        blob_bindings: &HashMap<String,String>) -> Arc<Workspace> 
    {
        
        todo!();
        /*
            checkStack();
        if (FLAGS_caffe2_workspace_stack_debug) {
          if (parent_ws_) {
            CAFFE_ENFORCE_EQ(parent_ws_, parent_ws, "Parent workspace mismatch");
          } else {
            parent_ws_ = parent_ws;
          }
          if (!blob_bindings_.empty()) {
            checkBindingsMatch(blob_bindings_, blob_bindings);
          } else {
            blob_bindings_ = blob_bindings;
          }
        }

        if (top_ == workspaces_.size() - 1) {
          workspaces_.push_back(
              std::make_shared<Workspace>(parent_ws, blob_bindings));
        } else {
          // when reusing workspace, make sure copies of external blobs are
          // removed and blob bindings are set
          auto& workspace = workspaces_[top_ + 1];
          const auto& local_blobs = workspace->LocalBlobs();
          std::unordered_set<std::string> local_blobs_set;
          local_blobs_set.insert(local_blobs.begin(), local_blobs.end());
          bool found_local_copy = false;
          for (const auto& blob_pair : blob_bindings) {
            if (local_blobs_set.count(blob_pair.first)) {
              workspace->RemoveBlob(blob_pair.first);
              found_local_copy = true;
            }
          }
          if (found_local_copy) {
            workspace->AddBlobMapping(parent_ws, blob_bindings);
          }
        }

        return workspaces_[++top_];
        */
    }
    
    #[inline] pub fn pop_gradient_workspace(
        &mut self, 
        parent_ws:          *mut Workspace,
        grad_blob_bindings: &HashMap<String,String>) -> Arc<Workspace> 
    {
        
        todo!();
        /*
            checkStack();
        if (FLAGS_caffe2_workspace_stack_debug) {
          if (parent_ws_) {
            CAFFE_ENFORCE_EQ(parent_ws_, parent_ws, "Parent workspace mismatch");
          } else {
            parent_ws_ = parent_ws;
          }
          if (!grad_blob_bindings_.empty()) {
            checkBindingsMatch(grad_blob_bindings_, grad_blob_bindings);
          } else {
            grad_blob_bindings_ = grad_blob_bindings;
          }
        }

        if (top_ < 0) {
          return nullptr;
        }
        auto& grad_workspace = workspaces_[top_];
        grad_workspace->AddBlobMapping(parent_ws, grad_blob_bindings, true);
        --top_;
        return grad_workspace;
        */
    }
    
    #[inline] pub fn reuse_last_forward_workspace(
        &mut self, 
        parent_ws: *mut Workspace) -> Arc<Workspace> 
    {
        
        todo!();
        /*
            return reuseLastForwardWorkspace(
            parent_ws, std::unordered_map<std::string, std::string>());
        */
    }
    
    #[inline] pub fn reuse_last_forward_workspace_with_bindings(
        &mut self, 
        parent_ws: *mut Workspace,
        blob_bindings: &HashMap<String,String>) -> Arc<Workspace> 
    {
        todo!();
        /*
            checkStack();
        if (top_ < 0) {
          return nullptr;
        }
        workspaces_[top_]->AddBlobMapping(parent_ws, blob_bindings);
        return workspaces_[top_];
        */
    }
    
    #[inline] pub fn clear(&mut self)  {
        
        todo!();
        /*
            checkStack();
        top_ = -1;
        */
    }
    
    #[inline] pub fn empty(&self) -> bool {
        
        todo!();
        /*
            return top_ < 0;
        */
    }
    
    #[inline] pub fn check_stack(&self)  {
        
        todo!();
        /*
            CAFFE_ENFORCE_GT(
            (int)workspaces_.size(), top_, "Corrupted workspaces stack");
        */
    }
    
    #[inline] pub fn check_bindings_match(
        &self, 
        bindings: &HashMap<String,String>,
        test_bindings: &HashMap<String,String>)  
    {
        todo!();
        /*
            CAFFE_ENFORCE_EQ(
            bindings.size(), test_bindings.size(), "Blob bindings mismatch");
        for (const auto& blob_binding : bindings) {
          CAFFE_ENFORCE(
              test_bindings.count(blob_binding.first), "Blob bindings mismatch");
          CAFFE_ENFORCE_EQ(
              test_bindings.at(blob_binding.first),
              blob_binding.second,
              "Blob bindings mismatch");
        }
        */
    }
}

/**
  | 'CreateScope' operator initializes
  | and outputs empty scope that is used
  | by Do operator to store local blobs
  |
  */
pub struct CreateScopeOp<Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,
}

impl<Context> CreateScopeOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
}

impl CreateScopeOp<CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto* ws_stack = OperatorStorage::Output<detail::WorkspaceStack>(0);
      ws_stack->clear();
      return true;
        */
    }
}

register_cpu_operator!{CreateScope, CreateScopeOp<CPUContext>}

should_not_do_gradient!{CreateScope}

num_inputs!{CreateScope, 0}

num_outputs!{CreateScope, 1}

/**
  | Checks whether scope blob has any saved
  | scopes left
  |
  */
pub struct HasScopeOp<Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,
}

impl<Context> HasScopeOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
}

caffe_known_type!{WorkspaceStack}

impl HasScopeOp<CPUContext> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
          const auto& ws_stack = OperatorStorage::Input<detail::WorkspaceStack>(0);

          auto* output = Output(0, {1}, at::dtype<bool>());
          bool* output_value = output->template mutable_data<bool>();
          *output_value = !ws_stack.empty();
          return true;
        */
    }
}

register_cpu_operator!{HasScope, HasScopeOp<CPUContext>}

should_not_do_gradient!{HasScope}

num_inputs!{HasScope, 1}

num_outputs!{HasScope, 1}
