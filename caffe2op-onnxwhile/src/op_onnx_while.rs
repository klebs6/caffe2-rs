crate::ix!();

pub struct OnnxWhileOpLocalScope {

    loop_ws:             *mut Workspace,

    /// owned by a workspace
    body_net:            *mut NetBase,
    iteration_var:       *mut Tensor,
    input_condition_var: *mut Tensor,
    condition_var:       *mut Tensor,
    lcd_tensors:         Vec<*mut Tensor>,
}

impl OnnxWhileOpLocalScope {
    
    pub fn new(
        loop_ws:      *mut Workspace,
        body_net_def: &NetDef,
        num_lcds:     usize) -> Self {
        todo!();
        /*
            : loop_ws_(loop_ws) 

          CAFFE_ENFORCE(loop_ws_, "Failed to initialize local loop workspace");

          // Create loop-carried deps in Workspace
          lcd_tensors_.clear();
          for (int i = 2; i < num_lcds + 2; ++i) {
            Blob* b = loop_ws_->CreateBlob(body_net_def.external_input(i));
            Tensor* t = BlobGetMutableTensor(b, Context::GetDeviceType());
            lcd_tensors_.push_back(t);
          }
          // First output is the iteration variable
          auto* iteration_var_blob =
              loop_ws_->CreateBlob(body_net_def.external_input(0));
          iteration_var_ =
              BlobGetMutableTensor(iteration_var_blob, Context::GetDeviceType());

          input_condition_var_ = BlobGetMutableTensor(
              loop_ws_->CreateBlob(body_net_def.external_input(1)),
              Context::GetDeviceType());

          auto* condition_var_blob =
              loop_ws_->CreateBlob(body_net_def.external_output(0));
          condition_var_ =
              BlobGetMutableTensor(condition_var_blob, Context::GetDeviceType());
          condition_var_->Resize(1);
          condition_var_->template mutable_data<bool>();

          body_net_ = loop_ws_->GetNet(body_net_def.name());
          if (!body_net_) {
            body_net_ = loop_ws_->CreateNet(body_net_def, true);
          }
          CAFFE_ENFORCE(body_net_, "Failed to initialize loop subnet");
        */
    }
    
    #[inline] pub fn net(&self) -> *mut NetBase {
        
        todo!();
        /*
            return body_net_;
        */
    }
    
    #[inline] pub fn workspace(&self) -> *mut Workspace {
        
        todo!();
        /*
            return loop_ws_;
        */
    }
    
    #[inline] pub fn iteration(&self) -> i64 {
        
        todo!();
        /*
            auto* iteration_var_ptr =
              iteration_var_->template mutable_data<int64_t>();
          return *iteration_var_ptr;
        */
    }
    
    #[inline] pub fn lcd_tensor(&mut self, idx: i32) -> *mut Tensor {
        
        todo!();
        /*
            return lcd_tensors_[idx];
        */
    }
    
    #[inline] pub fn set_iteration(&mut self, itr: i64)  {
        
        todo!();
        /*
            iteration_var_->Resize();
          auto* iteration_var_ptr =
              iteration_var_->template mutable_data<int64_t>();
          *iteration_var_ptr = itr;
        */
    }
    
    #[inline] pub fn set_input_condition<CondVarType>(&mut self, cond_value: bool)  {
        todo!();
        /*
            input_condition_var_->Resize(1);
          auto* input_condition_var_ptr =
              input_condition_var_->template mutable_data<CondVarType>();
          *input_condition_var_ptr = cond_value;
        */
    }
    
    #[inline] pub fn output_condition<CondVarType>(&self) -> bool {
        todo!();
        /*
            auto* condition_var_ptr =
              condition_var_->template mutable_data<CondVarType>();
          return *condition_var_ptr;
        */
    }
}

/**
| EXPERIMENTAL. This operator is
| a work-in-progress. No assumption should be made
| about the stability or correctness of this op. ***
|
| Generic Looping construct confirming to the ONNX
| Loop operator spec. This loop has multiple
| termination conditions:
|
| 1. Trip count. Iteration count specified at
|    runtime. Set by specifying the input
|    M. Optional. Set to empty string to omit. Note
|    that a static trip count (specified at graph
|    construction time) can be specified by passing in
|    a constant node for input M.
|
| 2. Loop termination condition. This is an input to
|    the op that determines whether to run the first
|    interation and also a loop-carried dependency for
|    the body graph. The body graph must yield a value
|    for the condition variable, whether this input is
|    provided or not.
|
| This table summarizes the operating modes of this
| operator with equivalent C-style code:
|
| Operator inputs defined as (max_trip_count,
| condition_var). Omitted optional inputs are
| represented as empty string. Concretely, in this
| caffe2 op an input is marked as omitted by setting
| its 'has_{name}' argument to False.
|
|     input ("", ""):
|         for (int i=0; ; ++i) {
|           cond = ... // Note this value is ignored, but is required in the body
|         }
|
|     input ("", cond) // Note this is analogous to a while loop
|         bool cond = ...;
|         for (int i=0; cond; ++i) {
|           cond = ...;
|         }
|
|     input ("", 1) // Note this is analogous to a do-while loop
|         bool cond = true
|         for (int i=0; cond; ++i) {
|           cond = ...;
|         }
|
|     input (trip_count, "") // Note this is analogous to a for loop
|         int trip_count = ...
|         for (int i=0; i < trip_count; ++i) {
|           cond = ...; // ignored
|         }
|
|     input (trip_count, cond)
|         int trip_count = ...;
|         bool cond = ...;
|         for (int i=0; i < trip_count && cond; ++i) {
|           cond = ...;
|         }
*/
pub struct ONNXWhileOp<Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage:                 OperatorStorage,
    context:                 Context,
    body_net_def:            NetDef,
    parent_ws:               *mut Workspace,
    ws_stack:                WorkspaceStack,
    has_trip_count:          bool,
    has_cond:                bool,
    save_scopes:             bool,
    disable_scopes:          bool,
    num_loop_carried_deps:   i64,
    scope:                   Arc<OnnxWhileOpLocalScope>,
}

num_inputs!{ONNXWhile, (2,INT_MAX)}

num_outputs!{ONNXWhile, (0,INT_MAX)}

inputs!{ONNXWhile, 
    0 => ("max_trip_count",       "Number of iterations to go out to. Used if the flag has_trip_count is True."),
    1 => ("first_iter_condition", "Dynamic condition value for the first iteration. For all subsequent iterations, the condition from the body graph is used. This input is used if the flag has_cond is true.")
}

args!{ONNXWhile, 
    0 => ("body", "Net executed on each iteration"),
    1 => ("has_trip_count", "Whether to use the trip count input"),
    2 => ("has_cond", "Whether to use the condition input"),
    3 => ("save_scopes", "Whether to save the scopes across iterations, as in for backprop"),
    4 => ("disable_scopes", "Do not create new scopes. Use this only if you're certain there will be no name collision, for example if you're converting from a fully-SSA IR")
}

allow_inplace!{ONNXWhile, 
    |input: i32, output: i32| -> bool {
        true
    }
}

impl<Context> ONNXWhileOp<Context> {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator<Context>(operator_def, ws),
            parent_ws_(ws),
            has_trip_count_(
                this->template GetSingleArgument<int64_t>("has_trip_count", 0)),
            has_cond_(this->template GetSingleArgument<int64_t>("has_cond", 0)),
            save_scopes_(
                this->template GetSingleArgument<int64_t>("save_scopes", 0)),
            disable_scopes_(
                this->template GetSingleArgument<int64_t>("disable_scopes", 0)),
            num_loop_carried_deps_(this->template GetSingleArgument<int64_t>(
                "num_loop_carried_deps",
                -1)) 

        CAFFE_ENFORCE(
            this->template HasSingleArgumentOfType<NetDef>("body"),
            "body net must be specified in ONNXWhile operator");
        if (disable_scopes_) {
          CAFFE_ENFORCE(
              !save_scopes_, "Cannot save scopes when disable_scopes=True");
        }
        body_net_def_ = this->template GetSingleArgument<NetDef>("body", NetDef());
        static int64_t counter = -1;
        if (!body_net_def_.has_name()) {
          if (counter == -1) {
            ++counter;
            body_net_def_.set_name("loop_net");
          } else {
            ++counter;
            body_net_def_.set_name("loop_net." + c10::to_string(counter));
          }
        }
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<int, bool, long>>::call(this, Input(1));
        */
    }

    /**
      | Operator
      |
      |  Inputs: max trip count, condition,
      |  initial loop-carried dependencies
      |
      |  Outputs: Final loop-carried dependencies,
      |  scan_outputs
      |
      | Body
      |
      |  Inputs: iteration number, condition,
      |  loop-carried dependencies
      |
      |  Outputs: condition, loop-carried
      |  dependencies, scan_outputs
      */
    #[inline] pub fn do_run_with_type<CondVarType>(&mut self) -> bool {
        todo!();
        /*
            // Clear workspaces from the previous invocations of the loop
        // and setup a local scope for the first iteration
        ws_stack_.clear();
        auto loop_ws = !disable_scopes_
            ? ws_stack_.pushForwardWorkspace(parent_ws_).get()
            : parent_ws_;

        constexpr int64_t num_inputs_before_lcds = 2;
        // First input is the maximumt trip count. Second input is the condition
        // variable (for the first iteration). The rest of the inputs are
        // loop-carried dependencies.
        int64_t num_loop_carried_deps;
        if (num_loop_carried_deps_ != -1) {
          num_loop_carried_deps = num_loop_carried_deps_;
        } else {
          num_loop_carried_deps = InputSize() - num_inputs_before_lcds;
        }
        int64_t max_trip_count = *Input(0).template data<int64_t>();
        const bool first_iter_condition = *Input(1).template data<CondVarType>();

        scope_ = std::make_shared<LocalScope>(
            loop_ws, body_net_def_, num_loop_carried_deps);

        // Body graph has 1+N+K outputs: recalculated condition variable, N
        // loop-carried dependencies, and K scan_outputs
        int num_scan_outputs =
            scope_->net()->external_output().size() - num_loop_carried_deps - 1;

        CAFFE_ENFORCE_GE(
            num_scan_outputs,
            0,
            "Body graph must have N+K outputs, where N is the number "
            "of loop-carried dependencies and K is the number of scan "
            "outputs");

        // Copy initial loop-carried dependencies
        for (int i = 0; i < num_loop_carried_deps; ++i) {
          scope_->lcd_tensor(i)->CopyFrom(Input(i + num_inputs_before_lcds));
        }

        // Initialize iteration variable
        scope_->set_iteration(0ll);

        // Initialize input condition variable
        scope_->template set_input_condition<CondVarType>(first_iter_condition);

        auto valid_iter_num = [this, max_trip_count](int64_t i) {
          if (has_trip_count_) {
            return i < max_trip_count;
          } else {
            return true;
          }
        };

        auto condition_true = [this, first_iter_condition](
                                  int64_t i, bool cond_value) {
          if (has_cond_) {
            if (i == 0) {
              return (bool)first_iter_condition;
            } else {
              return cond_value;
            }
          } else {
            return true;
          }
        };

        // Allocate scan_outputs for zero-iteration case
        for (int i = 0; i < num_scan_outputs; ++i) {
          Output(i + num_loop_carried_deps)->Resize(0);
          Output(i + num_loop_carried_deps)->template mutable_data<int32_t>();
        }

        // Use this to keep track of the sizes of the scan outputs and validate
        // they're the same across iterations.
        std::vector<std::vector<int64_t>> scan_outputs_sizes;

        Workspace* cur_ws = nullptr;
        bool cur_output_condition = false;

        while (true) {
          int64_t itr = scope_->iteration();
          if (valid_iter_num(itr) && condition_true(itr, cur_output_condition)) {
            if (!scope_->net()->Run()) {
              return false;
            }

            cur_ws = scope_->workspace();
            cur_output_condition = scope_->template output_condition<CondVarType>();
            if (save_scopes_) {
              loop_ws = ws_stack_.pushForwardWorkspace(parent_ws_).get();
              scope_ = std::make_shared<LocalScope>(
                  loop_ws, body_net_def_, num_loop_carried_deps);
            }

            // Copy forward loop-carried dependencies
            for (int i = 0; i < num_loop_carried_deps; ++i) {
              Blob* b = cur_ws->GetBlob(scope_->net()->external_output()[i + 1]);
              const Tensor& t = b->template Get<Tensor>();
              scope_->lcd_tensor(i)->CopyFrom(t);
            }
            // Copy out scan_outputs
            for (int i = 0; i < num_scan_outputs; ++i) {
              int net_output_idx = i + 1 + num_loop_carried_deps;
              const Tensor& scan_output =
                  cur_ws->GetBlob(scope_->net()->external_output()[net_output_idx])
                      ->template Get<Tensor>();
              auto* scan_output_target = Output(i + num_loop_carried_deps);
              if (itr == 0) {
                auto dims = scan_output.sizes().vec();
                scan_outputs_sizes.push_back(dims);
                dims.insert(dims.begin(), 1);
                scan_output_target->Resize(dims);
                scan_output_target->CopyFrom(scan_output);
              } else {
                auto dims = scan_output.sizes().vec();
                CAFFE_ENFORCE_EQ(
                    dims,
                    scan_outputs_sizes[i],
                    "Size of scan output changed across iterations");
                dims.insert(dims.begin(), itr);
                scan_output_target->Extend(1, 100);

                int64_t timestep_size = 1;
                for (const int64_t t : scan_outputs_sizes[i]) {
                  timestep_size *= t;
                }

                const void* src_data = scan_output.raw_data();
                auto& sot_meta = scan_output_target->dtype();
                void* dst_data =
                    (char*)scan_output_target->raw_mutable_data(sot_meta) +
                    timestep_size * scan_output.itemsize() * itr;
                memcpy(dst_data, src_data, timestep_size * scan_output.itemsize());
              }
            }
            scope_->set_iteration(itr + 1ll);
            scope_->template set_input_condition<CondVarType>(cur_output_condition);
          } else {
            break;
          }
        }

        // Copy out final loop-carried dependencies
        for (int i = 0; i < num_loop_carried_deps; ++i) {
          Output(i)->CopyFrom(*scope_->lcd_tensor(i));
        }

        return true;
        */
    }
}

register_cpu_operator!{ONNXWhile, ONNXWhileOp<CPUContext>}
