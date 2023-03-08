crate::ix!();

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
