crate::ix!();

pub struct Param {
    param:               String,
    grad:                String,
    cell_gradient:       String,
}

pub struct RecurrentInput {
    state:               String,
    input:               String,
}

pub struct RecurrentGradient {
    param:               String,
    grad:                String,
    external_grad:       String,
    last_external_grad:  String,
    offset:              i32,
}

pub struct OffsetAlias {
    src:                 String,
    dst:                 String,
    offset:              i32, //{0};
}

pub struct Link {
    internal:            String,
    external:            String,
    offset:              i32,//{0};
    window:              i32,//{1};
}

pub struct ScratchWorkspaces {
    step_workspaces:     Vec<Arc<Workspace>>,
    shared_blobs_ws:     Arc<Workspace>,
}

#[inline] pub fn update_timestep_blob(
    ws:        *mut Workspace,
    blob_name: String,
    t:         i32)  
{
    todo!();
    /*
        BlobGetMutableTensor(ws->CreateBlob(blob_name), CPU)->Resize(1);
      auto timestepBlob = ws->GetBlob(blob_name);
      CAFFE_ENFORCE(timestepBlob);
      BlobGetMutableTensor(timestepBlob, CPU)->template mutable_data<int32_t>()[0] =
          t;
    */
}

#[inline] pub fn apply_offset_alias<T, Context>(
    oc:      &OffsetAlias,
    ws:      *mut Workspace,
    context: *mut Context) 
{
    todo!();
    /*
        VLOG(1) << "Aliasing: " << oc.src << " to: " << oc.dst
              << " at offset: " << oc.offset;
      auto srcBlob = ws->GetBlob(oc.src);
      CAFFE_ENFORCE(srcBlob);
      auto* src = BlobGetMutableTensor(srcBlob, Context::GetDeviceType());
      auto* dst =
          BlobGetMutableTensor(ws->GetBlob(oc.dst), Context::GetDeviceType());
      auto timestep = src->numel() / src->size(0);
      auto dims = src->sizes().vec();
      const int32_t startDstTimestep =
          oc.offset >= 0 ? oc.offset : src->size(0) + oc.offset;
      const int32_t numDstTimesteps = src->size(0) - startDstTimestep;
      if (numDstTimesteps >= 1) {
        dims[0] = numDstTimesteps;
        dst->Resize(dims);
        CAFFE_ENFORCE(timestep == dst->numel() / numDstTimesteps, "Invalid offset");
        dst->ShareExternalPointer(
            src->template mutable_data<T>() + startDstTimestep * timestep);
      } else {
        CAFFE_ENFORCE_EQ(
            numDstTimesteps, 0, "Invalid number of timesteps: ", numDstTimesteps);
        dims[0] = 0;
        dst->Resize(dims);
        dst->template mutable_data<T>();
      }
    */
}

#[inline] pub fn repeat_copy<T, Context>(
    repeat_n: usize,
    n:        usize,
    src:      *const T,
    dst:      *mut T,
    context:  *mut Context) 
{
    todo!();
    /*
        for (int i = 0; i < repeat_n; ++i) {
        context->template CopySameDevice<T>(n, src, dst + i * n);
      }
    */
}

/**
  | Copy external input to the step net into
  | the first item of (T + 1) X batch_size
  | X input_size tensor
  |
  */
#[inline] pub fn initialize_recurrent_input<T, Context>(
    rc:         &RecurrentInput,
    seq_len:    i32,
    batch_size: i32,
    ws:         *mut Workspace,
    context:    *mut Context) 
{
    todo!();
    /*
        auto stateBlob = ws->GetBlob(rc.state);
      CAFFE_ENFORCE(stateBlob);
      auto* state = BlobGetMutableTensor(stateBlob, Context::GetDeviceType());

      auto inputBlob = ws->GetBlob(rc.input);
      CAFFE_ENFORCE(inputBlob);
      const auto& input = inputBlob->template Get<Tensor>();
      CAFFE_ENFORCE_GE(input.dim(), 1, rc.input);
      CAFFE_ENFORCE_LE(input.dim(), 3, rc.input);

      const auto stateSize = input.size(input.dim() - 1);
      // Sometimes we want to provide more than one initial step.
      // For example, if we do a convolution op in step net
      // and need a sufficient left padding around the input.
      // This could be used together with links where window != 1.
      auto initialStateLength = 1;
      if (input.dim() == 3) {
        initialStateLength = input.size(0);
      }
      // States at [0, ..., (T + initialStateLength - 1)] (inclusive)
      state->Resize(seqLen + initialStateLength, batchSize, stateSize);

      if (input.dim() >= 2) {
        CAFFE_ENFORCE_EQ(input.size(input.dim() - 2), batchSize, rc.input);
        context->template CopySameDevice<T>(
            batchSize * stateSize * initialStateLength,
            input.template data<T>(),
            state->template mutable_data<T>());
      } else {
        // Usually, the initial state is the same for all inputs in the batch.
        // So the op conveniently accepts 1-D input and copies it batchSize times.
        repeatCopy<T, Context>(
              batchSize,
              stateSize,
              input.template data<T>(),
              state->template mutable_data<T>(),
              context);
      }
    */
}

/**
  | Run the input network in a recurrent
  | fashion. This can be used to implement
  | fairly general recurrent neural networks
  | (RNNs).
  | 
  | The operator proceeds as follows.
  | 
  | - First, initialized the states from
  | the input recurrent states
  | 
  | - For each timestep T, apply the links
  | (that map offsets from input/output
  | tensors into the inputs/outputs for
  | the `step` network)
  | 
  | - Finally, alias the recurrent states
  | to the specified output blobs.
  | 
  | This is a fairly special-case meta-operator,
  | and so the implementation is somewhat
  | complex. It trades of generality (and
  | frankly usability) against performance
  | and control (compared to e.g. TF dynamic_rnn,
  | Theano scan, etc).
  | 
  | See the usage examples for a flavor of
  | how to use it.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct RecurrentNetworkOp<Context> {
    storage:             OperatorStorage,
    context:             Context,
    step_net_def:        NetDef,
    shared_ws:           *mut Workspace,
    enable_rnn_executor: bool,
    rnn_executor:        Box<RecurrentNetworkExecutorBase>,
    links:               Vec<Link>,
    aliases:             Vec<OffsetAlias>,
    recurrent_inputs:    Vec<RecurrentInput>,
    timestep:            String,
    operator_def:        OperatorDef,
}

num_inputs!{RecurrentNetwork, (1,INT_MAX)}

num_outputs!{RecurrentNetwork, (2,INT_MAX)}

impl<Context> RecurrentNetworkOp<Context> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator<Context>(operator_def, ws),
            sharedWs_(ws),
            enable_rnn_executor_(this->template GetSingleArgument<bool>( "enable_rnn_executor", false)),
            timestep_(this->template GetSingleArgument<std::string>( "timestep", "timestep")),
            operator_def_(operator_def) 

        CAFFE_ENFORCE(ws);

        stepNetDef_ = detail::extractNetDef(operator_def, "step_net");

        recurrentInputs_ = constructRecurrentInputs(operator_def, sharedWs_);
        links_ = constructLinks();
        aliases_ = constructAliases();

        stepNetDef_.add_external_input(timestep_);
        detail::AddApplyLinkOps(
            links_, timestep_, operator_def.device_option(), &stepNetDef_);

        if (FLAGS_caffe2_rnn_executor && enable_rnn_executor_) {
          InitializeExecutor(operator_def);
        }
        */
    }
    
    #[inline] pub fn num_observers(&mut self) -> usize {
        
        todo!();
        /*
            size_t num = this->observers_list_.size();
        if (rnnExecutor_) {
          num += rnnExecutor_->NumObserversStepNet();
        }
        return num;
        */
    }
    
    #[inline] pub fn construct_recurrent_inputs(
        &mut self, 
        operator_def: &OperatorDef,
        shared_ws: *mut Workspace) -> Vec<RecurrentInput> 
    {
        todo!();
        /*
            const auto states =
            this->template GetRepeatedArgument<std::string>("recurrent_states");
        const auto inputs =
            this->template GetRepeatedArgument<int>("initial_recurrent_state_ids");
        CAFFE_ENFORCE_EQ(states.size(), inputs.size(), "states/inputs mismatch");
        std::vector<detail::RecurrentInput> ris;
        for (auto i = 0; i < states.size(); ++i) {
          // States need to be "global" (since they are shared between
          // forward and backward).
          sharedWs->CreateBlob(states[i]);

          detail::RecurrentInput ri;
          ri.state = states[i];
          ri.input = operator_def.input(inputs[i]);
          ris.push_back(ri);
        }
        return ris;
        */
    }
    
    #[inline] pub fn construct_aliases(&mut self) -> Vec<OffsetAlias> {
        
        todo!();
        /*
            const auto& src =
            this->template GetRepeatedArgument<std::string>("alias_src");
        const auto& dst =
            this->template GetRepeatedArgument<std::string>("alias_dst");
        const auto& offset =
            this->template GetRepeatedArgument<int32_t>("alias_offset");
        CAFFE_ENFORCE(
            src.size() == offset.size(), "alias_src/alias_offset mismatch");
        CAFFE_ENFORCE(
            dst.size() == offset.size(), "alias_dst/alias_offset mismatch");
        std::vector<detail::OffsetAlias> aliases;
        for (auto i = 0; i < src.size(); ++i) {
          detail::OffsetAlias oc;
          oc.src = src[i];
          oc.dst = dst[i];
          oc.offset = offset[i];
          aliases.push_back(oc);
        }
        return aliases;
        */
    }
    
    /**
      | Some blobs can be marked as to be recomputed
      | on backward pass.
      | 
      | For those blobs, we do not want to allocate
      | on each step workspace, but we instead
      | store that blob in the shared workspace
      | so all steps can use the same buffer on
      | forward pass.
      |
      */
    #[inline] pub fn initialize_blobs_to_recompute_on_backward(&mut self, shared_blobs_ws: *mut Workspace)  {

        todo!();
        /*
            std::vector<std::string> v;
        const auto& blobs = this->template GetRepeatedArgument<std::string>(
            "recompute_blobs_on_backward", v);
        for (const auto& b : blobs) {
          // Note: if the blob already was created, this is a no-op.
          sharedBlobsWs->CreateBlob(b);
        }
        */
    }
    
    #[inline] pub fn construct_links(&mut self) -> Vec<Link> {
        
        todo!();
        /*
            std::vector<detail::Link> links;
        detail::extractLinks(
            this,
            "link_internal",
            "link_external",
            "link_offset",
            "link_window",
            &links);
        return links;
        */
    }

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            const auto seqLen = Input(0).dim32(0);
            const auto batchSize = Input(0).dim32(1);
            for (const auto& ri : recurrentInputs_) {
              detail::initializeRecurrentInput<T, Context>(
                  ri, seqLen, batchSize, sharedWs_, &context_);
            }

            // If we don't have a backward step net, this operator is forward_only
            // and we can avoid creating multiple workspaces.
            bool has_backward_pass =
                this->template HasSingleArgumentOfType<NetDef>("backward_step_net") ||
                (this->template HasSingleArgumentOfType<string>("backward_step_net") &&
                 this->template GetSingleArgument<string>("backward_step_net", "") !=
                     "");

            // With backward pass: we need to create workspace for each timestep
            detail::ScratchWorkspaces* scratch =
                OperatorStorage::Output<detail::ScratchWorkspaces>(OutputSize() - 1);
            std::vector<std::shared_ptr<Workspace>>& stepWorkspaces =
                scratch->stepWorkspaces;
            std::shared_ptr<Workspace>& sharedBlobsWs = scratch->sharedBlobsWs;
            if (!sharedBlobsWs) {
              sharedBlobsWs = std::make_shared<Workspace>(sharedWs_);
            }

            // Caller can decide that some of the forward activations
            // are recomputed on backward pass. Then those activations do not
            // have to be stored in step workspaces but can be shared.
            initializeBlobsToRecomputeOnBackward(sharedBlobsWs.get());

            if (has_backward_pass && seqLen > stepWorkspaces.size()) {
              stepWorkspaces.resize(seqLen);
            }

            // In forward-only mode, we cycle over workspaces. This limits the amount
            // of parallelism over timesteps that the RNNExecutor provides. So with
            // RNN executor we use more workspaces to get better perf.
            int num_workspaces_on_fwd_only = rnnExecutor_ ? 4 : 2;
            num_workspaces_on_fwd_only = this->template GetSingleArgument<int>(
                "num_workspaces", num_workspaces_on_fwd_only);

            if (!has_backward_pass && stepWorkspaces.size() < num_workspaces_on_fwd_only) {
              // Use alternating stepWorkspaces when forward_only=True.
              // Note that the step workspaces can be shared by other ops, thus
              // we cannot shrink it to 2 if there are more than 2 step workspaces.
              stepWorkspaces.resize(num_workspaces_on_fwd_only);
            }

            for (auto t = 0; t < seqLen; ++t) {
              auto& currentStepWorkspace =
                  (has_backward_pass ? stepWorkspaces[t] :
                      stepWorkspaces[t % num_workspaces_on_fwd_only]);
              if (!currentStepWorkspace) {
                currentStepWorkspace = std::make_shared<Workspace>(sharedBlobsWs.get());
              }

              if (rnnExecutor_) {
                if (!has_backward_pass) {
                  // Need to limit timestep parallelism because we cycle over workspaces
                  rnnExecutor_->SetMaxParallelTimesteps(num_workspaces_on_fwd_only);
                }
                rnnExecutor_->EnsureTimestepInitialized(
                    t, currentStepWorkspace.get(), this->observers_list_);
              } else {
                // Use plain Caffe2 nets
                detail::UpdateTimestepBlob(currentStepWorkspace.get(), timestep_, t);
                auto* stepNet = currentStepWorkspace->GetNet(stepNetDef_.name());
                if (stepNet == nullptr) {
                  stepNet = currentStepWorkspace->CreateNet(stepNetDef_);
                }
                CAFFE_ENFORCE(stepNet, "Step Net construction failure");
                // Since we have a SimpleNet, there are no races here.
                stepNet->RunAsync();
              }
            }

            if (rnnExecutor_) {
              try {
                rnnExecutor_->Run(seqLen);
              } catch (const std::exception& e) {
                LOG(ERROR) << "Encountered exception in RNN executor: " << e.what();
                InitializeExecutor(operator_def_);
                return false;
              } catch (...) {
                LOG(ERROR) << "Encountered exception in RNN executor: unknown";
                InitializeExecutor(operator_def_);
                return false;
              }
            }

            for (const auto& alias : aliases_) {
              detail::applyOffsetAlias<T, Context>(alias, sharedWs_, &context_);
            }

            return true;
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DoRunWithType<float>();
        */
    }
    
    #[inline] pub fn initialize_executor(&mut self, operator_def: &OperatorDef)  {
        
        todo!();
        /*
            VLOG(1) << "Use RecurrentNetworkExecutor";
        auto recurrent_map =
            detail::GetRecurrentMapping(links_, false /* backward */);
        rnnExecutor_ = createRNNExecutor<Context>(
            stepNetDef_, recurrent_map, timestep_, ArgumentHelper(operator_def));
        */
    }
}

///--------------------------------
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct RecurrentNetworkGradientOp<Context> {
    storage:               OperatorStorage,
    context:               Context,
    step_net_def:          NetDef,
    shared_ws:             *mut Workspace,
    enable_rnn_executor:   bool,
    rnn_executor:          Box<RecurrentNetworkExecutorBase>,
    links:                 Vec<Link>,
    params:                Vec<Param>,
    recurrent_gradients:   Vec<RecurrentGradient>,
    timestep:              String,

    /// For now we support only one input sequence
    num_sequences:         i32, //{1};
    recurrent_input_ids:   Vec<i32>,
    grad_inputs:           Vec<i32>,
}

impl<Context> RecurrentNetworkGradientOp<Context> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator<Context>(operator_def, ws),
            sharedWs_(ws),
            enable_rnn_executor_(this->template GetSingleArgument<bool>( "enable_rnn_executor", false)),
            timestep_(this->template GetSingleArgument<std::string>( "timestep", "timestep")),
            gradInputs_( this->template GetRepeatedArgument<int32_t>("outputs_with_grads")) 

        CAFFE_ENFORCE(ws);

        stepNetDef_ = detail::extractNetDef(operator_def, "backward_step_net");

        links_ = constructLinks();
        params_ = constructParams(operator_def);
        recurrentGradients_ = constructRecurrentGradients(operator_def);
        recurrentInputIds_ = this->template GetRepeatedArgument<int32_t>(
            "initial_recurrent_state_ids");

        /* Add operators to the backward step net to handle accumulation of
           gradients over timesteps
        */
        stepNetDef_.add_external_input(timestep_);

        AddGradientInputAccumulationOps(operator_def);
        detail::AddApplyLinkOps(
            links_, timestep_, operator_def.device_option(), &stepNetDef_);
        AddParamGradientAccumulationOps(operator_def);

        if (FLAGS_caffe2_rnn_executor && enable_rnn_executor_) {
          InitializeExecutor(operator_def);
        }
        */
    }

    /// Renaming maps (generated by memonger.py)
    #[inline] pub fn remapped_name(&mut self, blob_name: String) -> String {
        
        todo!();
        /*
            return this->template GetSingleArgument<std::string>(
            blob_name + ".rename", blob_name);
        */
    }
    
    #[inline] pub fn remapped_link(&mut self, link: &Link) -> Link {
        
        todo!();
        /*
            detail::Link renamed_link = link;
        renamed_link.internal = remappedName(link.internal);
        renamed_link.external = remappedName(link.external);
        return renamed_link;
        */
    }
    
    #[inline] pub fn rename_op_input_output(
        &mut self, 
        from_name: String,
        to_name:   String)  
    {
        todo!();
        /*
            for (int j = 0; j < stepNetDef_.op_size(); j++) {
          auto* op = stepNetDef_.mutable_op(j);
          for (int i = 0; i < op->input_size(); i++) {
            if (op->input(i) == from_name) {
              op->set_input(i, to_name);
            }
          }
          for (int i = 0; i < op->output_size(); i++) {
            if (op->output(i) == from_name) {
              op->set_output(i, to_name);
            }
          }
        }
        */
    }
    
    #[inline] pub fn construct_params(&mut self, operator_def: &OperatorDef) -> Vec<Param> {
        
        todo!();
        /*
            std::vector<detail::Param> params;
        const auto& param = this->template GetRepeatedArgument<int32_t>("param");
        const auto& param_grads =
            this->template GetRepeatedArgument<string>("param_grads");
        CAFFE_ENFORCE(
            param_grads.empty() || param_grads.size() == param.size(),
            param.size(),
            " != ",
            param_grads.size());
        for (int i = 0; i < param.size(); ++i) {
          detail::Param p;
          // Forward inputs come after [outputs_with_grads] gradient inputs
          p.param = operator_def.input(param[i] + gradInputs_.size());
          // See GetRecurrentNetworkGradient to understand offseting here
          p.grad = operator_def.output(i + numSequences_);

          std::string grad_blob =
              param_grads.empty() ? p.grad : remappedName(param_grads[i]);
          p.cellGradient = grad_blob + "_tmpstep";
          params.push_back(p);

          renameOpInputOutput(grad_blob, p.cellGradient);
        }
        return params;
        */
    }
    
    #[inline] pub fn construct_recurrent_gradients(&mut self, operator_def: &OperatorDef) -> Vec<RecurrentGradient> {
        
        todo!();
        /*
            std::vector<detail::RecurrentGradient> rgs;
        const auto& recurrent =
            this->template GetRepeatedArgument<std::string>("recurrent_states");
        const auto& alias_src =
            this->template GetRepeatedArgument<std::string>("alias_src");
        const auto& offset =
            this->template GetRepeatedArgument<int32_t>("alias_offset");

        for (auto i = 0; i < recurrent.size(); ++i) {
          detail::RecurrentGradient rg;
          rg.param = recurrent[i];
          rg.grad = remappedName(recurrent[i] + "_grad");

          for (int j = 0; j < alias_src.size(); ++j) {
            if (alias_src[j] != recurrent[i]) {
              continue;
            }
            int idx = -1;
            for (int k = 0; k < gradInputs_.size(); ++k) {
              if (gradInputs_[k] == j) {
                idx = k;
              }
            }
            if (idx == -1) {
              continue;
            }

            CAFFE_ENFORCE(offset[j] == 1 || offset[j] == -1);
            if (offset[j] == 1) {
              rg.externalGrad = operator_def.input(idx);
            } else if (offset[j] == -1) {
              rg.lastExternalGrad = operator_def.input(idx);
            }
          }
          rg.offset = 1;
          rgs.push_back(rg);
        }
        return rgs;
        */
    }
    
    #[inline] pub fn construct_links(&mut self) -> Vec<Link> {
        
        todo!();
        /*
            std::vector<detail::Link> links;
        detail::extractLinks(
            this,
            "link_internal",
            "link_external",
            "link_offset",
            "link_window",
            &links);
        detail::extractLinks(
            this,
            "backward_link_internal",
            "backward_link_external",
            "backward_link_offset",
            "",
            &links);
        for (int i = 0; i < links.size(); i++) {
          links[i] = remappedLink(links[i]);
        }
        return links;
        */
    }
    
    #[inline] pub fn initialize_executor(&mut self, operator_def: &OperatorDef)  {
        
        todo!();
        /*
            VLOG(1) << "Use RecurrentNetworkExecutor for backward";
        auto recurrent_map = detail::GetRecurrentMapping(links_, true /* backward */);
        rnnExecutor_ = createRNNExecutor<Context>(
          stepNetDef_, recurrent_map, timestep_, ArgumentHelper(operator_def));
        */
    }
    
    #[inline] pub fn add_gradient_input_accumulation_ops(&mut self, operator_def: &OperatorDef)  {
        
        todo!();
        /*
            /**
          * Add ops to the step net to accumulate input gradients.
          */
        std::vector<OperatorDef> ops;
        for (const auto& rg : recurrentGradients_) {
          if (rg.externalGrad.empty()) {
            continue;
          }
          VLOG(1) << "Accumulating into: " << rg.grad << " from " << rg.externalGrad
                  << ", offset: " << rg.offset;

          OperatorDef opdef;
          opdef.set_type("rnn_internal_accumulate_gradient_input");
          opdef.add_input(timestep_);
          opdef.add_input(rg.externalGrad);
          opdef.add_input(rg.grad);
          opdef.add_output(rg.grad);

          // Add also the linked blobs to outputs, to ensure correct
          // chaining.
          for (auto& l : links_) {
            if (rg.grad == l.external) {
              Argument* dep_arg = opdef.add_arg();
              dep_arg->set_name("rnn_dependency." + l.internal);
              dep_arg->set_s(l.internal);
            }
          }

          opdef.mutable_device_option()->CopyFrom(operator_def.device_option());

          Argument* offset_arg = opdef.add_arg();
          offset_arg->set_name("offset");
          offset_arg->set_i(rg.offset);
          ops.push_back(opdef);

          stepNetDef_.add_external_input(rg.externalGrad);
          stepNetDef_.add_external_input(rg.grad);
        }
        detail::PrependOps(ops, &stepNetDef_);
        */
    }
    
    #[inline] pub fn add_param_gradient_accumulation_ops(&mut self, operator_def: &OperatorDef)  {
        
        todo!();
        /*
            // If a user passes in param_grads mapping, we can copy dirrectly
        // form a blob where backward cell net written data to.
        // This becomes handy in a case where gradient from the cell net
        // is an internal blob of the backward cell. This happens, for example,
        // when SumOp is the first op of the cell
        for (const auto& param : params_) {
          OperatorDef opdef;
          opdef.set_type("Sum");
          opdef.add_input(param.grad);
          opdef.add_input(param.cellGradient);
          opdef.add_output(param.grad);
          opdef.mutable_device_option()->CopyFrom(operator_def.device_option());
          stepNetDef_.add_op()->CopyFrom(opdef);
          stepNetDef_.add_external_input(param.grad);
        }
        */
    }
    
    #[inline] pub fn create_shared_blobs(
        &mut self, 
        step_0ws: &Arc<Workspace>,
        shared_blobs_ws: *mut Workspace)  
    {
        todo!();
        /*
            /**
          * Create all output blobs created by ops of the backward step net, they
          * can be shared.
          */
        for (auto& op : stepNetDef_.op()) {
          for (const string& outp : op.output()) {
            if (!step0Ws->HasBlob(outp)) {
              sharedBlobsWs->CreateBlob(outp);
            }
          }
        }
        */
    }

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            const auto seqLen = Input(gradInputs_.size()).dim32(0);
            VLOG(1) << "seqLen: " << seqLen;

            const detail::ScratchWorkspaces& scratch =
                this->template Input<detail::ScratchWorkspaces>(InputSize() - 1);
            const std::vector<std::shared_ptr<Workspace>>& stepWorkspaces =
                scratch.stepWorkspaces;
            CAFFE_ENFORCE_GE(stepWorkspaces.size(), seqLen);
            Workspace& sharedBlobsWs = *scratch.sharedBlobsWs.get();

            const auto batchSize = Input(0).dim32(1);
            for (auto& param : params_) {
              auto pBlob = sharedWs_->GetBlob(param.param);
              CAFFE_ENFORCE(pBlob);
              const auto& p = pBlob->template Get<Tensor>();

              auto gBlob = sharedWs_->GetBlob(param.grad);
              CAFFE_ENFORCE(gBlob);
              auto* g = BlobGetMutableTensor(gBlob, Context::GetDeviceType());
              g->ResizeLike(p);
              math::Set<T, Context>(
                  g->numel(),
                  convert::To<float, T>(0.0),
                  g->template mutable_data<T>(),
                  &context_);
            }

            for (auto& rg : recurrentGradients_) {
              auto pBlob = sharedWs_->GetBlob(rg.param);
              CAFFE_ENFORCE(pBlob);
              const auto& p = pBlob->template Get<Tensor>();

              auto gBlob = sharedWs_->CreateBlob(rg.grad);
              CAFFE_ENFORCE(gBlob);
              auto* g = BlobGetMutableTensor(gBlob, Context::GetDeviceType());
              g->ResizeLike(p);
              CAFFE_ENFORCE_EQ(g->dim(), 3);
              const auto timestep = g->numel() / g->size(0);
              // Fill the last timestep with zeros for the gradient
              math::Set<T, Context>(
                  timestep,
                  convert::To<float, T>(0.0),
                  g->template mutable_data<T>() + (g->size(0) - 1) * timestep,
                  &context_);
            }

            // This code assumes that there are several inputs
            // sequences. Actually it is not supported by the rest of the code,
            // and numSequences_ is a constant, equal to 1.
            for (int i = 0; i < numSequences_; ++i) {
              // Offseting as the first gradInputs_.size() inputs of the op
              // are from GO. Then all I(0..N).
              const int gradientInputIndex = i + gradInputs_.size();
              const auto& inputName = this->debug_def().input(gradientInputIndex);
              auto gradientName = remappedName(inputName + "_grad");
              VLOG(1) << "Initializing gradient for input " << gradientInputIndex
                      << " (" << inputName << ") "
                      << " as blob " << gradientName
                      << ". Size: " << Input(gradientInputIndex).numel();
              auto pGradientBlob = sharedWs_->GetBlob(gradientName);
              CAFFE_ENFORCE(pGradientBlob);
              auto* g = BlobGetMutableTensor(pGradientBlob, Context::GetDeviceType());
              g->ResizeLike(Input(gradientInputIndex));
              g->template mutable_data<T>();
            }

            auto accumulateFinalInputGradients = [&]() {
              for (const auto& rg : recurrentGradients_) {
                if (rg.lastExternalGrad.empty()) {
                  continue;
                }
                VLOG(1) << "Accumulating into: " << rg.grad << " from "
                        << rg.lastExternalGrad << " for final time step (sep. blob)";
                auto gBlob = sharedWs_->GetBlob(rg.grad);
                CAFFE_ENFORCE(gBlob);
                auto* g = BlobGetMutableTensor(gBlob, Context::GetDeviceType());

                auto oglastBlob = sharedWs_->GetBlob(rg.lastExternalGrad);
                CAFFE_ENFORCE(oglastBlob);
                const auto& oglast = oglastBlob->template Get<Tensor>();
                CAFFE_ENFORCE_EQ(g->size(1), oglast.size(1));
                CAFFE_ENFORCE_EQ(g->size(2), oglast.size(2));

                const auto t = g->size(0) - 1;
                const auto timestep_size = g->numel() / g->size(0);
                CAFFE_ENFORCE_EQ(timestep_size, oglast.numel());
                T* g_data_with_offset =
                    g->template mutable_data<T>() + t * timestep_size;
                math::Add<T, Context>(
                    timestep_size,
                    oglast.template data<T>(),
                    g_data_with_offset,
                    g_data_with_offset,
                    &context_);
              }
            };

            accumulateFinalInputGradients();

            // Create shared blobs for blobs that can be shared between
            // all timesteps.
            if (stepWorkspaces.size() > 0) {
              CreateSharedBlobs(stepWorkspaces[0], &sharedBlobsWs);
            }
            for (int32_t t = seqLen - 1; t >= 0; --t) {
              if (rnnExecutor_) {
                rnnExecutor_->EnsureTimestepInitialized(
                    t, stepWorkspaces[t].get(), this->observers_list_);
              } else {
                auto* stepNet = stepWorkspaces[t].get()->GetNet(stepNetDef_.name());
                if (stepNet == nullptr) {
                  stepNet = stepWorkspaces[t].get()->CreateNet(stepNetDef_);
                }
                CAFFE_ENFORCE(stepNet);
                stepNet->RunAsync();
              }
            }

            if (rnnExecutor_) {
              rnnExecutor_->RunBackwards(seqLen);
            }

            CAFFE_ENFORCE_EQ(recurrentInputIds_.size(), recurrentGradients_.size());
            for (int i = 0; i < recurrentInputIds_.size(); ++i) {
              // See GetRecurrentNetworkGradient to understand offseting here
              // Outputs of the gradient are inputs of the forward pass.
              // So we need to offset on all inputs that go before recurrent
              // initial ones
              auto outputIdx = i + params_.size() + numSequences_;
              // because first gradInputs_.size() inputs are from GO
              int inputId = recurrentInputIds_[i] + gradInputs_.size();
              VLOG(1) << "Resetting output " << this->debug_def().output(outputIdx)
                      << " like input " << this->debug_def().input(inputId);
              Output(outputIdx)->ResizeLike(Input(inputId));
              T* output_data = Output(outputIdx)->template mutable_data<T>();
              auto pBlob = sharedWs_->GetBlob(recurrentGradients_[i].grad);
              CAFFE_ENFORCE(pBlob);
              auto* p = BlobGetMutableTensor(pBlob, Context::GetDeviceType());

              if (Input(inputId).dim() >= 2) {
                // Gradient states blob should live. And if it gets changed by the
                // backward pass, then output should be changed as well. Thus it should
                // be okay to share data here
                Output(outputIdx)->template ShareExternalPointer<T>(
                    p->template mutable_data<T>());
              } else {
                // We need to do a bunch of Adds any way. So lets not worry about
                // copy / share data here. One way to speed this up could be a kernel
                // which sums up several tensors together instead of going 1 by 1
                const auto recurrentStateSize = Input(inputId).dim32(0);

                math::Set<T, Context>(
                    recurrentStateSize,
                    convert::To<float,T>(0.0),
                    output_data,
                    &context_);

                math::AddStripedBatch<T, Context>(
                    recurrentStateSize,
                    p->template data<T>(),
                    output_data,
                    recurrentStateSize,
                    batchSize,
                    &context_);
              }
            }

            return true;
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DoRunWithType<float>();
        */
    }
}

///-------------------------------------
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct AccumulateInputGradientOp<Context> {
    storage: OperatorStorage,
    context: Context,
    offset:  i32,
}

impl<Context> AccumulateInputGradientOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            offset_(this->template GetSingleArgument<int>("offset", -1)) 

        CAFFE_ENFORCE(offset_ >= 0, "Offset not set");
        */
    }

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            const auto& t0 = this->template Input<Tensor>(0, CPU);
            const auto t = t0.template data<int32_t>()[0];
            auto& og = Input(1);
            auto* g = Output(0);

            T* g_data = g->template mutable_data<T>();
            const auto timestep_size = g->numel() / g->size(0);

            CAFFE_ENFORCE(
                (t + offset_) * timestep_size + timestep_size <= g->numel(),
                "Accumulation destination address over bounds");
            CAFFE_ENFORCE(
                t * timestep_size + timestep_size <= og.numel(),
                "Accumulation source address out of bounds");

            math::Add<T, Context>(
                timestep_size,
                og.template data<T>() + t * timestep_size,
                g_data + (t + offset_) * timestep_size,
                g_data + (t + offset_) * timestep_size,
                &context_);
            return true;
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<float>>::call(this, Input(1));
        */
    }
}

///-------------------------------
pub struct RNNApplyLinkOp<Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,

    offset: i32,
    window: i32,
}

impl<Context> RNNApplyLinkOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            offset_(this->template GetSingleArgument<int>("offset", -1)),
            window_(this->template GetSingleArgument<int>("window", -1)) 

        CAFFE_ENFORCE(offset_ >= 0, "offset not set");
        CAFFE_ENFORCE(window_ >= 0, "window not set");
        */
    }

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            // Both internal and external appear as both input and output to enforce
            // correct dependency computation.
            const auto& t0 = this->template Input<Tensor>(0, CPU);
            const auto t = t0.template data<int32_t>()[0];
            auto& external = Input(1);

            auto* internal_out = Output(0);
            auto* external_out = Output(1);

            CAFFE_ENFORCE_GT(external.numel(), 0);
            const int64_t externalTimestepSize = external.numel() / external.size(0);
            auto* externalData = external_out->template mutable_data<T>() +
                (t + offset_) * externalTimestepSize;
            auto internalDims = external_out->sizes().vec();
            internalDims[0] = window_;

            internal_out->Resize(internalDims);
            internal_out->ShareExternalPointer(
                externalData, externalTimestepSize * window_);
            return true;
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DoRunWithType<float>();
        */
    }
}

caffe_known_type!{detail::ScratchWorkspaces}

register_cpu_operator!{RecurrentNetwork, RecurrentNetworkOp<CPUContext>}

register_cpu_operator!{
    RecurrentNetworkGradient,
    RecurrentNetworkGradientOp<CPUContext>
}

/**
  | Internal RNN operator.
  |
  */
register_cpu_operator!{
    rnn_internal_accumulate_gradient_input,
    AccumulateInputGradientOp<CPUContext>
}

num_inputs!{rnn_internal_accumulate_gradient_input, 3}

num_outputs!{rnn_internal_accumulate_gradient_input, (1,INT_MAX)}

enforce_inplace!{rnn_internal_accumulate_gradient_input, 
    vec![(2, 0)]}

private_operator!{rnn_internal_accumulate_gradient_input}

/**
  | Internal RNN operator.
  |
  */
register_cpu_operator!{
    rnn_internal_apply_link,
    RNNApplyLinkOp<CPUContext>
}

num_inputs!{rnn_internal_apply_link, 2}

num_outputs!{rnn_internal_apply_link, 2}

enforce_inplace!{rnn_internal_apply_link, vec![(1, 1)]}

private_operator!{rnn_internal_apply_link}

pub struct GetRecurrentNetworkGradient<'a> {

    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetRecurrentNetworkGradient<'a> {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            ArgumentHelper argsHelper(def_);
        auto params = argsHelper.GetRepeatedArgument<int32_t>("param");
        auto recurrentInputs =
            argsHelper.GetRepeatedArgument<int32_t>("initial_recurrent_state_ids");

        std::vector<std::string> gradientInputs;

        // Argument specifies which outputs have external gradient, (0) by default
        auto outputs_with_grads =
            argsHelper.GetRepeatedArgument<int32_t>("outputs_with_grads");
        CAFFE_ENFORCE(outputs_with_grads.size() > 0);
        for (auto id : outputs_with_grads) {
          gradientInputs.push_back(GO(id));
        }

        // All inputs and outputs are passed back
        for (int i = 0; i < def_.input_size(); ++i) {
          gradientInputs.push_back(I(i));
        }
        for (int i = 0; i < def_.output_size(); ++i) {
          gradientInputs.push_back(O(i));
        }

        // We calculate gradients only for parameters and recurrent inputs
        std::vector<std::string> gradientOutputs;
        gradientOutputs.push_back(GI(0));
        for (auto id : params) {
          gradientOutputs.push_back(GI(id));
        }
        for (auto id : recurrentInputs) {
          gradientOutputs.push_back(GI(id));
        }

        VLOG(1) << "Gradient blobs: " << c10::Join(", ", gradientOutputs);

        return SingleGradientDef(
            "RecurrentNetworkGradient", "", gradientInputs, gradientOutputs);
        */
    }
}

register_gradient!{RecurrentNetwork, GetRecurrentNetworkGradient}

#[inline] pub fn get_recurrent_mapping(
    links: &Vec<Link>,
    backward: bool) -> HashMap<String,String> 
{
    todo!();
    /*
        std::map<string, string> mappings;
      for (auto it = links.begin(); it != links.end(); ++it) {
        const auto& l1 = *it;

        // In backward op we expect to see offset 1 before offset 0 and
        // vice versa.
        const int offset_l1 = backward ? 1 : 0;
        const int offset_l2 = 1 - offset_l1;
        if (l1.offset == offset_l1) {
          // Find offset = 1 from links. We could probaby rely on order, but
          // since the number of links is links small, O(n^2) algo is ok
          for (auto it2 = it + 1; it2 != links.end(); ++it2) {
            const auto& l2 = *it2;
            if (l2.offset == offset_l2 && l2.external == l1.external) {
              mappings[l2.internal] = l1.internal;
              break;
            }
          }
        }
      }
      return mappings;
    */
}


#[inline] pub fn prepend_ops(
    ops: Vec<OperatorDef>,
    netdef: *mut NetDef)  
{
    todo!();
    /*
        for (auto& o : netdef->op()) {
        ops.push_back(o);
      }
      netdef->mutable_op()->Clear();
      for (auto& o : ops) {
        auto* ao = netdef->add_op();
        ao->CopyFrom(o);
      }
    */
}

#[inline] pub fn add_apply_link_ops(
    links:          &Vec<Link>,
    timestep:       String,
    device_option:  &DeviceOption,
    netdef:         *mut NetDef)  
{
    todo!();
    /*
        std::vector<OperatorDef> ops;
      for (auto& link : links) {
        OperatorDef opdef;
        opdef.set_type("rnn_internal_apply_link");
        opdef.add_input(timestep);
        opdef.add_input(link.external);
        opdef.add_output(link.internal);
        opdef.add_output(link.external);
        opdef.mutable_device_option()->CopyFrom(device_option);

        Argument* offset_arg = opdef.add_arg();
        offset_arg->set_name("offset");
        offset_arg->set_i(link.offset);

        Argument* window_arg = opdef.add_arg();
        window_arg->set_name("window");
        window_arg->set_i(link.window);

        // Find out if the linked blob is used first as an output: then we need
        // to add control_input to that op
        for (auto& op : *netdef->mutable_op()) {
          if (HasInput(op, link.internal)) {
            // First appears as an input, no need to do antyhing
            continue;
          }
          if (HasOutput(op, link.internal)) {
            op.add_control_input(link.internal);
            break;
          }
        }

        ops.push_back(opdef);

        netdef->add_external_input(link.internal);
        netdef->add_external_input(link.external);
      }

      detail::PrependOps(ops, netdef);
    */
}

#[inline] pub fn extract_links(
    op:            *mut OperatorStorage,
    internal_arg:  &String,
    external_arg:  &String,
    offset_arg:    &String,
    window_arg:    &String,
    links:         *mut Vec<Link>)  
{
    todo!();
    /*
        const auto& internal = op->GetRepeatedArgument<std::string>(internalArg);
      const auto& external = op->GetRepeatedArgument<std::string>(externalArg);
      const auto& offset = op->GetRepeatedArgument<int32_t>(offsetArg);
      const auto& window = op->GetRepeatedArgument<int32_t>(
          windowArg, vector<int32_t>(offset.size(), 1));
      CAFFE_ENFORCE_EQ(
          internal.size(),
          offset.size(),
          "internal/offset mismatch: ",
          internalArg,
          " ",
          externalArg);
      CAFFE_ENFORCE_EQ(
          external.size(),
          offset.size(),
          "external/offset mismatch: ",
          externalArg,
          " ",
          offsetArg);
      CAFFE_ENFORCE_EQ(
          external.size(),
          window.size(),
          "external/window mismatch: ",
          externalArg,
          " ",
          windowArg);
      for (auto i = 0; i < internal.size(); ++i) {
        detail::Link l;
        l.internal = internal[i];
        l.external = external[i];
        l.offset = offset[i];
        l.window = window[i];
        links->push_back(l);
      }
    */
}

#[inline] pub fn extract_net_def(
    op:       &OperatorDef,
    arg_name: &String) -> NetDef 
{
    todo!();
    /*
        if (ArgumentHelper::HasSingleArgumentOfType<OperatorDef, NetDef>(
              op, argName)) {
        return ArgumentHelper::GetSingleArgument<OperatorDef, NetDef>(
            op, argName, NetDef());
      } else {
    #ifndef CAFFE2_RNN_NO_TEXT_FORMAT
        NetDef result;
        const auto netString =
            ArgumentHelper::GetSingleArgument<OperatorDef, string>(op, argName, "");
        CAFFE_ENFORCE(
            TextFormat::ParseFromString(netString, &result),
            "Invalid NetDef");
        return result;
    #else
        CAFFE_THROW("No valid NetDef for argument ", argName);
    #endif
      }
    */
}
