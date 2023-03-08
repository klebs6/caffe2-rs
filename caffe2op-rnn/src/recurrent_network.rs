crate::ix!();

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
