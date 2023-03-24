crate::ix!();

pub type ShouldContinue = fn(_u0: i32) -> bool;

define_bool!{
    caffe2_handle_executor_threads_exceptions,
    false,
    "If used we will handle exceptions in executor threads. This avoids SIGABRT but may cause process to deadlock"
}

define_int!{
    caffe2_plan_executor_exception_timeout,
    60,
    "Number of seconds to wait for concurrent threads to stop on exception before terminating."
}

/**
  | ExceptionWrapper holds an exception.
  | If exception pointers are being used,
  | it'll hold the original exception pointer
  | otherwise just the message.
  |
  */
pub struct ExceptionWrapper {
    has_exception:  bool,

    #[cfg(caffe2_use_exception_ptr)]
    exception: c_void,

    exception_msg:  String,
}

impl Default for ExceptionWrapper {
    
    fn default() -> Self {
        todo!();
        /*
            : hasException_(false
        */
    }
}

impl ExceptionWrapper {
    
    pub fn new(ex: &dyn Exception) -> Self {
    
        todo!();
        /*
            : hasException_(true), exceptionMsg_(ex.what()) 

    #ifdef CAFFE2_USE_EXCEPTION_PTR
        exception_ = std::current_exception();
    #endif
        */
    }
    
    #[inline] pub fn rethrow_exception(&mut self)  {
        
        todo!();
        /*
            #ifdef CAFFE2_USE_EXCEPTION_PTR
        std::rethrow_exception(exception_);
    #else
        CAFFE_THROW(exceptionMsg_);
    #endif
        */
    }
    
    #[inline] pub fn what(&self) -> &String {
        
        todo!();
        /*
            return exceptionMsg_;
        */
    }
}

impl Into<bool> for ExceptionWrapper {
    fn into(self) -> bool {
        self.has_exception
    }
}

/**
  | ExceptionWrapperTerminate terminates
  | the program with the specified exception.
  | This preserves the exception ptr and
  | ExceptionTracer will correctly grab
  | it on exit.
  |
  */
pub struct ExceptionWrapperTerminate {
    ew:  ExceptionWrapper,
}

impl ExceptionWrapperTerminate {

    pub fn new(ew: ExceptionWrapper) -> Self {
    
        todo!();
        /*
            : ew_(std::move(ew))
        */
    }
}

impl Drop for ExceptionWrapperTerminate {
    fn drop(&mut self) {
        todo!();
        /* 
        ew_.rethrowException();
       */
    }
}

/**
  | ScopeExitGuard runs the provided function
  | when it's destructed.
  |
  */
pub struct ScopeExitGuard {

    f:  fn() -> (),
}

impl ScopeExitGuard {
    
    pub fn new(f: fn() -> ()) -> Self {
    
        todo!();
        /*
            : f_(std::move(f))
        */
    }
}

impl Drop for ScopeExitGuard {
    fn drop(&mut self) {
        todo!();
        /* 
        f_();
       */
    }
}

///---------------------------------
pub struct NetDefInfo {
    net_def:         *const NetDef,

    /**
      | in order to keep the "override existing
      | nets" on the top-level workflow, we need
      | to mark the nets that already exist so
      | that we can override them exactly once.
      */
    needs_override:  bool,
}

pub type NetDefMap = HashMap<String,NetDefInfo>;

/**
  | Returns a function that returns `true`
  | if we should continue iterating, given
  | the current iteration count.
  |
  */
#[inline] pub fn get_continuation_test(
    ws: *mut Workspace, 
    step: &ExecutionStep) -> fn(_u0: i64) -> bool {
    
    todo!();
    /*
        if (step.has_should_stop_blob()) {
        CAFFE_ENFORCE(
            !step.has_num_iter(),
            "Must not specify num_iter if should_stop_blob is set");
      }

      if (!step.has_should_stop_blob()) { // control by iteration
        CAFFE_ENFORCE(!step.has_only_once(), "not supported");
        int64_t iterations = step.has_num_iter() ? step.num_iter() : 1;
        VLOG(1) << "Will execute step " << step.name() << " for " << iterations
                << " iterations.";
        return [=](int64_t i) { return i < iterations; };
      } else { // control by signal blob
        bool onlyOnce = step.has_only_once() && step.only_once();
        VLOG(1) << "Will execute step" << step.name() << (onlyOnce ? " once " : "")
                << " until stopped by blob " << step.should_stop_blob();
        if (onlyOnce) {
          return [](int64_t i) { return i == 0; };
        } else {
          return [](int64_t /*i*/) { return true; };
        }
      }
    */
}

/**
  | if the blob doesn't exist or is not initialized,
  | return false
  |
  */
#[inline] pub fn get_should_stop(b: *const Blob) -> bool {
    
    todo!();
    /*
        if (!b ||
          b->meta() ==
              ScalarType::Undefined) { // not exist or uninitialized
        return false;
      }

      const auto& t = b->Get<TensorCPU>();
      CAFFE_ENFORCE(t.IsType<bool>() && t.numel() == 1, "expects a scalar boolean");
      return *(t.template data<bool>());
    */
}

/**
  | Injects a blob named 'GLOBAL_WORKSPACE_ID'
  | for each workspace, only if another
  | blob named 'NODE_ID' is present. 'NODE_ID'
  | blob can be used in a distributed run
  | and in this case 'GLOBAL_WORKSPACE_ID'
  | can be used across machines for other
  | purposes (e.g. to support model parallelism).
  | Essentially, 'GLOBAL_WORKSPACE_ID'
  | is an identifier for a workspace that
  | is unique across all 'NODE_ID's.
  |
  */
pub struct WorkspaceIdInjector {
    seq:  Atomic<i32>, // default = 0
}

impl WorkspaceIdInjector {

    const NODE_ID:             &'static str = "NODE_ID";
    const GLOBAL_WORKSPACE_ID: &'static str = "GLOBAL_WORKSPACE_ID";
    
    #[inline] pub fn inject_workspace_id(&mut self, workspace: *mut Workspace)  {
        
        todo!();
        /*
            if (workspace->HasBlob(NODE_ID)) {
          Blob* node_id_blob = workspace->GetBlob(NODE_ID);
          const TensorCPU& node_id_tensor = node_id_blob->template Get<TensorCPU>();
          int node_id = node_id_tensor.template data<int32_t>()[0];
          CAFFE_ENFORCE(
              seq_ < (1 << 16),
              "Integer overflow while calculating GLOBAL_WORKSPACE_ID blob");
          int32_t global_ws_id = (seq_++) + (static_cast<int32_t>(node_id) << 16);
          Blob* global_ws_id_blob = workspace->CreateLocalBlob(GLOBAL_WORKSPACE_ID);
          TensorCPU* global_ws_id_tensor =
              BlobGetMutableTensor(global_ws_id_blob, CPU);
          global_ws_id_tensor->Resize();
          global_ws_id_tensor->template mutable_data<int32_t>()[0] = global_ws_id;
          VLOG(1) << "Adding " << GLOBAL_WORKSPACE_ID << " = " << global_ws_id;
        }
        */
    }
}

///-------------------------------------
pub struct CompiledGuard {
    
    compiled:      Box<CompiledExecutionStep>,
    compiled_ref:  *mut CompiledExecutionStep,
}

impl CompiledGuard {
    
    #[inline] pub fn reset(&mut self, compiled: Box<CompiledExecutionStep>)  {
        
        todo!();
        /*
            compiled_ = std::move(compiled);
            compiledRef_ = compiled_.get();
        */
    }
    
    #[inline] pub fn reset_with_compiled_ref(&mut self, compiled_ref: *mut CompiledExecutionStep)  {
        
        todo!();
        /*
            compiled_.reset();
            compiledRef_ = compiledRef;
        */
    }
}

impl Deref for CompiledGuard {
    type Target = CompiledExecutionStep;
    fn deref(&self) -> &Self::Target {
        unsafe {
            &(*self.compiled_ref)
        }
    }
}

impl DerefMut for CompiledGuard {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe {
            &mut (*self.compiled_ref)
        }
    }
}

/**
  | Controls compilation and runtime cloning
  | of execution steps.
  | 
  | If step.create_workspace=False,
  | this wrapper will compile the execution
  | step and its children once, and calls
  | to ExecutionStepWrapper::compiled()
  | will always return the same compiled
  | step. If step.create_workspace=True,
  | no compilation is done at creation time.
  | Instead, a new CompiledExecutionStep
  | is created for every compiled() call.
  | 
  | CompiledExecutionStep owns its Workspace,
  | and the lifetime of the compiled step
  | along with its workspace will be tied
  | to the lifetime of the `CompileGuard`
  | object returned by compiled().
  | 
  | ExecuteStepRecursive will call call
  | compiled() once before the given execution
  | step is run and keep it alive for the length
  | of its execution. This means that, for
  | steps with create_workspace=true,
  | a child workspace will be created every
  | time the step is executed, and destroyed
  | right afterwards.
  |
  */
pub struct ExecutionStepWrapper {

    step:                      *const ExecutionStep,
    external_workspace:        *mut Workspace,
    external_should_continue:  ShouldContinue,
    net_defs:                  *mut NetDefMap,
    compiled_step:             Box<CompiledExecutionStep>,
    ws_id_injector:            *mut WorkspaceIdInjector,
}

impl ExecutionStepWrapper {
    
    pub fn new(
        step:                     *const ExecutionStep,
        external_workspace:       *mut Workspace,
        external_should_continue: ShouldContinue,
        net_defs:                 *mut NetDefMap,
        ws_id_injector:           *mut WorkspaceIdInjector) -> Self {

        todo!();
        /*
            : step_(step),
            externalWorkspace_(externalWorkspace),
            externalShouldContinue_(externalShouldContinue),
            netDefs_(netDefs),
            ws_id_injector_(ws_id_injector) 

        // If this execution step does not create a child workspace,
        // then just eagerly-compile it. This will trigger CreateNet on the
        // nets used by this execution step.
        if (!step_->create_workspace()) {
          compiledStep_ = doCompile();
        }
        */
    }
    
    #[inline] pub fn step(&mut self) -> &ExecutionStep {
        
        todo!();
        /*
            return *step_;
        */
    }
    
    #[inline] pub fn compiled(&mut self) -> CompiledGuard {
        
        todo!();
        /*
            CompiledGuard guard;
        if (compiledStep_) {
          guard.reset(compiledStep_.get());
        } else {
          guard.reset(doCompile());
        }
        return guard;
        */
    }
}

///------------------
pub struct CompiledExecutionStep {

    step:                 *const ExecutionStep,
    workspace:            *mut Workspace,
    report_substeps:      Vec<Arc<ExecutionStepWrapper>>,
    recurring_substeps:   Vec<Arc<ExecutionStepWrapper>>,
    networks:             Vec<*mut NetBase>,
    report_net:           *mut NetBase,
    should_stop:          *mut Blob,  // default = nullptr
    net_should_continue:  ShouldContinue,
    should_continue:      ShouldContinue,
    got_failure:          AtomicBool, // default = false
    local_workspace:      Box<Workspace>,

    /// protects first_exception_
    exception_mutex:      parking_lot::RawMutex,

    first_exception:      ExceptionWrapper,
}

impl CompiledExecutionStep {
    
    #[inline] pub fn fail(&mut self, ex: &dyn Exception)  {
        
        todo!();
        /*
            {
                std::lock_guard<std::mutex> guard(exception_mutex_);
                if (!first_exception_) {
                    LOG(ERROR) << "Substep exception:\n" << c10::GetExceptionString(ex);
                    first_exception_ = ExceptionWrapper(ex);
                }
                gotFailure = true;
            }
            Cancel();
        */
    }
    
    #[inline] pub fn first_exception(&mut self) -> ExceptionWrapper {
        
        todo!();
        /*
            std::lock_guard<std::mutex> guard(exception_mutex_);
            return first_exception_;
        */
    }

    /**
      | Cancel attempts to cancel the running nets
      | in a best effort way. If the net or op
      | type does IO and doesn't implement
      | cancellation it may not be possible to
      | cancel leading to execution getting stuck
      | on error.
      */
    #[inline] pub fn cancel(&mut self)  {
        
        todo!();
        /*
            for (auto& substep : reportSubsteps) {
                substep->Cancel();
            }
            for (auto& substep : recurringSubsteps) {
                substep->Cancel();
            }
            for (auto& net : networks) {
                net->Cancel();
            }
            if (reportNet) {
                reportNet->Cancel();
            }
        */
    }
    
    pub fn new(
        main_step:                *const ExecutionStep,
        external_workspace:       *mut Workspace,
        external_should_continue: ShouldContinue,
        net_defs:                 *mut NetDefMap,
        ws_id_injector:           *mut WorkspaceIdInjector) -> Self {

        todo!();
        /*
            : step(mainStep) 

        if (mainStep->create_workspace()) {
          localWorkspace_.reset(new Workspace(externalWorkspace));
          workspace = localWorkspace_.get();
          ws_id_injector->InjectWorkspaceId(workspace);
        } else {
          workspace = externalWorkspace;
        }

        CAFFE_ENFORCE(
            (step->substep_size() == 0 || step->network_size() == 0),
            "An ExecutionStep should either have substep or networks"
            "but not both.");

        auto createAndGetNet = [&](const std::string& network_name) {
          auto it = netDefs->find(network_name);
          CAFFE_ENFORCE(
              it != netDefs->end(),
              "ExecutionStep " + mainStep->name() + " uses undefined net " +
                  network_name);
          // needsOverride does not need synchronization because it is only
          // relevant for non-dynamic executions steps. This is due to the fact
          // that concurrent nets run on child workspaces, that do not needOverride.
          if (it->second.needsOverride || !workspace->GetNet(network_name)) {
            workspace->CreateNet(*it->second.netDef, true);
            it->second.needsOverride = false;
          }
          auto* net = workspace->GetNet(network_name);
          CAFFE_ENFORCE(net != nullptr, "Network ", network_name, " not found.");
          return net;
        };

        if (step->substep_size()) {
          ShouldContinue substepShouldContinue;
          if (!step->concurrent_substeps() || step->substep().size() <= 1) {
            substepShouldContinue = externalShouldContinue;
          } else {
            substepShouldContinue = [this, externalShouldContinue](int64_t it) {
              return !gotFailure && externalShouldContinue(it);
            };
          }

          for (const auto& ss : step->substep()) {
            auto compiledSubstep = std::make_shared<ExecutionStepWrapper>(
                &ss, workspace, substepShouldContinue, netDefs, ws_id_injector);
            if (ss.has_run_every_ms()) {
              reportSubsteps.push_back(compiledSubstep);
            } else {
              recurringSubsteps.push_back(compiledSubstep);
            }
          }
        } else {
          for (const string& network_name : step->network()) {
            networks.push_back(createAndGetNet(network_name));
          }
        }

        if (step->has_should_stop_blob()) {
          shouldStop = workspace->GetBlob(step->should_stop_blob());
          CAFFE_ENFORCE(
              shouldStop, "blob ", step->should_stop_blob(), " does not exist");
        }

        if (step->has_report_net()) {
          CAFFE_ENFORCE(
              step->has_report_interval(),
              "A report_interval must be provided if report_net is set.");
          reportNet = createAndGetNet(step->report_net());
        } else {
          reportNet = nullptr;
        }

        netShouldContinue = getContinuationTest(workspace, *step);
        shouldContinue = [this, externalShouldContinue](int64_t iter) {
          return externalShouldContinue(iter) && this->netShouldContinue(iter);
        };
        */
    }
}

impl ExecutionStepWrapper {
    
    #[inline] pub fn cancel(&mut self)  {
        
        todo!();
        /*
            if (compiledStep_) {
        compiledStep_->Cancel();
      }
        */
    }
    
    #[inline] pub fn do_compile(&mut self) -> Box<CompiledExecutionStep> {
        
        todo!();
        /*
            return std::unique_ptr<CompiledExecutionStep>(new CompiledExecutionStep(
          step_,
          externalWorkspace_,
          externalShouldContinue_,
          netDefs_,
          ws_id_injector_));
        */
    }
}

///----------------------------
struct ReporterInstance {
    report_mutex:   parking_lot::RawMutex,
    report_cv:      std::sync::Condvar,
    report_thread:  std::thread::Thread,
    exception:      ExceptionWrapper,
}

impl ReporterInstance {
    
    pub fn new(
        interval_millis: i32,
        done:            *mut AtomicBool,
        f:               fn(_u0: c_void) -> c_void,
        compiled_step:   *mut CompiledGuard) -> Self {

        todo!();
        /*
            auto interval = std::chrono::milliseconds(intervalMillis);
            auto reportWorker = [=]() {
                std::unique_lock<std::mutex> lk(report_mutex);
                do {
                    report_cv.wait_for(lk, interval, [&]() { return done->load(); });
                    try {
                        f();
                    } catch (const std::exception& ex) {
                        LOG(ERROR) << "Reporter instance exception:\n"
                            << c10::GetExceptionString(ex);
                        if (!FLAGS_caffe2_handle_executor_threads_exceptions) {
                            throw;
                        }
                        (*compiledStep)->Fail(ex);
                        done->store(true);
                    }
                } while (!done->load());
            };
            report_thread = std::thread(reportWorker);
        */
    }
}

///----------------------------
struct Reporter {
    instances:      Vec<Box<ReporterInstance>>,
    done:           AtomicBool, // default = false
    compiled_step:  *mut CompiledGuard,
}

impl Reporter {
    
    pub fn new(compiled_step: *mut CompiledGuard) -> Self {
    
        todo!();
        /*
            : compiledStep_(compiledStep)
        */
    }
    
    #[inline] pub fn start(&mut self, interval_millis: i64, f: fn(_u0: c_void) -> c_void)  {
        
        todo!();
        /*
            instances_.emplace_back(
            new ReporterInstance(intervalMillis, &done_, f, compiledStep_));
        */
    }
}

impl Drop for Reporter {
    fn drop(&mut self) {
        todo!();
        /* 
        done_ = true;
        for (auto& instance : instances_) {
          if (!instance->report_thread.joinable()) {
            continue;
          }
          instance->report_cv.notify_all();
          instance->report_thread.join();
        }
       */
    }
}

#[macro_export] macro_rules! check_should_stop {
    ($step:ident, $shouldStop:ident) => {
        todo!();
        /*
        
          if (getShouldStop(shouldStop)) {                                
            VLOG(1) << "Execution step " << step.name() << " stopped by " 
                    << step.should_stop_blob();                           
            return true;                                                  
          }
        */
    }
}

#[inline] pub fn execute_step_recursive(step_wrapper: &mut ExecutionStepWrapper) -> bool {
    
    todo!();
    /*
        const auto& step = stepWrapper.step();
      auto compiledStep = stepWrapper.compiled();

      VLOG(1) << "Running execution step " << step.name();

      std::unique_ptr<Reporter> reporter;
      if (step.has_report_net() || compiledStep->reportSubsteps.size() > 0) {
        reporter = std::make_unique<Reporter>(&compiledStep);
        auto* reportNet = compiledStep->reportNet;
        if (reportNet) {
          VLOG(1) << "Starting reporter net";
          reporter->start(step.report_interval() * 1000, [reportNet]() {
            if (!reportNet->Run()) {
              LOG(WARNING) << "Error running report_net.";
            }
          });
        }
        for (auto& substepWrapper : compiledStep->reportSubsteps) {
          reporter->start(
              substepWrapper->step().run_every_ms(), [substepWrapper]() {
                if (!ExecuteStepRecursive(*substepWrapper)) {
                  LOG(WARNING) << "Error running report step.";
                }
              });
        }
      }

      const Blob* shouldStop = compiledStep->shouldStop;

      if (step.substep_size()) {
        bool sequential =
            (!step.concurrent_substeps() || step.substep().size() <= 1) &&
            (!step.has_num_concurrent_instances() ||
             step.num_concurrent_instances() <= 1);
        for (int64_t iter = 0; compiledStep->shouldContinue(iter); ++iter) {
          if (sequential) {
            VLOG(1) << "Executing step " << step.name() << " iteration " << iter;
            for (auto& substepWrapper : compiledStep->recurringSubsteps) {
              if (!ExecuteStepRecursive(*substepWrapper)) {
                return false;
              }
              CHECK_SHOULD_STOP(step, shouldStop);
            }
          } else {
            VLOG(1) << "Executing step " << step.name() << " iteration " << iter
                    << " with " << step.substep().size() << " concurrent substeps";

            std::atomic<int> next_substep{0};
            std::condition_variable cv;
            std::mutex exception_mutex; // protects done
            int done{0};
            auto worker = [&]() {
              ScopeExitGuard on_exit([&] {
                std::lock_guard<std::mutex> guard(exception_mutex);
                done += 1;
                cv.notify_all();
              });

              auto num_substeps = compiledStep->recurringSubsteps.size();
              int substep_id = next_substep++ % num_substeps;
              if (compiledStep->gotFailure) {
                return;
              }
              try {
                if (!ExecuteStepRecursive(
                        *compiledStep->recurringSubsteps.at(substep_id))) {
                  compiledStep->gotFailure = true;
                }
              } catch (const std::exception& ex) {
                compiledStep->Fail(ex);
                if (!FLAGS_caffe2_handle_executor_threads_exceptions) {
                  // In complex plans other threads might get stuck if another
                  // one fails. So we let exception to go out of thread which
                  // causes SIGABRT. In local setup one might use this flag
                  // in order to use Python debugger after a failure
                  throw;
                }
              }
            };

            std::unique_lock<std::mutex> guard(exception_mutex);

            std::vector<std::thread> threads;
            auto numThreads = compiledStep->recurringSubsteps.size();
            if (step.has_num_concurrent_instances()) {
              numThreads *= step.num_concurrent_instances();
            }
            for (size_t i = 0; i < numThreads; ++i) {
              threads.emplace_back(worker);
            }

            auto workersDone = [&] { return done == numThreads; };

            // If we get an exception, try to wait for all threads to stop
            // gracefully.
            cv.wait(
                guard, [&] { return workersDone() || compiledStep->gotFailure; });
            cv.wait_for(
                guard,
                std::chrono::seconds(FLAGS_caffe2_plan_executor_exception_timeout),
                [&] { return workersDone(); });
            auto first_exception = compiledStep->FirstException();
            if (!workersDone() && first_exception) {
              LOG(ERROR) << "failed to stop concurrent workers after exception: "
                         << first_exception.what();
              ExceptionWrapperTerminate(std::move(first_exception));
            }

            for (auto& thread : threads) {
              thread.join();
            }
            if (compiledStep->gotFailure) {
              LOG(ERROR) << "One of the workers failed.";
              if (first_exception) {
                first_exception.rethrowException();
              }
              return false;
            }
            // concurrent substeps should be careful about setting should_stop_blob
            CHECK_SHOULD_STOP(step, shouldStop);
          }
        }
        return true;
      } else {
        // If this ExecutionStep just contains nets, we can directly run it.
        for (int64_t iter = 0; compiledStep->shouldContinue(iter); ++iter) {
          VLOG(1) << "Executing networks " << step.name() << " iteration " << iter;
          for (NetBase* network : compiledStep->networks) {
            if (!network->Run()) {
              return false;
            }
            CHECK_SHOULD_STOP(step, shouldStop);
          }
        }
      }

      if (auto first_exception = compiledStep->FirstException()) {
        first_exception.rethrowException();
      }
      return !compiledStep->gotFailure;
    */
}

#[inline] pub fn run_plan_on_workspace(
    ws:              *mut Workspace,
    plan:            &PlanDef,
    should_continue: ShouldContinue) -> bool {
    
    todo!();
    /*
        LOG(INFO) << "Started executing plan " << plan.name();
      if (plan.execution_step_size() == 0) {
        LOG(WARNING) << "Nothing to run - did you define a correct plan?";
        // We will do nothing, but the plan is still legal so we will return true.
        return true;
      }
      LOG(INFO) << "Initializing networks for plan " << plan.name();

      NetDefMap net_defs;
      for (const NetDef& net_def : plan.network()) {
        LOG(INFO) << "Processing net '" << net_def.name() << "', type: '"
                  << net_def.type() << "', #ops: " << net_def.op_size()
                  << ", num_workers: " << net_def.num_workers();
        CAFFE_ENFORCE(
            net_defs.count(net_def.name()) == 0,
            "Your plan contains networks of the same name \"",
            net_def.name(),
            "\", which should not happen. Check your plan to see "
            "if you made a programming error in creating the plan.");
        auto netAlreadyExists = ws->GetNet(net_def.name()) != nullptr;
        net_defs[net_def.name()] = NetDefInfo{&net_def, netAlreadyExists};
      }
      WorkspaceIdInjector ws_id_injector;
      Timer plan_timer;
      for (const ExecutionStep& step : plan.execution_step()) {
        Timer step_timer;
        ExecutionStepWrapper stepWrapper(
            &step, ws, shouldContinue, &net_defs, &ws_id_injector);
        if (!ExecuteStepRecursive(stepWrapper)) {
          LOG(ERROR) << "Failed initializing step " << step.name();
          return false;
        }
        LOG(INFO) << "Step " << step.name() << " in plan " << plan.name()
                  << " took " << step_timer.Seconds() << " seconds.";
      }
      LOG(INFO) << "Total plan " << plan.name() << " took " << plan_timer.Seconds()
                << " seconds.";
      LOG(INFO) << "Plan " << plan.name() << " executed successfully.";
      return true;
    */
}
