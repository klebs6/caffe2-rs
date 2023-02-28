crate::ix!();

use crate::{
    OperatorStorage,
    PlanDef,
    Workspace,
    OperatorDef,
    CPUContext,
    Operator,
};

#[test] fn PlanExecutorTest_EmptyPlan() {
    todo!();
    /*
      PlanDef plan_def;
      Workspace ws;
      EXPECT_TRUE(ws.RunPlan(plan_def));
  */
}

lazy_static!{
    static ref cancel_count:  Atomic<i32> = Atomic::<i32>::new(0);
    static ref stuck_run:     AtomicBool = AtomicBool::new(false);
}

///-------------------
pub struct StuckBlockingOp {
    storage: OperatorStorage,
    context: CPUContext,

    cancelled:  AtomicBool, // {false};
}

impl StuckBlockingOp {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : Operator<CPUContext>(operator_def, ws)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            // StuckBlockingOp runs and notifies ErrorOp.
        stuckRun = true;

        while (!cancelled_) {
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        return true;
        */
    }
    
    #[inline] pub fn cancel(&mut self)  {
        
        todo!();
        /*
            LOG(INFO) << "cancelled StuckBlockingOp.";
        cancelCount += 1;
        cancelled_ = true;
        */
    }
}

register_cpu_operator!{StuckBlocking, StuckBlockingOp}

num_inputs!{StuckBlocking, 0}

num_outputs!{StuckBlocking, 0}

///------------
pub struct NoopOp {
    storage: OperatorStorage,
    context: CPUContext,
}

impl NoopOp {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : Operator<CPUContext>(operator_def, ws)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            // notify Error op we've ran.
        stuckRun = true;
        return true;
        */
    }
}

register_cpu_operator!{Noop, NoopOp}

num_inputs!{Noop, 0}

num_outputs!{Noop, 0}

///------------------
pub struct StuckAsyncOp {
    storage: OperatorStorage,
    context: CPUContext,
}

impl StuckAsyncOp {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : Operator<CPUContext>(operator_def, ws)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            // notify Error op we've ran.
        stuckRun = true;
        // explicitly don't call SetFinished so this gets stuck
        return true;
        */
    }
    
    #[inline] pub fn cancel_async_callback(&mut self)  {
        
        todo!();
        /*
            LOG(INFO) << "cancelled";
        cancelCount += 1;
        */
    }
    
    #[inline] pub fn has_async_part(&self) -> bool {
        
        todo!();
        /*
            return true;
        */
    }
}

register_cpu_operator!{StuckAsync, StuckAsyncOp}

num_inputs!{StuckAsync, 0}

num_outputs!{StuckAsync, 0}

pub struct TestError { }

pub trait Exception { 
    fn what() -> &'static str where Self: Sized;
}

impl Exception for TestError {
    fn what() -> &'static str {
        "test error"
    }
}


///-------------------
pub struct ErrorOp {
    storage: OperatorStorage,
    context: CPUContext,
}

impl ErrorOp {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : Operator<CPUContext>(operator_def, ws)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            // Wait for StuckAsyncOp or StuckBlockingOp to run first.
        while (!stuckRun) {
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        throw TestError();
        return true;
        */
    }
}

register_cpu_operator!{Error, ErrorOp}

num_inputs!{Error, 0}

num_outputs!{Error, 0}

lazy_static!{
    static ref blocking_error_runs: AtomicI32 = AtomicI32::new(0);
}

///------------------
pub struct BlockingErrorOp {
    storage: OperatorStorage,
    context: CPUContext,
}

impl BlockingErrorOp {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : Operator<CPUContext>(operator_def, ws)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            // First n op executions should block and then start throwing errors.
        if (blockingErrorRuns.fetch_sub(1) >= 1) {
          LOG(INFO) << "blocking";
          while (true) {
            std::this_thread::sleep_for(std::chrono::hours(10));
          }
        } else {
          LOG(INFO) << "throwing";
          throw TestError();
        }
        */
    }
}

register_cpu_operator!{BlockingError, BlockingErrorOp}

num_inputs!{BlockingError, 0}

num_outputs!{BlockingError, 0}

#[inline] pub fn parallel_error_plan() -> PlanDef {
    
    todo!();
    /*
        PlanDef plan_def;

      auto* stuck_net = plan_def.add_network();
      stuck_net->set_name("stuck_net");
      stuck_net->set_type("async_scheduling");
      {
        auto* op = stuck_net->add_op();
        op->set_type("StuckAsync");
      }

      auto* error_net = plan_def.add_network();
      error_net->set_name("error_net");
      error_net->set_type("async_scheduling");
      {
        auto op = error_net->add_op();
        op->set_type("Error");
      }

      auto* execution_step = plan_def.add_execution_step();
      execution_step->set_concurrent_substeps(true);
      {
        auto* substep = execution_step->add_substep();
        substep->add_network(stuck_net->name());
      }
      {
        auto* substep = execution_step->add_substep();
        substep->add_network(error_net->name());
      }

      return plan_def;
    */
}

#[inline] pub fn parallel_error_plan_with_cancellable_stuck_net() -> PlanDef {
    
    todo!();
    /*
        // Set a plan with two nets: one stuck net with blocking operator that never
      // returns; one error net with error op that throws.
      PlanDef plan_def;

      auto* stuck_blocking_net = plan_def.add_network();
      stuck_blocking_net->set_name("stuck_blocking_net");
      {
        auto* op = stuck_blocking_net->add_op();
        op->set_type("StuckBlocking");
      }

      auto* error_net = plan_def.add_network();
      error_net->set_name("error_net");
      {
        auto* op = error_net->add_op();
        op->set_type("Error");
      }

      auto* execution_step = plan_def.add_execution_step();
      execution_step->set_concurrent_substeps(true);
      {
        auto* substep = execution_step->add_substep();
        substep->add_network(stuck_blocking_net->name());
      }
      {
        auto* substep = execution_step->add_substep();
        substep->add_network(error_net->name());
      }

      return plan_def;
    */
}

#[inline] pub fn reporter_error_plan_with_cancellable_stuck_net() -> PlanDef {
    
    todo!();
    /*
        // Set a plan with a concurrent net and a reporter net: one stuck net with
      // blocking operator that never returns; one reporter net with error op
      // that throws.
      PlanDef plan_def;

      auto* stuck_blocking_net = plan_def.add_network();
      stuck_blocking_net->set_name("stuck_blocking_net");
      {
        auto* op = stuck_blocking_net->add_op();
        op->set_type("StuckBlocking");
      }

      auto* error_net = plan_def.add_network();
      error_net->set_name("error_net");
      {
        auto* op = error_net->add_op();
        op->set_type("Error");
      }

      auto* execution_step = plan_def.add_execution_step();
      execution_step->set_concurrent_substeps(true);
      {
        auto* substep = execution_step->add_substep();
        substep->add_network(stuck_blocking_net->name());
      }
      {
        auto* substep = execution_step->add_substep();
        substep->set_run_every_ms(1);
        substep->add_network(error_net->name());
      }

      return plan_def;
    */
}

pub struct HandleExecutorThreadExceptionsGuard { }

impl HandleExecutorThreadExceptionsGuard {
    
    #[inline] pub fn global_init(&mut self, args: Vec<String>)  {
        
        todo!();
        /*
            std::vector<char*> args_ptrs;
        for (auto& arg : args) {
          args_ptrs.push_back(const_cast<char*>(arg.data()));
        }
        char** new_argv = args_ptrs.data();
        int new_argc = args.size();
        CAFFE_ENFORCE(GlobalInit(&new_argc, &new_argv));
        */
    }
    
    pub fn new(timeout: Option<i32>) -> Self {
    
        let timeout: i32 = timeout.unwrap_or(60);

        todo!();
        /*
            globalInit({
            "caffe2",
            "--caffe2_handle_executor_threads_exceptions=1",
            "--caffe2_plan_executor_exception_timeout=" +
                caffe2::to_string(timeout),
        });
        */
    }
}

impl Drop for HandleExecutorThreadExceptionsGuard {
    fn drop(&mut self) {
        todo!();
        /* 
        globalInit({
            "caffe2",
        });
       */
    }
}

#[test] fn PlanExecutorTest_ErrorAsyncPlan() {
    todo!();
    /*
      HandleExecutorThreadExceptionsGuard guard;

      cancelCount = 0;
      PlanDef plan_def = parallelErrorPlan();
      Workspace ws;
      ASSERT_THROW(ws.RunPlan(plan_def), TestError);
      ASSERT_EQ(cancelCount, 1);
  */
}

/// death tests not supported on mobile
#[cfg(all(not(caffe2_is_xplat_build), not(c10_mobile)))]
#[test] fn PlanExecutorTest_BlockingErrorPlan() {
    todo!();
    /*
      // TSAN doesn't play nicely with death tests
    #if defined(__has_feature)
    #if __has_feature(thread_sanitizer)
      return;
    #endif
    #endif

      ASSERT_DEATH(
          [] {
            HandleExecutorThreadExceptionsGuard guard(/*timeout=*/1);

            PlanDef plan_def;

            std::string plan_def_template = R"DOC(
              network {
                name: "net"
                op {
                  type: "BlockingError"
                }
              }
              execution_step {
                num_concurrent_instances: 2
                substep {
                  network: "net"
                }
              }
            )DOC";

            CAFFE_ENFORCE(
                TextFormat::ParseFromString(plan_def_template, &plan_def));
            Workspace ws;
            blockingErrorRuns = 1;
            ws.RunPlan(plan_def);
            FAIL() << "shouldn't have reached this point";
          }(),
          "failed to stop concurrent workers after exception: test error");
  */
}

#[test] fn PlanExecutorTest_ErrorPlanWithCancellableStuckNet() {
    todo!();
    /*
      HandleExecutorThreadExceptionsGuard guard;

      cancelCount = 0;
      PlanDef plan_def = parallelErrorPlanWithCancellableStuckNet();
      Workspace ws;

      ASSERT_THROW(ws.RunPlan(plan_def), TestError);
      ASSERT_EQ(cancelCount, 1);
  */
}


#[test] fn PlanExecutorTest_ReporterErrorPlanWithCancellableStuckNet() {
    todo!();
    /*
      HandleExecutorThreadExceptionsGuard guard;

      cancelCount = 0;
      PlanDef plan_def = reporterErrorPlanWithCancellableStuckNet();
      Workspace ws;

      ASSERT_THROW(ws.RunPlan(plan_def), TestError);
      ASSERT_EQ(cancelCount, 1);
  */
}

#[inline] pub fn should_stop_with_cancel_plan() -> PlanDef {
    
    todo!();
    /*
        // Set a plan with a looping net with should_stop_blob set and a concurrent
      // net that throws an error. The error should cause should_stop to return
      // false and end the concurrent net.
      PlanDef plan_def;

      auto* should_stop_net = plan_def.add_network();
      {
        auto* op = should_stop_net->add_op();
        op->set_type("Noop");
      }
      should_stop_net->set_name("should_stop_net");
      should_stop_net->set_type("async_scheduling");

      auto* error_net = plan_def.add_network();
      error_net->set_name("error_net");
      {
        auto* op = error_net->add_op();
        op->set_type("Error");
      }

      auto* execution_step = plan_def.add_execution_step();
      execution_step->set_concurrent_substeps(true);
      {
        auto* substep = execution_step->add_substep();
      execution_step->set_concurrent_substeps(true);
        substep->set_name("concurrent_should_stop");
        substep->set_should_stop_blob("should_stop_blob");
        auto* substep2 = substep->add_substep();
        substep2->set_name("should_stop_net");
        substep2->add_network(should_stop_net->name());
        substep2->set_num_iter(10);
      }
      {
        auto* substep = execution_step->add_substep();
        substep->set_name("error_step");
        substep->add_network(error_net->name());
      }

      return plan_def;
    */
}

#[test] fn PlanExecutorTest_ShouldStopWithCancel() {
    todo!();
    /*
      HandleExecutorThreadExceptionsGuard guard;

      stuckRun = false;
      PlanDef plan_def = shouldStopWithCancelPlan();
      Workspace ws;

      Blob* blob = ws.CreateBlob("should_stop_blob");
      Tensor* tensor = BlobGetMutableTensor(blob, CPU);
      const vector<int64_t>& shape{1};
      tensor->Resize(shape);
      tensor->mutable_data<bool>()[0] = false;

      ASSERT_THROW(ws.RunPlan(plan_def), TestError);
      ASSERT_TRUE(stuckRun);
  */
}
