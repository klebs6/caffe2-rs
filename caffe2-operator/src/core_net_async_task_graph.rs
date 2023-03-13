crate::ix!();


/**
  | AsyncTaskGraph represents an execution
  | of a net, it owns the tasks and associated
  | futures, sets up future callbacks and
  | propagates errors.
  | 
  | Usage steps:
  | 
  | - Adding graph nodes and edges through
  | CreateNode/AddDependency;
  | 
  | - Freezing the graph (FreezeGraph),
  | after the freezing a future can be obtained
  | using GetFuture;
  | 
  | - Execution of the graph is scheduled
  | through ExecuteGraph, after each execution
  | Reset must be called to prepare the graph
  | for the next run
  |
  */
pub trait AsyncTaskGraphBase {

    fn create_node(&mut self, node_id: i32, ops: &Vec<*mut OperatorStorage>) -> bool;

    fn add_dependency(&mut self, child_node_id: i32, parent_node_ids: &Vec<i32>) -> bool;

    fn freeze_graph(&mut self);

    fn execute_graph(&mut self) -> *mut AsyncTaskFuture;

    fn get_future(&mut self) -> *mut AsyncTaskFuture;

    fn reset(&mut self);
}

pub struct AsyncTaskGraph {

    /**
      | used to, e.g., get access to executor's
      | thread pools
      |
      | TODO: pass tracer and counters through
      | ExecutorHelper
      */
    helper:        *mut ExecutorHelper,

    options:       ExecutionOptions,
    frozen:        bool,
    nodes:         HashMap<i32,Box<AsyncTask>>,
    parents:       HashMap<i32,HashSet<i32>>,
    children:      HashMap<i32,HashSet<i32>>,
    edge_futures:  Vec<Box<AsyncTaskFuture>>,
    root_tasks:    Vec<*mut AsyncTask>,
    run_future:    Box<AsyncTaskFuture>,
}

impl AsyncTaskGraph {

    pub fn new(helper: *mut ExecutorHelper, options: &ExecutionOptions) -> Self {
    
        todo!();
        /*
            : helper_(helper), options_(options), frozen_(false)
        */
    }
    
}

impl AsyncTaskGraphBase for AsyncTaskGraph {

    fn create_node(&mut self, node_id: i32, ops: &Vec<*mut OperatorStorage>) -> bool {
        
        todo!();
        /*
            CAFFE_ENFORCE(!frozen_);
      if (!nodes_.count(node_id)) {
        nodes_[node_id] = std::make_unique<AsyncTask>(ops);
        return true;
      } else {
        return false;
      }
        */
    }
    
    fn add_dependency(&mut self, child_node_id: i32, parent_node_ids: &Vec<i32>) -> bool {
        
        todo!();
        /*
            CAFFE_ENFORCE(!frozen_);
      CAFFE_ENFORCE(!parent_node_ids.empty());
      CAFFE_ENFORCE(nodes_.count(child_node_id));
      for (auto node_id : parent_node_ids) {
        CAFFE_ENFORCE(nodes_.count(node_id));
      }
      CAFFE_ENFORCE(!parents_.count(child_node_id));

      auto* child_task = nodes_[child_node_id].get();
      auto child_device = child_task->GetDeviceOption();

      std::vector<AsyncTaskFuture*> parent_futures;
      for (auto node_id : parent_node_ids) {
        parents_[child_node_id].insert(node_id);
        children_[node_id].insert(child_node_id);
        parent_futures.push_back(&nodes_[node_id]->GetFuture());
      }

      AsyncTaskFuture* parents_future = nullptr;
      if (parent_futures.size() > 1) {
        edge_futures_.push_back(
            std::make_unique<AsyncTaskFuture>(parent_futures));
        parents_future = edge_futures_.back().get();
      } else {
        CAFFE_ENFORCE_EQ(parent_futures.size(), 1);
        parents_future = parent_futures.back();
      }

      // TODO: CUDA polling
      parents_future->SetCallback(
          [this, child_task, child_device](const AsyncTaskFuture* f) {
            CAFFE_ENFORCE(f->IsCompleted());
            if (!f->IsFailed()) {
              // if we're in the correct thread pool and DFS scheduling is enabled,
              // immediately call task inline, otherwise send task into thread pool
              auto* pool = helper_->GetPool(child_device);
              if (pool->inThreadPool() && options_.use_dfs_scheduling_) {
                child_task->Run(options_);
              } else {
                pool->run([this, child_task]() { child_task->Run(options_); });
              }
            } else {
              // skip task execution and propagate error further
              child_task->GetFuture().SetCompleted(f->ErrorMessage().c_str());
            }
          });

      return true;
        */
    }
    
    fn freeze_graph(&mut self)  {
        
        todo!();
        /*
            if (frozen_) {
        return;
      }

      CAFFE_ENFORCE(!run_future_);
      CAFFE_ENFORCE(root_tasks_.empty());

      std::vector<AsyncTaskFuture*> final_futures;
      for (auto& kv : nodes_) {
        auto task_id = kv.first;
        auto* task = kv.second.get();

        if (parents_[task_id].empty()) {
          root_tasks_.push_back(task);
        }

        if (children_[task_id].empty()) {
          auto& future = task->GetFuture();
          final_futures.push_back(&future);
        }
      }

      CAFFE_ENFORCE(!root_tasks_.empty());
      CAFFE_ENFORCE(!final_futures.empty());

      run_future_ = std::make_unique<AsyncTaskFuture>(final_futures);

      frozen_ = true;
        */
    }
    
    fn execute_graph(&mut self) -> *mut AsyncTaskFuture {
        
        todo!();
        /*
            CAFFE_ENFORCE(frozen_);
      CAFFE_ENFORCE(run_future_ && !run_future_->IsCompleted());

      // TODO: run root tasks inline in inference mode
      for (auto* task : root_tasks_) {
        auto task_device = task->GetDeviceOption();
        helper_->GetPool(task_device)->run([this, task]() { task->Run(options_); });
      }

      return run_future_.get();
        */
    }
    
    fn get_future(&mut self) -> *mut AsyncTaskFuture {
        
        todo!();
        /*
            CAFFE_ENFORCE(frozen_);
      return run_future_.get();
        */
    }
    
    fn reset(&mut self)  {
        
        todo!();
        /*
            CAFFE_ENFORCE(frozen_);
      for (auto& kv : nodes_) {
        kv.second->Reset();
      }
      for (auto& future : edge_futures_) {
        future->ResetState();
      }
      if (run_future_) {
        run_future_->ResetState();
      }
        */
    }
}
