crate::ix!();

use crate::{
    Workspace,
    OperatorStorage,
    NetBase,
    ObserverBase,
};

static counter: AtomicI32 = AtomicI32::new(0);

pub struct DummyObserver<T> {
    base: ObserverBase<T>,
}

impl<T> DummyObserver<T> {

    pub fn new(subject: *mut T) -> Self {
    
        todo!();
        /*
            : ObserverBase<T>(subject_)
        */
    }
}

impl DummyObserver<NetBase> {

    #[inline] pub fn start(&mut self)  {
        
        todo!();
        /*
            vector<OperatorStorage*> operators = subject_->GetOperators();
      for (auto& op : operators) {
        op->AttachObserver(std::make_unique<DummyObserver<OperatorStorage>>(op));
      }
      counter.fetch_add(1000);
        */
    }
}

impl DummyObserver<OperatorStorage> {
    
    #[inline] pub fn start(&mut self)  {
        
        todo!();
        /*
            counter.fetch_add(100);
        */
    }
}

impl DummyObserver<NetBase> {

    #[inline] pub fn stop(&mut self)  {
        
        todo!();
        /*
            counter.fetch_add(10);
        */
    }
}

impl DummyObserver<OperatorStorage> {

    #[inline] pub fn stop(&mut self)  {
        
        todo!();
        /*
            counter.fetch_add(1);
        */
    }
}

///-----------------------------
pub struct ObsTestDummyOp {
    base: OperatorStorage,
}
impl ObsTestDummyOp {

    #[inline] pub fn run(&mut self, unused: i32) -> bool {
        
        todo!();
        /*
            StartAllObservers();
        StopAllObservers();
        return true;
        */
    }
}

register_cpu_operator!{ObsTestDummy, ObsTestDummyOp}

register_cuda_operator!{ObsTestDummy, ObsTestDummyOp}

num_inputs!{ObsTestDummy, (0,INT_MAX)}

num_outputs!{ObsTestDummy, (0,INT_MAX)}

allow_inplace!{ObsTestDummy, vec![(0, 0), (1, 1)]}

#[inline] pub fn create_net_test_helper(
    ws:    *mut Workspace, 
    isDAG: Option<bool>) -> Box<NetBase> 
{
    let isDAG: bool = isDAG.unwrap_or(false);

    todo!();
    /*
        NetDef net_def;
      if (isDAG) {
        net_def.set_type("dag");
      }
      {
        auto& op = *(net_def.add_op());
        op.set_type("ObsTestDummy");
        op.add_input("in");
        op.add_output("hidden");
      }
      {
        auto& op = *(net_def.add_op());
        op.set_type("ObsTestDummy");
        op.add_input("hidden");
        op.add_output("out");
      }
      net_def.add_external_input("in");
      net_def.add_external_output("out");

      return CreateNet(net_def, ws);
    */
}

#[test] fn ObserverTest_TestNotify() {
    todo!();
    /*
      auto count_before = counter.load();
      Workspace ws;
      ws.CreateBlob("in");
      NetDef net_def;
      unique_ptr<NetBase> net(CreateNetTestHelper(&ws));
      EXPECT_EQ(caffe2::dynamic_cast_if_rtti<SimpleNet*>(net.get()), net.get());
      unique_ptr<DummyObserver<NetBase>> net_ob =
          make_unique<DummyObserver<NetBase>>(net.get());
      net.get()->AttachObserver(std::move(net_ob));
      net.get()->Run();
      auto count_after = counter.load();
      EXPECT_EQ(1212, count_after - count_before);
  */
}


#[test] fn ObserverTest_TestUniqueMap() {
    todo!();
    /*
      auto count_before = counter.load();
      Workspace ws;
      ws.CreateBlob("in");
      NetDef net_def;
      unique_ptr<NetBase> net(CreateNetTestHelper(&ws));
      EXPECT_EQ(caffe2::dynamic_cast_if_rtti<SimpleNet*>(net.get()), net.get());
      unique_ptr<DummyObserver<NetBase>> net_ob =
          make_unique<DummyObserver<NetBase>>(net.get());
      auto* ref = net.get()->AttachObserver(std::move(net_ob));
      net.get()->Run();
      unique_ptr<Observable<NetBase>::Observer> test =
          net.get()->DetachObserver(ref);
      auto count_after = counter.load();
      EXPECT_EQ(1212, count_after - count_before);
  */
}


#[test] fn ObserverTest_TestNotifyAfterDetach() {
    todo!();
    /*
      auto count_before = counter.load();
      Workspace ws;
      ws.CreateBlob("in");
      NetDef net_def;
      unique_ptr<NetBase> net(CreateNetTestHelper(&ws));
      unique_ptr<DummyObserver<NetBase>> net_ob =
          make_unique<DummyObserver<NetBase>>(net.get());
      auto* ob = net.get()->AttachObserver(std::move(net_ob));
      net.get()->DetachObserver(ob);
      net.get()->Run();
      auto count_after = counter.load();
      EXPECT_EQ(0, count_after - count_before);
  */
}


#[test] fn ObserverTest_TestDAGNetBase() {
    todo!();
    /*
      auto count_before = counter.load();
      Workspace ws;
      ws.CreateBlob("in");
      NetDef net_def;
      unique_ptr<NetBase> net(CreateNetTestHelper(&ws, true));
      unique_ptr<DummyObserver<NetBase>> net_ob =
          make_unique<DummyObserver<NetBase>>(net.get());
      net.get()->AttachObserver(std::move(net_ob));
      net.get()->Run();
      auto count_after = counter.load();
      EXPECT_EQ(1212, count_after - count_before);
  */
}

/**
  | This test intermittently segfaults,
  | see https://github.com/pytorch/pytorch/issues/9137
  |
  */
#[test] fn ObserverTest_TestMultipleNetBase() {
    todo!();
    /*
      Workspace ws;
      ws.CreateBlob("in");
      NetDef net_def;
      unique_ptr<NetBase> net(CreateNetTestHelper(&ws, true));
      EXPECT_EQ(caffe2::dynamic_cast_if_rtti<NetBase*>(net.get()), net.get());

      // There may be some default observers
      const size_t prev_num = net.get()->NumObservers();
      const int num_tests = 100;
      vector<const Observable<NetBase>::Observer*> observers;
      for (int i = 0; i < num_tests; ++i) {
        unique_ptr<DummyObserver<NetBase>> net_ob =
            make_unique<DummyObserver<NetBase>>(net.get());
        observers.emplace_back(net.get()->AttachObserver(std::move(net_ob)));
      }

      net.get()->Run();

      for (const auto& observer : observers) {
        net.get()->DetachObserver(observer);
      }

      EXPECT_EQ(net.get()->NumObservers(), prev_num);
  */
}
