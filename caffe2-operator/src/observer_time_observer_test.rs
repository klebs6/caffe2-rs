crate::ix!();

pub struct SleepOp {
    base: OperatorStorage,
}

impl SleepOp {
    
    #[inline] pub fn run(&mut self, unused: i32) -> bool {
        
        todo!();
        /*
            StartAllObservers();
        std::this_thread::sleep_for(std::chrono::milliseconds(3000));
        StopAllObservers();
        return true;
        */
    }
}

register_cpu_operator!{SleepOp, SleepOp}

register_cuda_operator!{SleepOp, SleepOp}

num_inputs!{SleepOp, (0,INT_MAX)}

num_outputs!{SleepOp, (0,INT_MAX)}

allow_inplace!{SleepOp, vec![(0, 0), (1, 1)]}

#[inline] pub fn create_net_test_helper(ws: *mut Workspace) -> Box<NetBase> {
    
    todo!();
    /*
        NetDef net_def;
      {
        auto& op = *(net_def.add_op());
        op.set_type("SleepOp");
        op.add_input("in");
        op.add_output("hidden");
      }
      {
        auto& op = *(net_def.add_op());
        op.set_type("SleepOp");
        op.add_input("hidden");
        op.add_output("out");
      }
      net_def.add_external_input("in");
      net_def.add_external_output("out");

      return CreateNet(net_def, ws);
    */
}

#[test] fn TimeObserverTest_Test3Seconds() {
    todo!();
    /*
      Workspace ws;
      ws.CreateBlob("in");
      NetDef net_def;
      unique_ptr<NetBase> net(CreateNetTestHelper(&ws));
      auto net_ob = std::make_unique<TimeObserver>(net.get());
      const auto* ob = net_ob.get();
      net->AttachObserver(std::move(net_ob));
      net->Run();
      CAFFE_ENFORCE(ob);
      LOG(INFO) << "av time children: " << ob->average_time_children();
      LOG(INFO) << "av time: " << ob->average_time();
      CAFFE_ENFORCE(ob->average_time_children() > 3000);
      CAFFE_ENFORCE(ob->average_time_children() < 3500);
      CAFFE_ENFORCE(ob->average_time() > 6000);
      CAFFE_ENFORCE(ob->average_time() < 6500);
  */
}
