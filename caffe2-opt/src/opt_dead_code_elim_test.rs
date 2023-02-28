crate::ix!();

#[test] fn DeadCodeElim_BasicElim() {
    todo!();
    /*
      caffe2::NetDef net;
      {
        caffe2::OperatorDef* def = net.add_op();
        def->set_type("Fake");
        def->add_input("X");
        def->add_output("Y");
      }

      auto nn = caffe2::convertToNNModule(net);
      auto pass = caffe2::OptimizationPassRegistry()->Create("DeadCodeElim", &nn);
      pass->run();
      auto optimized_net = caffe2::convertToCaffe2Proto(nn, net);
      EXPECT_EQ(optimized_net.op().size(), 0);
      */
}

#[test] fn DeadCodeElim_BasicNoElim() {
    todo!();
    /*
      caffe2::NetDef net;
      {
        caffe2::OperatorDef* def = net.add_op();
        def->set_type("Fake");
        def->add_input("X");
        def->add_output("Y");
      }
      net.add_external_output("Y");

      auto nn = caffe2::convertToNNModule(net);
      auto pass = caffe2::OptimizationPassRegistry()->Create("DeadCodeElim", &nn);
      pass->run();
      auto optimized_net = caffe2::convertToCaffe2Proto(nn, net);
      EXPECT_EQ(optimized_net.op().size(), 1);
      */
}

#[test] fn DeadCodeElim_PartiallyUsedNoElim() {
    todo!();
    /*
      caffe2::NetDef net;
      {
        caffe2::OperatorDef* def = net.add_op();
        def->set_type("Fake");
        def->add_input("X");
        def->add_output("Y");
        def->add_output("Z");
      }
      net.add_external_output("Y");
      // Z is unused, but we should keep Fake because Y is

      auto nn = caffe2::convertToNNModule(net);
      auto pass = caffe2::OptimizationPassRegistry()->Create("DeadCodeElim", &nn);
      pass->run();
      auto optimized_net = caffe2::convertToCaffe2Proto(nn, net);
      EXPECT_EQ(optimized_net.op().size(), 1);
      */
}
