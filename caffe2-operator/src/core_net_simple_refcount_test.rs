crate::ix!();

use crate::{
    Workspace,
    OperatorStorage,
    OperatorDef,
    CPUContext,
    Operator,
};

/**
  | A net test dummy op that does nothing
  | but scaffolding.
  | 
  | Here, we inherit from OperatorStorage
  | because we instantiate on both CPU and
  | 
  | GPU.
  | 
  | In general, you want to only inherit
  | from Operator<Context>.
  |
  */
pub struct NetSimpleRefCountTestOp {
    //USE_OPERATOR_FUNCTIONS(CPUContext);
    storage: OperatorStorage,
    context: CPUContext,
}

impl NetSimpleRefCountTestOp {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : Operator<CPUContext>(operator_def, ws)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const int32_t& input = OperatorStorage::Input<int32_t>(0);
        int32_t* output = OperatorStorage::Output<int32_t>(0);
        *output = input + 1;
        return true;
        */
    }
}

register_cpu_operator!{NetSimpleRefCountTest, NetSimpleRefCountTestOp}

num_inputs!{NetSimpleRefCountTest, 1}
num_outputs!{NetSimpleRefCountTest, 1}

#[test] fn net_simple_ref_count_test_correctness() {
    todo!();
    /*
      Workspace ws;
      *(ws.CreateBlob("a")->GetMutable<int32_t>()) = 1;
      NetDef net_def;
      net_def.set_type("simple_refcount");
      net_def.add_op()->CopyFrom(
          CreateOperatorDef("NetSimpleRefCountTest", "", {"a"}, {"b"}));
      net_def.add_op()->CopyFrom(
          CreateOperatorDef("NetSimpleRefCountTest", "", {"b"}, {"c"}));
      net_def.add_op()->CopyFrom(
          CreateOperatorDef("NetSimpleRefCountTest", "", {"b"}, {"d"}));
      net_def.add_op()->CopyFrom(
          CreateOperatorDef("NetSimpleRefCountTest", "", {"c"}, {"e"}));
      // After execution, what should look like is:
      // a = 1
      // b = deallocated
      // c = deallocated
      // d = 3
      // e = 4
      std::unique_ptr<NetBase> net(CreateNet(net_def, &ws));
      net->Run();
      // Note on ASSERT vs EXPECT: ASSERT will quit directly if condition not
      // met, which is why we guard IsType<> calls with ASSERT so that the
      // subsequent Get() calls do not product an exception.
      ASSERT_TRUE(ws.GetBlob("a")->IsType<int32_t>());
      EXPECT_EQ(ws.GetBlob("a")->Get<int32_t>(), 1);
      EXPECT_EQ(ws.GetBlob("b")->GetRaw(), nullptr);
      EXPECT_EQ(ws.GetBlob("c")->GetRaw(), nullptr);
      ASSERT_TRUE(ws.GetBlob("d")->IsType<int32_t>());
      EXPECT_EQ(ws.GetBlob("d")->Get<int32_t>(), 3);
      ASSERT_TRUE(ws.GetBlob("e")->IsType<int32_t>());
      EXPECT_EQ(ws.GetBlob("e")->Get<int32_t>(), 4);
  */
}

