crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/test/mobile_memory_cleanup.cpp]

#[cfg(USE_XNNPACK)]
#[test] fn memory_clean_up_no_error_without_release() {
    todo!();
    /*
    
      Module m("m");
      m.register_parameter("weight", Torchones({20, 1, 5, 5}), false);
      m.register_parameter("bias", Torchones({20}), false);
      m.define(R"(
        def forward(self, input):
          return torch._convolution(input, self.weight, self.bias, [1, 1], [0, 0], [1, 1], False, [0, 0], 1, False, False, True, True)
      )");
      m.eval();
      auto m_optimized = optimizeForMobile(m);
      stringstream ss;
      EXPECT_NO_THROW(m_optimized.save(ss));

    */
}

#[cfg(USE_XNNPACK)]
#[test] fn memory_clean_up_unpack_error() {
    todo!();
    /*
    
      globalContext().setReleaseWeightsWhenPrepacking(true);
      Module m("m");
      m.register_parameter("weight", Torchones({20, 1, 5, 5}), false);
      m.register_parameter("bias", Torchones({20}), false);
      m.define(R"(
        def forward(self, input):
          return torch._convolution(input, self.weight, self.bias, [1, 1], [0, 0], [1, 1], False, [0, 0], 1, False, False, True, True)
      )");
      m.eval();
      auto m_optimized = optimizeForMobile(m);
      stringstream ss;
      EXPECT_ANY_THROW(m_optimized.save(ss));

    */
}
