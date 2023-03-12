crate::ix!();

#[inline] pub fn relative_error(a: f32, b: f32) -> f32 {
    
    todo!();
    /*
        return std::abs(a - b) / (0.5f * (std::abs(a) + std::abs(b)));
    */
}

#[inline] pub fn compare(
    n:                            i32,
    inputC:                       i32,
    h:                            i32,
    w:                            i32,
    outputC:                      i32,
    kernelH:                      i32,
    kernelW:                      i32,
    strideH:                      i32,
    strideW:                      i32,
    padT:                         i32,
    padL:                         i32,
    padB:                         i32,
    padR:                         i32,
    adjH:                         i32,
    adjW:                         i32,
    max_rel_err:                  f32,
    abs_err_for_rel_err_failure:  f32)  
{
    todo!();
    /*
        LOG(INFO) <<
        "running N " << N << " inputC " << inputC << " H " << H << " W " << W <<
        " outputC " << outputC <<
        " kernelH " << kernelH << " kernelW " << kernelW <<
        " strideH " << strideH << " strideW " << strideW <<
        " padT " << padT << " padL " << padL <<
        " padB " << padB << " padR " << padR <<
        " adjH " << adjH << " adjW " << adjW;

      Workspace ws;

      OperatorDef def1;
      def1.set_name("test");
      def1.set_type("ConvTranspose");
      def1.set_engine("MOBILE");
      def1.add_input("X");
      def1.add_input("W");
      def1.add_input("B");
      def1.add_output("Y1");

      def1.add_arg()->CopyFrom(MakeArgument("kernel_h", kernelH));
      def1.add_arg()->CopyFrom(MakeArgument("kernel_w", kernelW));
      def1.add_arg()->CopyFrom(MakeArgument("stride_h", strideH));
      def1.add_arg()->CopyFrom(MakeArgument("stride_w", strideW));
      def1.add_arg()->CopyFrom(MakeArgument("pad_t", padT));
      def1.add_arg()->CopyFrom(MakeArgument("pad_l", padL));
      def1.add_arg()->CopyFrom(MakeArgument("pad_b", padB));
      def1.add_arg()->CopyFrom(MakeArgument("pad_r", padR));
      def1.add_arg()->CopyFrom(MakeArgument("adj_h", adjH));
      def1.add_arg()->CopyFrom(MakeArgument("adj_w", adjW));

      AddNoiseInput(vector<int64_t>{N, inputC, H, W}, "X", &ws);
      AddNoiseInput(vector<int64_t>{inputC, outputC, kernelH, kernelW}, "W", &ws);
      AddNoiseInput(vector<int64_t>{outputC}, "B", &ws);

      unique_ptr<OperatorStorage> op1(CreateOperator(def1, &ws));
      EXPECT_NE(nullptr, op1.get());

      OperatorDef def2;
      def2.set_name("test");
      def2.set_type("ConvTranspose");
      def2.add_input("X");
      def2.add_input("W");
      def2.add_input("B");
      def2.add_output("Y2");

      def2.add_arg()->CopyFrom(MakeArgument("kernel_h", kernelH));
      def2.add_arg()->CopyFrom(MakeArgument("kernel_w", kernelW));
      def2.add_arg()->CopyFrom(MakeArgument("stride_h", strideH));
      def2.add_arg()->CopyFrom(MakeArgument("stride_w", strideW));
      def2.add_arg()->CopyFrom(MakeArgument("pad_t", padT));
      def2.add_arg()->CopyFrom(MakeArgument("pad_l", padL));
      def2.add_arg()->CopyFrom(MakeArgument("pad_b", padB));
      def2.add_arg()->CopyFrom(MakeArgument("pad_r", padR));
      def2.add_arg()->CopyFrom(MakeArgument("adj_h", adjH));
      def2.add_arg()->CopyFrom(MakeArgument("adj_w", adjW));

      unique_ptr<OperatorStorage> op2(CreateOperator(def2, &ws));
      EXPECT_NE(nullptr, op2.get());

      EXPECT_TRUE(op1->Run());
      Blob* Y1blob = ws.GetBlob("Y1");
      EXPECT_NE(nullptr, Y1blob);
      auto& Y1 = Y1blob->Get<TensorCPU>();

      EXPECT_TRUE(op2->Run());
      Blob* Y2blob = ws.GetBlob("Y2");
      EXPECT_NE(nullptr, Y2blob);
      auto& Y2 = Y2blob->Get<TensorCPU>();

      // Compare all output points
      for (int n = 0; n < Y1.dim32(0); ++n) {
        for (int c = 0; c < Y1.dim32(1); ++c) {
          for (int h = 0; h < Y1.dim32(2); ++h) {
            for (int w = 0; w < Y1.dim32(3); ++w) {
              int offset =
                n * Y1.dim32(1) * Y1.dim32(2) * Y1.dim32(3) +
                c * Y1.dim32(2) * Y1.dim32(3) +
                h * Y1.dim32(3) +
                w;

              auto v1 = Y1.data<float>()[offset];
              auto v2 = Y2.data<float>()[offset];

              float relErr = relativeError(v1, v2);
              float absErr = std::abs(v1 - v2);

              // For small values / small difference, the relative error
              // can be huge but the absolute error will be small
              EXPECT_TRUE(relErr <= maxRelErr ||
                          (absErr <= absErrForRelErrFailure)) <<
                v1 << " " << v2 << " (rel err " << relErr << ") " <<
                "(" << n << " " << c << " " << h << " " << w << ") " <<
                "running N " << N << " inputC " << inputC <<
                " H " << H << " W " << W <<
                " outputC " << outputC <<
                " kernelH " << kernelH << " kernelW " << kernelW <<
                " strideH " << strideH << " strideW " << strideW <<
                " padT " << padT << " padL " << padL <<
                " padB " << padB << " padR " << padR <<
                " adjH " << adjH << " adjW " << adjW;

            }
          }
        }
      }
    */
}
