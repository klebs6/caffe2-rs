crate::ix!();

pub struct TestParams {
    n:           i32,
    c:           i32,
    h:           i32,
    w:           i32,
    n_rois:      i32,
    rois_array:  Vec<f32>,
}

#[inline] pub fn create_and_run<Context>(
    out_result:  *mut TensorCPU,
    order:       &String,
    test_params: &TestParams,
    random_test: bool) {
    todo!();
    /*
        Workspace ws;
      Context context;

      if (random_test) {
        const int N = test_params.N;
        const int C = test_params.C;
        const int H = test_params.H;
        const int W = test_params.W;
        vector<float> features(N * C * H * W);
        std::iota(features.begin(), features.end(), 0);
        // utils::AsEArrXt(features) /= features.size();
        AddInput<Context>(vector<int64_t>{N, C, H, W}, features, "X", &ws);
        const int n_rois = test_params.n_rois;
        const vector<float>& rois = test_params.rois_array;
        AddInput<Context>(vector<int64_t>{n_rois, 5}, rois, "R", &ws);
      } else {
        const int N = 2;
        const int C = 3;
        const int H = 100;
        const int W = 110;
        vector<float> features(N * C * H * W);
        std::iota(features.begin(), features.end(), 0);
        // utils::AsEArrXt(features) /= features.size();
        AddInput<Context>(vector<int64_t>{N, C, H, W}, features, "X", &ws);
        vector<float> rois{0, 0,            0,            79,           59,
                           0, 0,            5.0005703f,   52.63237f,    43.69501495f,
                           0, 24.13628387f, 7.51243401f,  79,           46.06628418f,
                           0, 0,            7.50924301f,  68.47792816f, 46.03357315f,
                           0, 0,            23.09477997f, 51.61448669f, 59,
                           0, 0,            39.52141571f, 52.44710541f, 59,
                           0, 23.57396317f, 29.98791885f, 79,           59,
                           0, 0,            41.90219116f, 79,           59,
                           0, 0,            23.30098343f, 79,           59};
        AddInput<Context>(vector<int64_t>{9, 5}, rois, "R", &ws);
      }

      std::vector<unique_ptr<OperatorStorage>> ops;
      EXPECT_TRUE(order == "NCHW" || order == "NHWC");
      if (order == "NCHW") {
        OperatorDef def;
        def.set_name("test");
        def.set_type("RoIAlign");
        def.add_input("X");
        def.add_input("R");
        def.add_output("Y");
        def.mutable_device_option()->set_device_type(
            TypeToProto(GetDeviceType<Context>()));
        def.add_arg()->CopyFrom(MakeArgument("spatial_scale", 1.0f / 16.0f));
        def.add_arg()->CopyFrom(MakeArgument("pooled_h", 6));
        def.add_arg()->CopyFrom(MakeArgument("pooled_w", 8));
        def.add_arg()->CopyFrom(MakeArgument("sampling_ratio", 2));

        ops.push_back(CreateOperator(def, &ws));
      } else if (order == "NHWC") {
        OperatorDef def_roialign;
        def_roialign.set_name("test");
        def_roialign.set_type("RoIAlign");
        def_roialign.add_input("X_NHWC");
        def_roialign.add_input("R");
        def_roialign.add_output("Y_NHWC");
        def_roialign.mutable_device_option()->set_device_type(
            TypeToProto(GetDeviceType<Context>()));
        def_roialign.add_arg()->CopyFrom(
            MakeArgument("spatial_scale", 1.0f / 16.0f));
        def_roialign.add_arg()->CopyFrom(MakeArgument("pooled_h", 6));
        def_roialign.add_arg()->CopyFrom(MakeArgument("pooled_w", 8));
        def_roialign.add_arg()->CopyFrom(MakeArgument("sampling_ratio", 2));
        def_roialign.add_arg()->CopyFrom(MakeArgument<string>("order", "NHWC"));

        OperatorDef def_x;
        def_x.set_name("test_x");
        def_x.set_type("NCHW2NHWC");
        def_x.add_input("X");
        def_x.add_output("X_NHWC");
        def_x.mutable_device_option()->set_device_type(
            TypeToProto(GetDeviceType<Context>()));

        OperatorDef def_y;
        def_y.set_name("test_y");
        def_y.set_type("NHWC2NCHW");
        def_y.add_input("Y_NHWC");
        def_y.add_output("Y");
        def_y.mutable_device_option()->set_device_type(
            TypeToProto(GetDeviceType<Context>()));

        ops.push_back(CreateOperator(def_x, &ws));
        ops.push_back(CreateOperator(def_roialign, &ws));
        ops.push_back(CreateOperator(def_y, &ws));
      }

      for (auto const& op : ops) {
        EXPECT_NE(nullptr, op.get());
        EXPECT_TRUE(op->Run());
      }

      Blob* Y_blob = ws.GetBlob("Y");
      EXPECT_NE(nullptr, Y_blob);

      auto& Y = Y_blob->Get<Tensor>();
      outResult->CopyFrom(Y);
    */
}
