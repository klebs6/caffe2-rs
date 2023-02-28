crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/test/vulkan_test.cpp]

#[cfg(not(USE_VULKAN_API))]
mod not_use_vulkan_api {
    use super::*;

    pub fn check_rtol(
            diff:   &Tensor,
            inputs: Vec<Tensor>) -> bool {
        
        todo!();
            /*
                double maxValue = 0.0;
          for (auto& tensor : inputs) {
            maxValue = fmax(tensor.abs().max().item<float>(), maxValue);
          }
          return diff.abs().max().item<float>() < (0.01 + 2e-2 * maxValue);
            */
    }

    pub fn almost_equal(
            a: &Tensor,
            b: &Tensor) -> bool {
        
        todo!();
            /*
                return checkRtol(a - b, {a, b});
            */
    }

    pub fn exactly_equal(
        a: &Tensor,
        b: &Tensor) -> bool {
        
        todo!();
            /*
                return (a - b).abs().max().item<float>() == 0.f;
            */
    }

    #[test] fn vulkan_test_to_cpu() {
        todo!();
        /*
        
          if (!is_vulkan_available())
            return;
          auto t =
              rand({1, 2, 2, 3}, TensorOptions(kCPU).dtype(kFloat));
          auto tv = t.vulkan();
          ASSERT_TRUE(tv.options().device().type() == kVulkan);
          auto t2 = tv.cpu();
          ASSERT_TRUE(t2.options().device().type() == kCPU);
          ASSERT_TRUE(almostEqual(t2, t));

        */
    }

    #[test] fn vulkan_test_upsample_nearest2d() {
        todo!();
        /*
        
          if (!is_vulkan_available())
            return;

          auto t_in =
              rand({1, 2, 2, 3}, TensorOptions(kCPU).dtype(kFloat));
          auto t_out_expected = upsample_nearest2d(t_in, {4, 6});
          auto tv_in =
              t_in.to(TensorOptions{Device{kVulkan}}.dtype(kFloat));

          auto tv_out = upsample_nearest2d(tv_in, {4, 6});
          auto t_out =
              tv_out.to(TensorOptions{Device{kCPU}}.dtype(kFloat));

          bool check = almostEqual(t_out_expected, t_out);
          if (!check) {
            cout << "expected:\n" << t_out_expected << endl;
            cout << "got:\n" << t_out << endl;
          }
          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_test_add() {
        todo!();
        /*
        
          if (!is_vulkan_available())
            return;
          auto t_in0 = rand({1, 2, 2, 3}, device(kCPU).dtype(kFloat));
          auto t_in1 = rand({1, 2, 2, 3}, device(kCPU).dtype(kFloat));
          auto t_out_expected = add(t_in0, t_in1, 2);
          auto tv_in0 = t_in0.vulkan();
          auto tv_in1 = t_in1.vulkan();
          auto tv_out = add(tv_in0, tv_in1, 2);
          auto t_out = tv_out.cpu();

          ASSERT_TRUE(almostEqual(t_out, t_out_expected));

        */
    }

    #[test] fn vulkan_test_add_not4dim() {
        todo!();
        /*
        
          if (!is_vulkan_available())
            return;
          auto t_in0 = rand({1, 1000}, device(kCPU).dtype(kFloat));
          auto t_in1 = rand({1000}, device(kCPU).dtype(kFloat));
          auto t_out_expected = add(t_in0, t_in1, 2);
          auto tv_in0 = t_in0.vulkan();
          auto tv_in1 = t_in1.vulkan();
          auto tv_out = add(tv_in0, tv_in1, 2);
          auto t_out = tv_out.cpu();

          ASSERT_TRUE(almostEqual(t_out, t_out_expected));

        */
    }

    #[test] fn vulkan_test_add_cpu() {
        todo!();
        /*
        
          if (!is_vulkan_available())
            return;
          auto t_in0 = rand({2, 96, 1000}, device(kCPU).dtype(kFloat));
          auto t_in1 =
              rand({1, 2, 96, 1000}, device(kCPU).dtype(kFloat));
          auto t_out_expected = add(t_in0, t_in1, 2);
          auto tv_in0 = t_in0.vulkan();
          auto tv_in1 = t_in1.vulkan();

          auto tv_out1 = add(tv_in0, t_in1, 2);
          auto t_out1 = tv_out1.cpu();
          ASSERT_TRUE(almostEqual(t_out1, t_out_expected));

          auto tv_out2 = add(t_in0, tv_in1, 2);
          auto t_out2 = tv_out2.cpu();
          ASSERT_TRUE(almostEqual(t_out2, t_out_expected));

        */
    }

    #[test] fn vulkan_test_add() {
        todo!();
        /*
        
          if (!is_vulkan_available())
            return;
          auto t_in0 = rand({1, 2, 2, 2}, device(kCPU).dtype(kFloat));
          auto t_in1 = rand({1, 2, 2, 2}, device(kCPU).dtype(kFloat));
          auto tv_in0 = t_in0.vulkan();
          auto tv_in1 = t_in1.vulkan();

          t_in0.add_(t_in1, 2);
          tv_in0.add_(tv_in1, 2);
          auto t_out = tv_in0.cpu();
          bool check = almostEqual(t_out, t_in0);
          if (!check) {
            cout << "expected:\n" << t_in0 << endl;
            cout << "got:\n" << t_out << endl;
          }
          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_test_mul_scalar() {
        todo!();
        /*
        
          if (!is_vulkan_available())
            return;
          auto t_in = rand({3, 2, 2, 3}, device(kCPU).dtype(kFloat));
          const float other = 3.14;
          auto t_out_expected = t_in.mul(other);
          auto tv_in = t_in.vulkan();
          auto tv_out = tv_in.mul(other);
          auto t_out = tv_out.cpu();

          bool check = almostEqual(t_out, t_out_expected);
          if (!check) {
            cout << "expected:\n" << t_out_expected << endl;
            cout << "got:\n" << t_out << endl;
          }
          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_test_add_scalar() {
        todo!();
        /*
        
          if (!is_vulkan_available())
            return;
          auto t_in = rand({3, 2, 2, 3}, device(kCPU).dtype(kFloat));
          float* data = t_in.data_ptr<float>();
          auto numel = t_in.numel();
          for (int i = 0; i < numel; i++) {
            data[i] = i;
          }

          const float other = 3.14;
          const float alpha = 2;
          auto t_out_expected = t_in.add(other, alpha);
          auto tv_in = t_in.vulkan();
          auto tv_out = tv_in.add(other, alpha);
          auto t_out = tv_out.cpu();

          bool check = almostEqual(t_out, t_out_expected);
          if (!check) {
            cout << "expected:\n" << t_out_expected << endl;
            cout << "got:\n" << t_out << endl;
          }
          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_test_conv2d() {
        todo!();
        /*
        
          if (!is_vulkan_available())
            return;
          auto OC = 2;
          auto C = 3;
          i64 H = 3;
          i64 W = 3;
          i64 KH = 2;
          i64 KW = 2;
          auto t_in = rand({1, C, H, W}, device(kCPU).dtype(kFloat));
          auto t_w = rand({OC, C, KH, KW}, device(kCPU).dtype(kFloat));
          auto t_b = zeros({OC}, device(kCPU).dtype(kFloat));
          i64 groups = 1;
          vector<i64> stride{1, 1};
          vector<i64> padding{0, 0};
          vector<i64> dilation{1, 1};

          auto t_out_expected =
              conv2d(t_in, t_w, t_b, stride, padding, dilation, groups);
          auto tv_in = t_in.vulkan();
          auto tv_out = conv2d(tv_in, t_w, t_b, stride, padding, dilation, groups);
          auto t_out = tv_out.cpu();
          bool check = almostEqual(t_out, t_out_expected);
          if (!check) {
            cout << "expected:\n" << t_out_expected << endl;
            cout << "got:\n" << t_out << endl;
          }
          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_test_conv2d_dww_eights_on_cpu() {
        todo!();
        /*
        
          if (!is_vulkan_available())
            return;
          auto C = 3;
          i64 groups = C;
          i64 H = 3;
          i64 W = 3;
          i64 KH = 2;
          i64 KW = 2;
          auto t_in = rand({1, C, H, W}, device(kCPU).dtype(kFloat));
          auto t_w =
              rand({groups, 1, KH, KW}, device(kCPU).dtype(kFloat));
          auto t_b = zeros({groups}, device(kCPU).dtype(kFloat));
          vector<i64> stride{1, 1};
          vector<i64> padding{0, 0};
          vector<i64> dilation{1, 1};
          auto t_out_expected =
              conv2d(t_in, t_w, t_b, stride, padding, dilation, groups);
          auto tv_in = t_in.vulkan();
          auto tv_out = conv2d(tv_in, t_w, t_b, stride, padding, dilation, groups);
          auto t_out = tv_out.cpu();
          bool check = almostEqual(t_out_expected, t_out);
          if (!check) {
            cout << "expected:\n" << t_out_expected << endl;
            cout << "got:\n" << t_out << endl;
          }
          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_test_addmm() {
        todo!();
        /*
        
          if (!is_vulkan_available())
            return;
          auto t_m1 = rand({2, 2}, device(kCPU).dtype(kFloat));
          auto t_m2 = rand({2, 3}, device(kCPU).dtype(kFloat));
          auto t_b = rand({2, 3}, device(kCPU).dtype(kFloat));

          float beta = 100;
          float alpha = 2;
          auto t_out_expected = addmm(t_b, t_m1, t_m2, beta, alpha);

          auto tv_m1 = t_m1.vulkan();
          auto tv_m2 = t_m2.vulkan();
          auto tv_b = t_b.vulkan();
          auto tv_out = addmm(tv_b, tv_m1, tv_m2, beta, alpha);
          auto t_out = tv_out.cpu();
          bool check = almostEqual(t_out_expected, t_out);
          if (!check) {
            cout << "expected:\n" << t_out_expected << endl;
            cout << "got:\n" << t_out << endl;
          }
          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_test_mm() {
        todo!();
        /*
        
          if (!is_vulkan_available())
            return;
          auto t_m1 = rand({10, 20}, device(kCPU).dtype(kFloat));
          auto t_m2 = rand({20, 30}, device(kCPU).dtype(kFloat));

          auto t_out_expected = t_m1.mm(t_m2);

          auto tv_m1 = t_m1.vulkan();
          auto tv_m2 = t_m2.vulkan();
          auto tv_out = tv_m1.mm(tv_m2);
          auto t_out = tv_out.cpu();
          bool check = almostEqual(t_out_expected, t_out);
          if (!check) {
            cout << "expected:\n" << t_out_expected << endl;
            cout << "got:\n" << t_out << endl;
          }
          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_test_clamp() {
        todo!();
        /*
        
          if (!is_vulkan_available())
            return;
          float min = -0.5;
          float max = 0.5;
          auto t_in = rand({1, 3, 16, 16}, device(kCPU).dtype(kFloat));
          auto t_out_expected = clamp(t_in, min, max);

          auto tv_in = t_in.vulkan();
          auto tv_out = clamp(tv_in, min, max);
          auto t_out = tv_out.cpu();

          ASSERT_TRUE(almostEqual(t_out, t_out_expected));

        */
    }

    #[test] fn vulkan_test_hardtanh() {
        todo!();
        /*
        
          if (!is_vulkan_available())
            return;
          float min = -0.5;
          float max = 0.5;
          auto t_in = rand({1, 3, 16, 16}, device(kCPU).dtype(kFloat));
          auto t_out_expected = hardtanh_(t_in, min, max);

          auto tv_in = t_in.vulkan();
          auto tv_out = hardtanh_(tv_in, min, max);
          auto t_out = tv_out.cpu();

          ASSERT_TRUE(almostEqual(t_out, t_out_expected));

        */
    }

    #[test] fn vulkan_test_relu() {
        todo!();
        /*
        
          if (!is_vulkan_available())
            return;
          auto t = empty({1, 2, 2, 2}, device(kCPU).dtype(kFloat));
          auto t_in = t.uniform_(-1, 1);
          auto tv_in = t_in.vulkan();

          t_in.relu_();
          tv_in.relu_();
          auto tv_out = tv_in.cpu();
          bool check = almostEqual(t_in, tv_out);
          if (!check) {
            cout << "expected:\n" << t_in << endl;
            cout << "got:\n" << tv_out << endl;
          }
          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_test_mean() {
        todo!();
        /*
        
          if (!is_vulkan_available())
            return;
          auto t_in = rand({2, 3, 3, 3}, device(kCPU).dtype(kFloat));
          auto t_out_expected = mean(t_in, {2, 3}, false);
          auto tv_in = t_in.vulkan();
          auto tv_out = mean(tv_in, {2, 3}, false);
          auto t_out = tv_out.cpu();
          bool check = almostEqual(t_out_expected, t_out);
          if (!check) {
            cout << "expected:\n" << t_out_expected << endl;
            cout << "got:\n" << t_out << endl;
          }
          ASSERT_TRUE(check);

        */
    }

    pub enum OpType { 
        conv2d, 
        hardtanh_, 
        mean, 
        addmm 
    }

    pub struct BaseOp {
        ty: OpType,
    }

    impl BaseOp {
        
        pub fn new(t: OpType) -> Self {
        
            todo!();
            /*
            : ty(t),

            
            */
        }
    }


    //----------------------------------
    pub trait BaseOpInterface:
    Run
    + ToString {}

    pub trait Run {

        fn run(&mut self, _0: &mut Tensor) -> Tensor;
    }

    pub trait ToString {
        
        fn to_string(&mut self) -> String;
    }


    pub struct Hardtanh_ {
        base: BaseOp,
    }

    impl Default for Hardtanh_ {
        
        fn default() -> Self {
            todo!();
            /*
            : base_op(OpType::hardtanh_),

            
            */
        }
    }

    impl Hardtanh_ {
        
        pub fn run(&mut self, t: &mut Tensor) -> Tensor {
            
            todo!();
            /*
                return hardtanh_(t, 0, 6);
            */
        }
        
        pub fn to_string(&mut self) -> String {
            
            todo!();
            /*
                return "hardtanh_";
            */
        }
    }

    pub struct Mean {
        base: BaseOp,
    }

    impl Default for Mean {
        
        fn default() -> Self {
            todo!();
            /*
            : base_op(OpType::mean),

            
            */
        }
    }

    impl Mean {
        
        pub fn run(&mut self, t: &mut Tensor) -> Tensor {
            
            todo!();
            /*
                return mean(t, {2, 3}, false);
            */
        }
        
        pub fn to_string(&mut self) -> String {
            
            todo!();
            /*
                return "mean";
            */
        }
    }

    pub struct Addmm {
        base:  BaseOp,
        m2:    Tensor,
        m2v:   Tensor,
        b:     Tensor,
        bv:    Tensor,
        beta:  f32,
        alpha: f32,
    }

    impl Addmm {

        pub fn new(
            m1h:   i64,
            m1w:   i64,
            m2w:   i64,
            beta:  f32,
            alpha: f32) -> Self {
        
            todo!();
            /*


                : BaseOp(OpType::addmm), beta(_beta), alpha(_alpha) 

            m2 = rand(
                IntArrayRef({m1W, m2W}), device(kCPU).dtype(kFloat));
            m2v = m2.vulkan();
            b = rand(
                IntArrayRef({m1H, m2W}), device(kCPU).dtype(kFloat));
            bv = b.vulkan();
            */
        }
        
        pub fn run(&mut self, t: &mut Tensor) -> Tensor {
            
            todo!();
            /*
                if (t.is_vulkan()) {
              return addmm(bv, t, m2v, beta, alpha);
            }
            return addmm(b, t, m2, beta, alpha);
            */
        }
        
        pub fn to_string(&mut self) -> String {
            
            todo!();
            /*
                return "addmm";
            */
        }
    }

    pub struct Conv2d {
        base:    BaseOp,
        stride:  i64,
        padding: i64,
        groups:  i64,
        w:       Tensor,
        b:       Tensor,
    }

    impl Conv2d {

        pub fn new(
            wsizes: &[i32],
            g:      i64,
            s:      i64,
            p:      i64) -> Self {
        
            todo!();
            /*
                : BaseOp(OpType::conv2d), stride(s), padding(p), groups(g) 

            w = rand(wsizes, device(kCPU).dtype(kFloat));
            b = zeros(wsizes[0], device(kCPU).dtype(kFloat));
          }{
            */
        }
        
        pub fn run(&mut self, t: &mut Tensor) -> Tensor {
            
            todo!();
            /*
                return conv2d(t, w, b, {stride}, {padding}, {1}, groups);
            */
        }
        
        pub fn to_string(&mut self) -> String {
            
            todo!();
            /*
                return "conv2d";
            */
        }
    }

    pub struct OpsList {
        ops: Vec<Box<BaseOp>>,
    }

    impl OpsList {
        
        pub fn new(ops: &mut Vec<Box<BaseOp>>) -> Self {
        
            todo!();
            /*
            : ops(move(_ops)),
            */
        }
        
        pub fn run_dual(&mut self, 
            in_: &mut Tensor,
            vin: &mut Tensor) -> Auto {
            
            todo!();
            /*
                Tensor t = in;
            Tensor tv = vin;
            int i = 0;
            for (const auto& op : ops) {
              t = op->run(t);
              tv = op->run(tv);
              auto tv_cpu = t.cpu();
              TORCH_INTERNAL_ASSERT(
                  almostEqual(t, tv_cpu),
                  "Not almost equal cpu vs vulkan op i:",
                  i,
                  " ",
                  op->toString());
              i++;
            }
            return make_pair(t, tv);
            */
        }
        
        pub fn run(&mut self, in_: &mut Tensor) -> Auto {
            
            todo!();
            /*
                Tensor t = in;
            int i = 0;
            for (const auto& op : ops) {
              t = op->run(t);
              i++;
            }
            return t;
            */
        }
    }

    pub struct MobileNetV2 {
        base: OpsList,
    }

    impl Default for MobileNetV2 {
        
        fn default() -> Self {
            todo!();
            /*


                ops.emplace_back(new Conv2d({32, 3, 3, 3}, 1, 2, 1));
            ops.emplace_back(new Hardtanh_());
            ops.emplace_back(new Conv2d({32, 1, 3, 3}, 32, 1, 1));
            ops.emplace_back(new Hardtanh_());
            ops.emplace_back(new Conv2d({16, 32, 1, 1}, 1, 1, 0));
            ops.emplace_back(new Conv2d({96, 16, 1, 1}, 1, 1, 0));
            ops.emplace_back(new Hardtanh_());
            ops.emplace_back(new Conv2d({96, 1, 3, 3}, 96, 2, 1));
            ops.emplace_back(new Hardtanh_());
            ops.emplace_back(new Conv2d({24, 96, 1, 1}, 1, 1, 0));
            ops.emplace_back(new Conv2d({144, 24, 1, 1}, 1, 1, 0));
            ops.emplace_back(new Hardtanh_());
            ops.emplace_back(new Conv2d({144, 1, 3, 3}, 144, 1, 1));
            ops.emplace_back(new Hardtanh_());
            ops.emplace_back(new Conv2d({24, 144, 1, 1}, 1, 1, 0));
            ops.emplace_back(new Conv2d({144, 24, 1, 1}, 1, 1, 0));
            ops.emplace_back(new Hardtanh_());
            ops.emplace_back(new Conv2d({144, 1, 3, 3}, 144, 2, 1));
            ops.emplace_back(new Hardtanh_());
            ops.emplace_back(new Conv2d({32, 144, 1, 1}, 1, 1, 0));
            ops.emplace_back(new Conv2d({192, 32, 1, 1}, 1, 1, 0));
            ops.emplace_back(new Hardtanh_());
            ops.emplace_back(new Conv2d({192, 1, 3, 3}, 192, 1, 1));
            ops.emplace_back(new Hardtanh_());
            ops.emplace_back(new Conv2d({32, 192, 1, 1}, 1, 1, 0));
            ops.emplace_back(new Conv2d({192, 32, 1, 1}, 1, 1, 0));
            ops.emplace_back(new Hardtanh_());
            ops.emplace_back(new Conv2d({192, 1, 3, 3}, 192, 1, 1));
            ops.emplace_back(new Hardtanh_());
            ops.emplace_back(new Conv2d({32, 192, 1, 1}, 1, 1, 0));
            ops.emplace_back(new Conv2d({192, 32, 1, 1}, 1, 1, 0));
            ops.emplace_back(new Hardtanh_());
            ops.emplace_back(new Conv2d({192, 1, 3, 3}, 192, 2, 1));
            ops.emplace_back(new Hardtanh_());
            ops.emplace_back(new Conv2d({64, 192, 1, 1}, 1, 1, 0));
            ops.emplace_back(new Conv2d({384, 64, 1, 1}, 1, 1, 0));
            ops.emplace_back(new Hardtanh_());
            ops.emplace_back(new Conv2d({384, 1, 3, 3}, 384, 1, 1));
            ops.emplace_back(new Hardtanh_());
            ops.emplace_back(new Hardtanh_());
            ops.emplace_back(new Conv2d({64, 384, 1, 1}, 1, 1, 0));
            ops.emplace_back(new Conv2d({384, 64, 1, 1}, 1, 1, 0));
            ops.emplace_back(new Hardtanh_());
            ops.emplace_back(new Conv2d({384, 1, 3, 3}, 384, 1, 1));
            ops.emplace_back(new Hardtanh_());
            ops.emplace_back(new Conv2d({64, 384, 1, 1}, 1, 1, 0));
            ops.emplace_back(new Conv2d({384, 64, 1, 1}, 1, 1, 0));
            ops.emplace_back(new Hardtanh_());
            ops.emplace_back(new Conv2d({384, 1, 3, 3}, 384, 1, 1));
            ops.emplace_back(new Hardtanh_());
            ops.emplace_back(new Conv2d({64, 384, 1, 1}, 1, 1, 0));
            ops.emplace_back(new Conv2d({384, 64, 1, 1}, 1, 1, 0));
            ops.emplace_back(new Hardtanh_());
            ops.emplace_back(new Conv2d({384, 1, 3, 3}, 384, 1, 1));
            ops.emplace_back(new Hardtanh_());
            ops.emplace_back(new Conv2d({96, 384, 1, 1}, 1, 1, 0));
            ops.emplace_back(new Conv2d({576, 96, 1, 1}, 1, 1, 0));
            ops.emplace_back(new Hardtanh_());
            ops.emplace_back(new Conv2d({576, 1, 3, 3}, 576, 1, 1));
            ops.emplace_back(new Hardtanh_());
            ops.emplace_back(new Conv2d({96, 576, 1, 1}, 1, 1, 0));
            ops.emplace_back(new Conv2d({576, 96, 1, 1}, 1, 1, 0));
            ops.emplace_back(new Hardtanh_());
            ops.emplace_back(new Conv2d({576, 1, 3, 3}, 576, 1, 1));
            ops.emplace_back(new Hardtanh_());
            ops.emplace_back(new Conv2d({96, 576, 1, 1}, 1, 1, 0));
            ops.emplace_back(new Conv2d({576, 96, 1, 1}, 1, 1, 0));
            ops.emplace_back(new Hardtanh_());
            ops.emplace_back(new Conv2d({576, 1, 3, 3}, 576, 2, 1));
            ops.emplace_back(new Hardtanh_());
            ops.emplace_back(new Conv2d({160, 576, 1, 1}, 1, 1, 0));
            ops.emplace_back(new Conv2d({960, 160, 1, 1}, 1, 1, 0));
            ops.emplace_back(new Hardtanh_());
            ops.emplace_back(new Conv2d({960, 1, 3, 3}, 960, 1, 1));
            ops.emplace_back(new Hardtanh_());
            ops.emplace_back(new Conv2d({160, 960, 1, 1}, 1, 1, 0));
            ops.emplace_back(new Conv2d({960, 160, 1, 1}, 1, 1, 0));
            ops.emplace_back(new Hardtanh_());
            ops.emplace_back(new Conv2d({960, 1, 3, 3}, 960, 1, 1));
            ops.emplace_back(new Hardtanh_());
            ops.emplace_back(new Conv2d({160, 960, 1, 1}, 1, 1, 0));
            ops.emplace_back(new Conv2d({960, 160, 1, 1}, 1, 1, 0));
            ops.emplace_back(new Hardtanh_());
            ops.emplace_back(new Conv2d({960, 1, 3, 3}, 960, 1, 1));
            ops.emplace_back(new Hardtanh_());
            ops.emplace_back(new Conv2d({320, 960, 1, 1}, 1, 1, 0));
            ops.emplace_back(new Conv2d({1280, 320, 1, 1}, 1, 1, 0));
            ops.emplace_back(new Hardtanh_());
            ops.emplace_back(new Mean());
            ops.emplace_back(new Addmm(1, 1280, 1000, 0, 1));
            */
        }
    }

    #[test] fn vulkan_test_disabled_mobilenetv2() {
        todo!();
        /*
        
          if (!is_vulkan_available())
            return;

          MobileNetV2 mn2{};
          auto t_in =
              rand({1, 3, 224, 224}, device(kCPU).dtype(kFloat));
          auto tv_in = t_in.vulkan();
          mn2.runDual(t_in, tv_in);

        */
    }

    #[test] fn vulkan_test_ops_list() {
        todo!();
        /*
        
          if (!is_vulkan_available())
            return;

          vector<unique_ptr<BaseOp>> ops;
          ops.emplace_back(new Conv2d({32, 3, 3, 3}, 1, 2, 1));
          ops.emplace_back(new Hardtanh_());
          ops.emplace_back(new Conv2d({32, 1, 3, 3}, 32, 1, 1));
          ops.emplace_back(new Hardtanh_());
          ops.emplace_back(new Conv2d({16, 32, 1, 1}, 1, 1, 0));
          ops.emplace_back(new Conv2d({96, 16, 1, 1}, 1, 1, 0));
          ops.emplace_back(new Hardtanh_());
          ops.emplace_back(new Conv2d({96, 1, 3, 3}, 96, 2, 1));
          ops.emplace_back(new Hardtanh_());
          ops.emplace_back(new Conv2d({24, 96, 1, 1}, 1, 1, 0));
          ops.emplace_back(new Conv2d({144, 24, 1, 1}, 1, 1, 0)); // 1, 144, 56, 56
          ops.emplace_back(new Hardtanh_());
          ops.emplace_back(new Mean());
          ops.emplace_back(new Addmm(1, 144, 1000, 0, 1));
          OpsList opsList(ops);
          auto t_in =
              rand({1, 3, 224, 224}, device(kCPU).dtype(kFloat));
          auto t_out_expected = opsList.run(t_in);

          auto tv_in = t_in.vulkan();

          auto tv_out = opsList.run(t_in);
          auto t_out = tv_out.cpu();

          ASSERT_TRUE(almostEqual(t_out, t_out_expected));

        */
    }

    #[inline] pub fn make_stack<Inputs>(inputs: Inputs) -> Vec<IValue> {

        todo!();
            /*
                return {forward<Inputs>(inputs)...};
            */
    }

    #[inline] pub fn call_op_by_handle<Args>(
        op:   &OperatorHandle,
        args: Args) -> Vec<IValue> {

        todo!();
            /*
                auto stack = makeStack(forward<Args>(args)...);
          Dispatcher::singleton().callBoxed(op, &stack);
          return stack;
            */
    }

    #[inline] pub fn call_op_by_name<Args>(
        func_name:     *const u8,
        overload_name: *const u8,
        args:          Args) -> Vec<IValue> {

        todo!();
            /*
                const optional<OperatorHandle> op_handle =
              Dispatcher::singleton().findSchema({func_name, overload_name});
          assert(op_handle.has_value());
          return callOpByHandle(op_handle.value(), forward<Args>(args)...);
            */
    }

    #[test] fn vulkan_test_conv2d_prepack() {
        todo!();
        /*
        
          if (!is_vulkan_available())
            return;
          auto OC = 2;
          auto C = 3;
          i64 groups = 1;
          auto t_in = rand({1, C, 3, 3}, device(kCPU).dtype(kFloat));
          auto t_w = rand({OC, C, 2, 2}, device(kCPU).dtype(kFloat));
          auto t_b = zeros({OC}, device(kCPU).dtype(kFloat));

          vector<i64> stride{1, 1};
          vector<i64> padding{0, 0};
          vector<i64> dilation{1, 1};
          float output_min = 0.25;
          float output_max = 1.0;

          auto t_out_conv2d =
              conv2d(t_in, t_w, t_b, stride, padding, dilation, groups);
          auto t_out_expected = clamp(t_out_conv2d, output_min, output_max);

          auto tv_in = t_in.vulkan();
          auto tv_out_conv2d =
              conv2d(tv_in, t_w, t_b, stride, padding, dilation, groups);
          auto tv_out = clamp(tv_out_conv2d, output_min, output_max);

          auto t_out = tv_out.cpu();
          bool no_prepack_check = almostEqual(t_out, t_out_expected);
          if (!no_prepack_check) {
            cout << "t_out_expected:\n" << t_out_expected << endl;
            cout << "t_out:\n" << t_out << endl;
          }
          ASSERT_TRUE(no_prepack_check);

          auto prepack = callOpByName(
              "vulkan_prepack::conv2d_clamp_prepack",
              "",
              t_w,
              t_b,
              stride,
              padding,
              dilation,
              groups,
              output_min,
              output_max);
          auto tv_out_prepack_ivalues =
              callOpByName("vulkan_prepack::conv2d_clamp_run", "", tv_in, prepack[0]);
          auto tv_out_prepack = tv_out_prepack_ivalues[0].toTensor();
          auto t_out_prepack = tv_out_prepack.cpu();
          const auto prepack_check = almostEqual(t_out_prepack, t_out_expected);
          if (!prepack_check) {
            cout << "expected:\n" << t_out_expected << endl;
            cout << "got:\n" << t_out_prepack << endl;
          }
          ASSERT_TRUE(prepack_check);

        */
    }

    #[test] fn vulkan_test_adaptive_avg_pool2d() {
        todo!();
        /*
        
          if (!is_vulkan_available())
            return;

          auto t_in =
              rand({1, 2, 7, 7}, TensorOptions(kCPU).dtype(kFloat));
          auto t_out_expected = adaptive_avg_pool2d(t_in, {3, 3});
          auto tv_in = t_in.vulkan();

          auto tv_out = adaptive_avg_pool2d(tv_in, {3, 3});
          auto t_out = tv_out.cpu();

          const auto check = almostEqual(t_out, t_out_expected);
          if (!check) {
            cout << "expected:" << t_out_expected << endl;
            cout << "got:" << t_out << endl;
          }
          ASSERT_TRUE(check);

        */
    }

    /**
      | TODO: Enable when view operator for
      | Vulkan landed
      |
      */
    #[test] fn vulkan_test_disabled_adaptive_avg_pool2d_2() {
        todo!();
        /*
        
          if (!is_vulkan_available())
            return;

          auto t_in =
              rand({1, 1280, 7, 7}, TensorOptions(kCPU).dtype(kFloat));
          auto t_out_expected = adaptive_avg_pool2d(t_in, {1, 1});
          auto tv_in = t_in.vulkan();

          auto tv_out = adaptive_avg_pool2d(tv_in, {1, 1});
          auto t_out = tv_out.cpu();

          const auto check = almostEqual(t_out, t_out_expected);
          if (!check) {
            cout << "expected:" << t_out_expected << endl;
            cout << "got:" << t_out << endl;
          }
          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_test_reshape() {
        todo!();
        /*
        
          if (!is_vulkan_available())
            return;

          auto t_in =
              rand({1, 8, 1, 1}, TensorOptions(kCPU).dtype(kFloat));
          auto t_out_expected = reshape(t_in, {1, 8});
          auto tv_in = t_in.vulkan();
          auto tv_out = reshape(tv_in, {1, 8});
          auto t_out = tv_out.cpu();

          const auto check = almostEqual(t_out, t_out_expected);
          if (!check) {
            cout << "expected:" << t_out_expected << endl;
            cout << "got:" << t_out << endl;
          }
          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_test_reshape2() {
        todo!();
        /*
        
          if (!is_vulkan_available())
            return;

          auto t_in =
              rand({1, 3, 2, 2}, TensorOptions(kCPU).dtype(kFloat));
          auto t_out_expected = reshape(t_in, {2, 3, 1, 2});

          auto tv_in = t_in.vulkan();
          auto tv_out = reshape(tv_in, {2, 3, 1, 2});
          auto t_out = tv_out.cpu();

          const auto check = almostEqual(t_out, t_out_expected);
          if (!check) {
            cout << "expected:" << t_out_expected << endl;
            cout << "got:" << t_out << endl;
          }
          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_test_tensor5d() {
        todo!();
        /*
        
          if (!is_vulkan_available())
            return;

          auto t_in =
              rand({2, 2, 2, 3, 3}, TensorOptions(kCPU).dtype(kFloat));
          auto tv_in = t_in.vulkan();

        */
    }

    #[test] fn vulkan_test_tensor5d_transpose() {
        todo!();
        /*
        
          if (!is_vulkan_available())
            return;

          auto t_in =
              empty({1, 2, 3, 2, 1}, TensorOptions(kCPU).dtype(kFloat));
          float* data = t_in.data_ptr<float>();
          auto numel = t_in.numel();
          for (int i = 0; i < numel; i++) {
            data[i] = i;
          }

          auto tv_in = t_in.vulkan();

          auto t_out_expected = t_in.transpose(1, 2);
          auto t_out = tv_in.transpose(1, 2).cpu();
          const auto check = almostEqual(t_out, t_out_expected);
          if (!check) {
            cout << "expected:" << t_out_expected << endl;
            cout << "got:" << t_out << endl;
          }
          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_test_view() {
        todo!();
        /*
        
          if (!is_vulkan_available())
            return;

          auto t_in =
              rand({2, 4, 3, 3}, TensorOptions(kCPU).dtype(kFloat));
          auto t_out_expected = t_in.view({2, 2, 2, 3, 3});
          auto tv_in = t_in.vulkan();
          auto tv_out = tv_in.view({2, 2, 2, 3, 3});
          auto t_out = tv_out.cpu();

          const auto check = almostEqual(t_out, t_out_expected);
          if (!check) {
            cout << "expected:" << t_out_expected << endl;
            cout << "got:" << t_out << endl;
          }
          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_test_slice() {
        todo!();
        /*
        
          if (!is_vulkan_available())
            return;

          auto t_in =
              empty({1, 4, 2, 2}, TensorOptions(kCPU).dtype(kFloat));
          float* data = t_in.data_ptr<float>();
          auto numel = t_in.numel();
          for (int i = 0; i < numel; i++) {
            data[i] = i;
          }

          auto tv_in = t_in.vulkan();

          auto t_out_expected = t_in.slice(1, 2, 4, 1);
          auto t_out = tv_in.slice(1, 2, 4, 1).cpu();
          const auto check = almostEqual(t_out, t_out_expected);
          if (!check) {
            cout << "expected:" << t_out_expected << endl;
            cout << "got:" << t_out << endl;
          }
          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_test_select() {
        todo!();
        /*
        
          if (!is_vulkan_available())
            return;

          auto t_in =
              empty({1, 4, 2, 2}, TensorOptions(kCPU).dtype(kFloat));
          float* data = t_in.data_ptr<float>();
          auto numel = t_in.numel();
          for (int i = 0; i < numel; i++) {
            data[i] = i;
          }

          auto tv_in = t_in.vulkan();

          auto t_out_expected = t_in.slice(1, 1);
          auto t_out = tv_in.slice(1, 1).cpu();
          const auto check = almostEqual(t_out, t_out_expected);
          if (!check) {
            cout << "expected:" << t_out_expected << endl;
            cout << "got:" << t_out << endl;
          }
          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_test_unsqueeze() {
        todo!();
        /*
        
          if (!is_vulkan_available())
            return;

          auto t_in =
              empty({1, 2, 2}, TensorOptions(kCPU).dtype(kFloat));
          float* data = t_in.data_ptr<float>();
          auto numel = t_in.numel();
          for (int i = 0; i < numel; i++) {
            data[i] = i;
          }

          auto tv_in = t_in.vulkan();

          auto t_out_expected = t_in.unsqueeze(1);
          auto t_out = tv_in.unsqueeze(1).cpu();
          const auto check = almostEqual(t_out, t_out_expected);
          if (!check) {
            cout << "expected:" << t_out_expected << endl;
            cout << "got:" << t_out << endl;
          }
          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_test_cat() {
        todo!();
        /*
        
          if (!is_vulkan_available())
            return;

          auto t_in0 =
              rand({1, 1, 3, 3}, TensorOptions(kCPU).dtype(kFloat));
          auto t_in1 =
              rand({1, 2, 3, 3}, TensorOptions(kCPU).dtype(kFloat));
          auto t_in2 =
              rand({1, 5, 3, 3}, TensorOptions(kCPU).dtype(kFloat));

          auto t_out_expected = cat({t_in0, t_in1, t_in2}, 1);
          auto tv_out = cat({t_in0.vulkan(), t_in1.vulkan(), t_in2.vulkan()}, 1);
          auto t_out = tv_out.cpu();

          const auto check = almostEqual(t_out, t_out_expected);
          if (!check) {
            cout << "expected:" << t_out_expected << endl;
            cout << "got:" << t_out << endl;
          }
          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_test_disabled_max_pool2d() {
        todo!();
        /*
        
          if (!is_vulkan_available())
            return;

          auto t_in =
              rand({1, 3, 7, 7}, TensorOptions(kCPU).dtype(kFloat));
          auto t_out_expected = max_pool2d(t_in, {2, 2}, {1}, {0}, {1});
          auto tv_in = t_in.vulkan();

          auto tv_out = max_pool2d(tv_in, {2, 2}, {1}, {0}, {1});
          auto t_out = tv_out.cpu();

          const auto check = almostEqual(t_out, t_out_expected);
          if (!check) {
            cout << "expected:" << t_out_expected << endl;
            cout << "got:" << t_out << endl;
          }
          ASSERT_TRUE(check);

        */
    }

    #[test] fn vulkan_test_avg_pool2d() {
        todo!();
        /*
        
          if (!is_vulkan_available())
            return;

          auto t_in =
              rand({1, 3, 7, 7}, TensorOptions(kCPU).dtype(kFloat));
          auto t_out_expected = avg_pool2d(t_in, {2, 2}, {1}, {0}, true);
          auto tv_in = t_in.vulkan();

          auto tv_out = avg_pool2d(tv_in, {2, 2}, {1}, {0}, true);
          auto t_out = tv_out.cpu();

          const auto check = almostEqual(t_out, t_out_expected);
          if (!check) {
            cout << "expected:" << t_out_expected << endl;
            cout << "got:" << t_out << endl;
          }
          ASSERT_TRUE(check);

        */
    }
}
