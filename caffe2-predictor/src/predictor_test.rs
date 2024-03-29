crate::ix!();


pub const predictSpec: &'static str = "
    name: \"predict\"
    type: \"dag\"
    external_input: \"data\"
    external_input: \"W\"
    external_input: \"b\"
    external_output: \"y\"
    op {
      input: \"data\"
      input: \"W\"
      input: \"b\"
      output: \"y\"
      type: \"FC\"
    }
";

pub const initSpec: &'static str = "
    name: \"init\"
    type: \"dag\"
    op {
      type: \"ConstantFill\"
      output: \"W\"
      arg {
        name: \"shape\"
        ints: 10
        ints: 4
      }
      arg {
        name: \"value\"
        f: 2.0
      }
    }
    op {
      type: \"ConstantFill\"
      output: \"b\"
      arg {
        name: \"shape\"
        ints: 10
      }
      arg {
        name: \"value\"
        f: 2.0
      }
    }
";

pub const metaSpec: &'static str = "
  blobs {
    key: \"INPUTS_BLOB_TYPE\"
    value: \"data\"
  }
  blobs {
      key: \"OUTPUTS_BLOB_TYPE\"
      value: \"y\"
  }
  nets {
    key: \"GLOBAL_INIT_NET_TYPE\"
    value: {
      name: \"init\"
      type: \"dag\"
      op {
        type: \"ConstantFill\"
        output: \"data\"
        arg {
          name: \"shape\"
          ints: 1
          ints: 4
        }
        arg {
          name: \"value\"
          f: 2.0
        }
      }
      op {
        type: \"ConstantFill\"
        output: \"W\"
        arg {
          name: \"shape\"
          ints: 10
          ints: 4
        }
        arg {
          name: \"value\"
          f: 2.0
        }
      }
      op {
        type: \"ConstantFill\"
        output: \"b\"
        arg {
          name: \"shape\"
          ints: 10
        }
        arg {
          name: \"value\"
          f: 2.0
        }
      }
    }
  }
  nets {
    key: \"PREDICT_NET_TYPE\"
    value: {
      name: \"predict\"
      type: \"dag\"
      external_input: \"data\"
      external_input: \"W\"
      external_input: \"b\"
      external_output: \"y\"
      op {
        input: \"data\"
        input: \"W\"
        input: \"b\"
        output: \"y\"
        type: \"FC\"
      }
    }
  }
";

#[inline] pub fn random_tensor(dims: &Vec<i64>, ctx: *mut CPUContext) -> Box<Blob> {
    
    todo!();
    /*
        auto blob = make_unique<Blob>();
      auto* t = BlobGetMutableTensor(blob.get(), CPU);
      t->Resize(dims);
      math::RandUniform<float, CPUContext>(
          t->numel(), -1.0, 1.0, t->template mutable_data<float>(), ctx);
      return blob;
    */
}

#[inline] pub fn parse_net_def(value: &String) -> NetDef {
    
    todo!();
    /*
        NetDef def;
      CAFFE_ENFORCE(
          TextFormat::ParseFromString(value, &def),
          "Failed to parse NetDef with value: ",
          value);
      return def;
    */
}

#[inline] pub fn parse_meta_net_def(value: &String) -> MetaNetDef {
    
    todo!();
    /*
        MetaNetDef def;
      CAFFE_ENFORCE(
          TextFormat::ParseFromString(value, &def),
          "Failed to parse NetDef with value: ",
          value);
      return def;
    */
}

pub struct PredictorTest<T> {
    base: Test<T>,
    ctx:  Box<CPUContext>,
    p:    Box<Predictor>,
}

impl<T> PredictorTest<T> {
    
    #[inline] pub fn set_up(&mut self)  {
        
        todo!();
        /*
            DeviceOption op;
        op.set_random_seed(1701);
        ctx_ = std::make_unique<CPUContext>(op);
        NetDef init, run;
        p_ = std::make_unique<Predictor>(
            makePredictorConfig(parseNetDef(initSpec), parseNetDef(predictSpec)));
        */
    }
}

#[test] fn predictor_test_simple_batch_sized() {
    todo!();
    /*
      auto inputData = randomTensor({1, 4}, ctx_.get());
      Predictor::&[Tensor] input;
      auto tensor = BlobGetMutableTensor(inputData.get(), CPU);
      input.emplace_back(tensor->Alias());
      Predictor::&[Tensor] output;
      (*p_)(input, &output);
      EXPECT_EQ(output.size(), 1);
      EXPECT_EQ(output.front().sizes().size(), 2);
      EXPECT_EQ(output.front().size(0), 1);
      EXPECT_EQ(output.front().size(1), 10);
      EXPECT_NEAR(output.front().data<float>()[4], 4.9556, 1E-4);
  */
}

#[test] fn predictor_test_simple_batch_sized_map_input() {
    todo!();
    /*
      auto inputData = randomTensor({1, 4}, ctx_.get());
      Predictor::TensorMap input;
      auto tensor = BlobGetMutableTensor(inputData.get(), CPU);
      input.emplace("data", tensor->Alias());

      Predictor::&[Tensor] output;
      (*p_)(input, &output);
      EXPECT_EQ(output.size(), 1);
      EXPECT_EQ(output.front().sizes().size(), 2);
      EXPECT_EQ(output.front().size(0), 1);
      EXPECT_EQ(output.front().size(1), 10);
      EXPECT_NEAR(output.front().data<float>()[4], 4.9556, 1E-4);
  */
}

