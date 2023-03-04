crate::ix!();

/**
  | for quantized Add, the error shouldn't
  | exceed 2 * scale
  |
  */
#[inline] pub fn add_error_tolerance(scale: f32) -> f32 {
    
    todo!();
    /*
        return 2 * scale;
    */
}

#[inline] pub fn q(dims: &Vec<i64>) -> Box<Int8TensorCPU> {
    
    todo!();
    /*
        auto r = std::make_unique<int8::Int8TensorCPU>();
      r->scale = 0.01;
      r->zero_point = static_cast<int32_t>(uint8_t::max) / 2;
      ReinitializeTensor(&r->t, dims, at::dtype<uint8_t>().device(CPU));
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_int_distribution<uint32_t> dis{
          0, uint8_t::max};
      for (auto i = 0; i < r->t.numel(); ++i) {
        r->t.mutable_data<uint8_t>()[i] = dis(gen);
      }
      return r;
    */
}

#[inline] pub fn biasq(dims: &Vec<i64>, scale: f64) -> Box<Int8TensorCPU> {
    
    todo!();
    /*
        auto r = std::make_unique<int8::Int8TensorCPU>();
      r->scale = scale;
      r->zero_point = 0;
      r->t.Resize(dims);
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<float> dis(-1, 1);
      for (auto i = 0; i < r->t.numel(); ++i) {
        r->t.mutable_data<int32_t>()[i] =
            static_cast<int32_t>(dis(gen) / scale + r->zero_point);
      }
      return r;
    */
}

#[inline] pub fn dq(xQ: &Int8TensorCPU) -> Box<TensorCPU> {
    
    todo!();
    /*
        auto r = std::make_unique<Tensor>(CPU);
      r->Resize(XQ.t.sizes());
      for (auto i = 0; i < r->numel(); ++i) {
        r->mutable_data<float>()[i] =
            (static_cast<int32_t>(XQ.t.data<uint8_t>()[i]) - XQ.zero_point) *
            XQ.scale;
      }
      return r;
    */
}

#[inline] pub fn biasdq(xQ: &Int8TensorCPU) -> Box<TensorCPU> {
    
    todo!();
    /*
        auto r = std::make_unique<Tensor>(CPU);
      r->Resize(XQ.t.sizes());
      for (auto i = 0; i < r->numel(); ++i) {
        r->mutable_data<float>()[i] =
            (XQ.t.data<int32_t>()[i] - XQ.zero_point) * XQ.scale;
      }
      return r;
    */
}


#[macro_export] macro_rules! expect_tensor_eq {
    ($_YA:ident, $_YE:ident) => {
        /*
        EXPECT_TRUE((_YA).sizes() == (_YE).sizes());                       
               for (auto i = 0; i < (_YA).numel(); ++i) {                         
                   EXPECT_FLOAT_EQ((_YA).data<float>()[i], (_YE).data<float>()[i]); 
               }                                                                  
        */
    }
}

#[macro_export] macro_rules! expect_tensor_approx_eq {
    ($_YA:ident, $_YE:ident, $_tol:ident) => {
        /*
        EXPECT_TRUE((_YA).sizes() == (_YE).sizes());                           
        for (auto i = 0; i < (_YA).numel(); ++i) {                             
            EXPECT_NEAR((_YA).data<float>()[i], (_YE).data<float>()[i], (_tol)); 
        }                                                                      
        */
    }
}

#[inline] pub fn int_8copy(dst: *mut Int8TensorCPU, src: &Int8TensorCPU)  {
    
    todo!();
    /*
        dst->zero_point = src.zero_point;
      dst->scale = src.scale;
      dst->t.CopyFrom(src.t);
    */
}


#[inline] pub fn add_input(
    shape:  &Vec<i64>,
    values: &Vec<f32>,
    name:   &String,
    ws:     *mut Workspace)  
{
    
    todo!();
    /*
        // auto* t = ws->CreateBlob(name)->GetMutable<TensorCPU>();
      auto t = std::make_unique<Tensor>(CPU);
      t->Resize(shape);
      std::copy(values.begin(), values.end(), t->mutable_data<float>());
      BlobGetMutableTensor(ws->CreateBlob(name), CPU)->CopyFrom(*t);
    */
}

#[inline] pub fn random_int(a: i32, b: i32) -> i32 {
    
    todo!();
    /*
        static std::random_device rd;
      static std::mt19937 gen(rd());
      return std::uniform_int_distribution<int>(a, b)(gen);
    */
}
