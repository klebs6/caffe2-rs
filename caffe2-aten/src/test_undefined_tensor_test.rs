crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/test/undefined_tensor_test.cpp]

#[test] fn test_undefined() {
    todo!();
    /*
    
      manual_seed(123);

      // mainly test ops on undefined tensors don't segfault and give a reasonable errror message.
      Tensor und;
      Tensor ft = ones({1}, CPU(kFloat));

      stringstream ss;
      ss << und << endl;
      ASSERT_FALSE(und.defined());
      ASSERT_EQ(string("UndefinedType"), und.toString());

      ASSERT_ANY_THROW(und.strides());
      ASSERT_EQ(und.dim(), 1);
      ASSERT_ANY_THROW([]() { return Tensor(); }() = Scalar(5));
      ASSERT_ANY_THROW(und.add(und));
      ASSERT_ANY_THROW(und.add(ft));
      ASSERT_ANY_THROW(ft.add(und));
      ASSERT_ANY_THROW(und.add(5));
      ASSERT_ANY_THROW(und.mm(und));

      // public variable API
      ASSERT_ANY_THROW(und.variable_data());
      ASSERT_ANY_THROW(und.tensor_data());
      ASSERT_ANY_THROW(und.is_view());
      ASSERT_ANY_THROW(und._base());
      ASSERT_ANY_THROW(und.name());
      ASSERT_ANY_THROW(und.grad_fn());
      ASSERT_ANY_THROW(und.remove_hook(0));
      ASSERT_ANY_THROW(und.register_hook([](const Tensor& x) -> Tensor { return x; }));

      // copy_
      ASSERT_ANY_THROW(und.copy_(und));
      ASSERT_ANY_THROW(und.copy_(ft));
      ASSERT_ANY_THROW(ft.copy_(und));

      ASSERT_ANY_THROW(und.toBackend(Backend::CPU));
      ASSERT_ANY_THROW(ft.toBackend(Backend::Undefined));

      Tensor to_move = ones({1}, CPU(kFloat));
      Tensor m(move(to_move));
      ASSERT_FALSE(to_move.defined());
      ASSERT_EQ(to_move.unsafeGetTensorImpl(), UndefinedTensorImpl::singleton());

    */
}
