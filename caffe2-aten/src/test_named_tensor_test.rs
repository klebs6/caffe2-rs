crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/test/NamedTensor_test.cpp]

pub fn dimname_from_string(str_: &String) -> Dimname {
    
    todo!();
        /*
            return Dimname::fromSymbol(Symbol::dimname(str));
        */
}

#[test] fn named_tensor_test_is() {
    todo!();
    /*
    
      auto tensor = zeros({3, 2, 5, 7});
      ASSERT_FALSE(tensor.has_names());

      tensor = zeros({3, 2, 5, 7});
      ASSERT_FALSE(tensor.has_names());

      tensor = zeros({3, 2, 5, 7});
      auto N = dimnameFromString("N");
      auto C = dimnameFromString("C");
      auto H = dimnameFromString("H");
      auto W = dimnameFromString("W");
      vector<Dimname> names = { N, C, H, W };
      internal_set_names_inplace(tensor, names);
      ASSERT_TRUE(tensor.has_names());

    */
}

pub fn dimnames_equal(
        names: &[Dimname],
        other: &[Dimname]) -> bool {
    
    todo!();
        /*
            if (names.size() != other.size()) {
        return false;
      }
      for (const auto i : irange(names.size())) {
        const auto& name = names[i];
        const auto& other_name = other[i];
        if (name.type() != other_name.type() || name.symbol() != other_name.symbol()) {
          return false;
        }
      }
      return true;
        */
}

#[test] fn named_tensor_test_attach_metadata() {
    todo!();
    /*
    
      auto tensor = zeros({3, 2, 5, 7});
      auto N = dimnameFromString("N");
      auto C = dimnameFromString("C");
      auto H = dimnameFromString("H");
      auto W = dimnameFromString("W");
      vector<Dimname> names = { N, C, H, W };

      internal_set_names_inplace(tensor, names);

      const auto retrieved_meta = tensor.get_named_tensor_meta();
      ASSERT_TRUE(dimnames_equal(retrieved_meta->names(), names));

      // Test dropping metadata
      tensor.unsafeGetTensorImpl()->set_named_tensor_meta(nullptr);
      ASSERT_FALSE(tensor.has_names());

    */
}

#[test] fn named_tensor_test_internal_set_names_inplace() {
    todo!();
    /*
    
      auto tensor = zeros({3, 2, 5, 7});
      auto N = dimnameFromString("N");
      auto C = dimnameFromString("C");
      auto H = dimnameFromString("H");
      auto W = dimnameFromString("W");
      vector<Dimname> names = { N, C, H, W };
      ASSERT_FALSE(tensor.has_names());

      // Set names
      internal_set_names_inplace(tensor, names);
      const auto retrieved_names = tensor.opt_names().value();
      ASSERT_TRUE(dimnames_equal(retrieved_names, names));

      // Drop names
      internal_set_names_inplace(tensor, nullopt);
      ASSERT_TRUE(tensor.get_named_tensor_meta() == nullptr);
      ASSERT_TRUE(tensor.opt_names() == nullopt);

    */
}

#[test] fn named_tensor_test_empty() {
    todo!();
    /*
    
      auto N = Dimname::fromSymbol(Symbol::dimname("N"));
      auto C = Dimname::fromSymbol(Symbol::dimname("C"));
      auto H = Dimname::fromSymbol(Symbol::dimname("H"));
      auto W = Dimname::fromSymbol(Symbol::dimname("W"));
      vector<Dimname> names = { N, C, H, W };

      auto tensor = empty({});
      ASSERT_EQ(tensor.opt_names(), nullopt);

      tensor = empty({1, 2, 3});
      ASSERT_EQ(tensor.opt_names(), nullopt);

      tensor = empty({1, 2, 3, 4}, names);
      ASSERT_TRUE(dimnames_equal(tensor.opt_names().value(), names));

      ASSERT_THROW(empty({1, 2, 3}, names), Error);

    */
}

#[test] fn named_tensor_test_dimname_to_position() {
    todo!();
    /*
    
      auto N = dimnameFromString("N");
      auto C = dimnameFromString("C");
      auto H = dimnameFromString("H");
      auto W = dimnameFromString("W");
      vector<Dimname> names = { N, C, H, W };

      auto tensor = empty({1, 1, 1});
      ASSERT_THROW(dimname_to_position(tensor, N), Error);

      tensor = empty({1, 1, 1, 1}, names);
      ASSERT_EQ(dimname_to_position(tensor, H), 2);

    */
}

pub fn tensornames_unify_from_right(
        names:       &[Dimname],
        other_names: &[Dimname]) -> Vec<Dimname> {
    
    todo!();
        /*
            auto names_wrapper = namedinference::TensorNames(names);
      auto other_wrapper = namedinference::TensorNames(other_names);
      return names_wrapper.unifyFromRightInplace(other_wrapper).toDimnameVec();
        */
}

pub fn check_unify(
        names:       &[Dimname],
        other_names: &[Dimname],
        expected:    &[Dimname])  {
    
    todo!();
        /*
            // Check legacy unify_from_right
      const auto result = unify_from_right(names, other_names);
      ASSERT_TRUE(dimnames_equal(result, expected));

      // Check with TensorNames::unifyFromRight.
      // In the future we'll merge unify_from_right and
      // TensorNames::unifyFromRight, but for now, let's test them both.
      const auto also_result = tensornames_unify_from_right(names, other_names);
      ASSERT_TRUE(dimnames_equal(also_result, expected));
        */
}

pub fn check_unify_error(
        names:       &[Dimname],
        other_names: &[Dimname])  {
    
    todo!();
        /*
            // In the future we'll merge unify_from_right and
      // TensorNames::unifyFromRight. For now, test them both.
      ASSERT_THROW(unify_from_right(names, other_names), Error);
      ASSERT_THROW(tensornames_unify_from_right(names, other_names), Error);
        */
}

#[test] fn named_tensor_test_unify_from_right() {
    todo!();
    /*
    
      auto N = dimnameFromString("N");
      auto C = dimnameFromString("C");
      auto H = dimnameFromString("H");
      auto W = dimnameFromString("W");
      auto None = dimnameFromString("*");

      vector<Dimname> names = { N, C };

      check_unify({ N, C, H, W }, { N, C, H, W }, { N, C, H, W });
      check_unify({ W }, { C, H, W }, { C, H, W });
      check_unify({ None, W }, { C, H, W }, { C, H, W });
      check_unify({ None, None, H, None }, { C, None, W }, { None, C, H, W });

      check_unify_error({ W, H }, { W, C });
      check_unify_error({ W, H }, { C, H });
      check_unify_error({ None, H }, { H, None });
      check_unify_error({ H, None, C }, { H });

    */
}

#[test] fn named_tensor_test_alias() {
    todo!();
    /*
    
      // tensor.alias is not exposed in Python so we test its name propagation here
      auto N = dimnameFromString("N");
      auto C = dimnameFromString("C");
      vector<Dimname> names = { N, C };

      auto tensor = empty({2, 3}, vector<Dimname>{ N, C });
      auto aliased = tensor.alias();
      ASSERT_TRUE(dimnames_equal(tensor.opt_names().value(), aliased.opt_names().value()));

    */
}

#[test] fn named_tensor_test_no_names_guard() {
    todo!();
    /*
    
      auto N = dimnameFromString("N");
      auto C = dimnameFromString("C");
      vector<Dimname> names = { N, C };

      auto tensor = empty({2, 3}, names);
      ASSERT_TRUE(NamesMode::is_enabled());
      {
        NoNamesGuard guard;
        ASSERT_FALSE(NamesMode::is_enabled());
        ASSERT_FALSE(tensor.opt_names());
        ASSERT_FALSE(get_opt_names(tensor.unsafeGetTensorImpl()));
      }
      ASSERT_TRUE(NamesMode::is_enabled());

    */
}

pub fn nchw() -> Vec<Dimname> {
    
    todo!();
        /*
            auto N = dimnameFromString("N");
      auto C = dimnameFromString("C");
      auto H = dimnameFromString("H");
      auto W = dimnameFromString("W");
      return { N, C, H, W };
        */
}

#[test] fn named_tensor_test_name_print() {
    todo!();
    /*
    
      auto names = nchw();
      {
        auto N = TensorName(names, 0);
        ASSERT_EQ(
            str(N),
            "'N' (index 0 of ['N', 'C', 'H', 'W'])");
      }
      {
        auto H = TensorName(names, 2);
        ASSERT_EQ(
            str(H),
            "'H' (index 2 of ['N', 'C', 'H', 'W'])");
      }

    */
}

#[test] fn named_tensor_test_names_check_unique() {
    todo!();
    /*
    
      auto names = nchw();
      {
        // smoke test to check that this doesn't throw
        TensorNames(names).checkUnique("op_name");
      }
      {
        vector<Dimname> nchh = { names[0], names[1], names[2], names[2] };
        auto tensornames = TensorNames(nchh);
        ASSERT_THROW(tensornames.checkUnique("op_name"), Error);
      }

    */
}
