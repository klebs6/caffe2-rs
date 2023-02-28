crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/test/ivalue_test.cpp]

/// Snippets for checking assembly.
pub fn inspect_tuple_construction() -> IValue {
    
    todo!();
        /*
            tuple<string, string> s = make_tuple(
          "abcdefghijklmnopqrstuvwxyz", "ABCDEFGHIJKLMNOPQRSTUVWXYZ");
      return IValue(s);
        */
}

#[test] fn i_value_test_basic() {
    todo!();
    /*
    
      List<i64> foo({3, 4, 5});
      ASSERT_EQ(foo.use_count(), 1);
      IValue bar{foo};
      ASSERT_EQ(foo.use_count(), 2);
      auto baz = bar;
      ASSERT_EQ(foo.use_count(), 3);
      auto foo2 = move(bar);
      ASSERT_EQ(foo.use_count(), 3);
      ASSERT_TRUE(foo2.isIntList());
      ASSERT_TRUE(bar.isNone());
      foo2 = IValue(4.0);
      ASSERT_TRUE(foo2.isDouble());
      ASSERT_EQ(foo2.toDouble(), 4.0);
      ASSERT_EQ(foo.use_count(), 2);
      ASSERT_TRUE(baz.toIntVector() == vector<i64>({3, 4, 5}));

      auto move_it = move(baz).toIntList();
      ASSERT_EQ(foo.use_count(), 2);
      ASSERT_TRUE(baz.isNone());
      IValue i(4);
      ASSERT_TRUE(i.isInt());
      ASSERT_EQ(i.toInt(), 4);
      IValue dlist(List<double>({3.5}));
      ASSERT_TRUE(dlist.isDoubleList());
      ASSERT_TRUE(dlist.toDoubleVector() == vector<double>({3.5}));
      move(dlist).toDoubleList();
      ASSERT_TRUE(dlist.isNone());
      dlist = IValue(List<double>({3.4}));
      ASSERT_TRUE(dlist.toDoubleVector() == vector<double>({3.4}));
      IValue the_list(
          Tuple::create({IValue(3.4), IValue(4), IValue(foo)}));
      ASSERT_EQ(foo.use_count(), 3);
      ASSERT_TRUE(the_list.isTuple());
      auto first = the_list.toTuple()->elements()[1];
      ASSERT_EQ(first.toInt(), 4);
      Tensor tv = rand({3, 4});
      IValue ten(tv);
      ASSERT_EQ(tv.use_count(), 2);
      auto ten2 = ten;
      ASSERT_EQ(tv.use_count(), 3);
      ASSERT_TRUE(ten2.toTensor().equal(ten.toTensor()));
      move(ten2).toTensor();
      ASSERT_EQ(tv.use_count(), 2);

      auto elem1 = complex<double>(3, 4);
      auto elem2 = complex<double>(3, -4);
      auto elem3 = complex<double>(5, 0);
      List<complex<double>> foo1({elem1, elem2, elem3});
      ASSERT_EQ(foo1.use_count(), 1);
      IValue bar1{foo1};
      ASSERT_EQ(foo1.use_count(), 2);
      auto baz1 = bar1;
      ASSERT_EQ(foo1.use_count(), 3);
      auto foo12 = move(bar1);
      ASSERT_EQ(foo1.use_count(), 3);
      ASSERT_TRUE(foo12.isComplexDoubleList());
      ASSERT_EQ(foo12.toComplexDoubleList(), foo1);

      ASSERT_TRUE(bar1.isNone());
      auto foo3 = IValue(complex<double>(3, 4));
      ASSERT_TRUE(foo3.isComplexDouble());
      ASSERT_EQ(foo3.toComplexDouble(), complex<double>(3,4));

      ASSERT_TRUE(baz1.toComplexDoubleVector() == vector<complex<double>>({elem1, elem2, elem3}));
      IValue complex_tuple(
          Tuple::create({IValue(complex<double>(3.4, 4.7)), IValue(foo1)}));
      ASSERT_TRUE(complex_tuple.isTuple());
      ASSERT_EQ(complex_tuple.toTuple()->elements()[0].toComplexDouble(), complex<double>(3.4, 4.7));
      ASSERT_EQ(complex_tuple.toTuple()->elements()[1], foo1);

    */
}

#[test] fn i_value_test_complex_dict() {
    todo!();
    /*
    
      typedef complex<double> c_type;
      Dict<c_type, c_type> m;
      auto num1 = c_type(2.3, -3.5);
      auto num2 = c_type(0, 5);
      m.insert(num1, 2 * num1);
      m.insert(num2, 2 * num2);
      IValue dict(move(m));
      auto m_ = dict.toGenericDict();
      ASSERT_EQ(m_.at(num1), 2 * num1);
      ASSERT_EQ(m_.at(num2), 2 * num2);

    */
}

pub fn make_sample_ivalues() -> [IValue; 5] {
    
    todo!();
        /*
            return { rand({3, 4}), "hello", 42, true, 1.5 };
        */
}

pub fn make_more_sample_ivalues() -> [IValue; 5] {
    
    todo!();
        /*
            return { rand({3, 4}), "goodbye", 23, false, 0.5 };
        */
}

/// IValue::operator== doesn't seem to work on Tensors.
#[macro_export] macro_rules! expect_ivalue_eq {
    ($a:ident, $b:ident) => {
        /*
        
          EXPECT_EQ((a).isTensor(), (b).isTensor());            
          if ((a).isTensor()) {                                 
            EXPECT_TRUE(a.toTensor().equal(b.toTensor()));      
          } else {                                              
            EXPECT_EQ(a, b);                                    
          }
        */
    }
}

#[test] fn i_value_test_swap() {
    todo!();
    /*
    
      // swap() has the following 3 cases: tensor, intrusive_ptr, or
      // neither. Exercise all pairs of the three.

      auto sampleInputs = makeSampleIValues();
      auto sampleTargets = makeMoreSampleIValues();
      for (const auto& input: sampleInputs) {
        for (const auto& target: sampleTargets) {
          IValue a(input);
          IValue b(target);
          EXPECT_IVALUE_EQ(a, input);
          EXPECT_IVALUE_EQ(b, target);
          a.swap(b);
          EXPECT_IVALUE_EQ(a, target);
          EXPECT_IVALUE_EQ(b, input);
        }
      }

    */
}

#[test] fn i_value_test_copy_construct() {
    todo!();
    /*
    
      auto sampleInputs = makeSampleIValues();
      for (const IValue& v: sampleInputs) {
        IValue copy(v);
        EXPECT_IVALUE_EQ(copy, v);
      }

    */
}

#[test] fn i_value_test_move_construct() {
    todo!();
    /*
    
      auto sampleInputs = makeSampleIValues();
      for (const IValue& v: sampleInputs) {
        IValue source(v);
        IValue target(move(source));
        EXPECT_IVALUE_EQ(target, v);
        EXPECT_TRUE(source.isNone());
      }

    */
}

#[test] fn i_value_test_copy_assign() {
    todo!();
    /*
    
      auto sampleInputs = makeSampleIValues();
      auto sampleTargets = makeMoreSampleIValues();

      for (const IValue& input: sampleInputs) {
        for (const IValue& target: sampleTargets) {
          IValue copyTo(target);
          IValue copyFrom(input);
          copyTo = copyFrom;
          EXPECT_IVALUE_EQ(copyTo, input);
          EXPECT_IVALUE_EQ(copyFrom, input);
          EXPECT_IVALUE_EQ(copyTo, copyFrom);
        }
      }

    */
}

#[test] fn i_value_test_move_assign() {
    todo!();
    /*
    
      auto sampleInputs = makeSampleIValues();
      auto sampleTargets = makeMoreSampleIValues();

      for (const IValue& input: sampleInputs) {
        for (const IValue& target: sampleTargets) {
          IValue moveTo(target);
          IValue moveFrom(input);
          moveTo = move(moveFrom);
          EXPECT_IVALUE_EQ(moveTo, input);
          EXPECT_TRUE(moveFrom.isNone());
        }
      }

    */
}

#[test] fn i_value_test_tuple() {
    todo!();
    /*
    
      tuple<i64, Tensor> t = make_tuple(123, randn({1}));
      auto iv = IValue(t);
      auto t_ = iv.to<tuple<i64, Tensor>>();
      ASSERT_EQ(get<0>(t_), 123);
      ASSERT_EQ(
          get<1>(t_).item().to<float>(), get<1>(t).item().to<float>());

    */
}

#[test] fn i_value_test_unsafe_remove_attr() {
    todo!();
    /*
    
      auto cu = make_shared<CompilationUnit>();
      auto cls = ClassType::create("foo.bar", cu);
      cls->addAttribute("attr1", TensorType::get());
      cls->addAttribute("attr2", TensorType::get());
      auto obj = Object::create(
          StrongTypePtr(cu, cls), cls->numAttributes());
      obj->unsafeRemoveAttr("attr1");
      // attr1 is not removed in the type
      ASSERT_TRUE(cls->hasAttribute("attr1"));
      ASSERT_TRUE(cls->hasAttribute("attr2"));
      ASSERT_TRUE(obj->slots().size() == 1);

    */
}

#[test] fn i_value_test_tuple_print() {
    todo!();
    /*
    
      {
        IValue tp = make_tuple(3);

        stringstream ss;
        ss << tp;
        ASSERT_EQ(ss.str(), "(3,)");
      }

      {
        IValue tp = make_tuple(3, 3);
        stringstream ss;
        ss << tp;
        ASSERT_EQ(ss.str(), "(3, 3)");
      }

    */
}

#[test] fn i_value_test_complex_ivalue_print() {
    todo!();
    /*
    
      {
        IValue complex(complex<double>(2, -3));
        stringstream ss;
        ss << complex;
        ASSERT_EQ(ss.str(), "2.-3.j");
      }

      {
        IValue complex(complex<double>(2, 0));
        stringstream ss;
        ss << complex;
        ASSERT_EQ(ss.str(), "2.+0.j");
      }

      {
        IValue complex(complex<double>(0, 3));
        stringstream ss;
        ss << complex;
        ASSERT_EQ(ss.str(), "0.+3.j");
      }

    */
}

#[test] fn i_value_test_complex() {
    todo!();
    /*
    
      auto c = complex<double>(2, 3);
      auto c_ = complex<double>(2, -3);
      IValue c1(c), c2(c_), c3{Scalar(c)};

      ASSERT_TRUE(c1.isComplexDouble());
      ASSERT_TRUE(c3.isComplexDouble());

      ASSERT_EQ(c, c1.toComplexDouble());
      ASSERT_FALSE(c1 == c2);
      ASSERT_TRUE(c1 == c3);

      ASSERT_TRUE(c1.isScalar());
      ASSERT_TRUE(c2.toScalar().equal(c_));

    */
}

#[test] fn i_value_test_basic_future() {
    todo!();
    /*
    
      auto f1 = make_intrusive<Future>(IntType::get());
      ASSERT_FALSE(f1->completed());

      f1->markCompleted(IValue(42));
      ASSERT_TRUE(f1->completed());
      ASSERT_EQ(42, f1->value().toInt());
      IValue iv(f1);
      ASSERT_EQ(42, iv.toFuture()->value().toInt());

    */
}

#[test] fn i_value_test_future_callbacks() {
    todo!();
    /*
    
      auto f2 = make_intrusive<Future>(IntType::get());
      int calledTimesA = 0;
      int calledTimesB = 0;
      f2->addCallback([&calledTimesA](Future& f2) {
        ASSERT_TRUE(f2.completed());
        ASSERT_EQ(f2.value().toInt(), 43);
        ++calledTimesA;
      });
      f2->markCompleted(IValue(43));
      ASSERT_EQ(calledTimesA, 1);
      ASSERT_EQ(calledTimesB, 0);
      // Post-markCompleted()
      f2->addCallback([&calledTimesB](Future& f2) {
        ASSERT_TRUE(f2.completed());
        ASSERT_EQ(f2.value().toInt(), 43);
        ++calledTimesB;
      });
      ASSERT_EQ(calledTimesA, 1);
      ASSERT_EQ(calledTimesB, 1);
      ASSERT_FALSE(f2->hasError());

    */
}

#[test] fn i_value_test_future_exceptions() {
    todo!();
    /*
    
      auto f3 = make_intrusive<Future>(IntType::get());
      int calledTimes = 0;
      f3->addCallback([&calledTimes](Future& f3) {
        ASSERT_TRUE(f3.completed());
        try {
          (void)f3.value();
        } catch (const exception& e) {
          if (string(e.what()) == "My Error") {
            ++calledTimes;
          }
        }
      });
      Future::FutureError err("My Error");
      f3->setError(make_exception_ptr(err));
      ASSERT_EQ(calledTimes, 1);
      ASSERT_TRUE(f3->hasError());
      ASSERT_EQ(f3->tryRetrieveErrorMessage(), string("My Error"));

    */
}

#[test] fn i_value_test_future_set_error() {
    todo!();
    /*
    
      auto f1 = make_intrusive<Future>(IntType::get());
      f1->setError(make_exception_ptr(runtime_error("foo")));
      try {
        f1->setError(make_exception_ptr(runtime_error("bar")));
        FAIL() << "Expected to throw";
      } catch (exception& e) {
        EXPECT_THAT(e.what(), ::testing::HasSubstr("Error already set"));
        EXPECT_THAT(e.what(), ::testing::HasSubstr("foo"));
        EXPECT_THAT(e.what(), ::testing::HasSubstr("bar"));
      }

    */
}

#[test] fn i_value_test_equality() {
    todo!();
    /*
    
      EXPECT_EQ(IValue("asdf"), IValue("asdf"));
      EXPECT_NE(IValue("asdf"), IValue("ASDF"));
      EXPECT_NE(IValue("2"), IValue(2));
      EXPECT_EQ(IValue(1), IValue(1));

      // Check the equals() variant that returns an IValue
      auto res = IValue("asdf").equals("asdf");
      EXPECT_TRUE(res.isBool());
      EXPECT_TRUE(res.toBool());

      res = IValue("asdf").equals(1);
      EXPECT_TRUE(res.isBool());
      EXPECT_FALSE(res.toBool());

    */
}

#[test] fn i_value_test_tensor_equality() {
    todo!();
    /*
    
      auto rawTensor = Torchzeros({2, 3});
      auto rawTensorCopy = rawTensor.clone();
      auto t = IValue(rawTensor);
      auto tCopy = IValue(rawTensorCopy);

      // This should throw, because elementwise equality is ambiguous for
      // multi-element Tensors.
      auto testEquality = []() {
        return IValue(Torchones({2, 3})) == IValue(Torchrand({2, 3}));
      };
      EXPECT_ANY_THROW(testEquality());

      // equals() should return a tensor of all `true`.
      IValue eqTensor = t.equals(tCopy);
      EXPECT_TRUE(eqTensor.isTensor());
      auto booleanTrue = Torchones({2, 3}).to(TorchkBool);
      EXPECT_TRUE(eqTensor.toTensor().equal(booleanTrue));

      // Test identity checking
      EXPECT_TRUE(t.is(t));
      EXPECT_FALSE(t.is(tCopy));
      IValue tReference = t;
      EXPECT_TRUE(t.is(tReference));

    */
}

#[test] fn i_value_test_list_equality() {
    todo!();
    /*
    
      IValue c1 = vector<i64>{0, 1, 2, 3};
      IValue c2 = vector<i64>{0, 1, 2, 3};
      IValue c3 = vector<i64>{0, 1, 2, 3, 4};
      EXPECT_EQ(c1, c1);
      EXPECT_EQ(c1, c2);
      EXPECT_FALSE(c1.is(c2));
      EXPECT_NE(c1, c3);
      EXPECT_NE(c2, c3);

    */
}

#[test] fn i_value_test_dict_equality() {
    todo!();
    /*
    
      auto innerDict = Dict<string, string>();
      innerDict.insert("foo", "bar");

      auto d1 = Dict<string, Dict<string, string>>();
      d1.insert("one", innerDict);
      d1.insert("two", innerDict);
      d1.insert("three", innerDict);
      auto c1 = IValue(d1);

      auto d2 = Dict<string, Dict<string, string>>();
      d2.insert("one", innerDict.copy());
      d2.insert("two", innerDict.copy());
      d2.insert("three", innerDict.copy());
      auto c2 = IValue(d2);

      auto d3 = Dict<string, Dict<string, string>>();
      d3.insert("one", innerDict.copy());
      d3.insert("two", innerDict.copy());
      d3.insert("three", innerDict.copy());
      d3.insert("four", innerDict.copy());
      auto c3 = IValue(d3);

      auto d4 = Dict<string, Dict<string, string>>();
      d4.insert("one", innerDict.copy());
      d4.insert("two", innerDict.copy());
      auto innerDictNotEqual = Dict<string, string>();
      innerDictNotEqual.insert("bar", "foo");
      d4.insert("three", innerDictNotEqual);
      auto c4 = IValue(d4);

      EXPECT_EQ(c1, c1);
      EXPECT_EQ(c1, c2);
      EXPECT_FALSE(c1.is(c2));
      EXPECT_NE(c1, c3);
      EXPECT_NE(c2, c3);
      EXPECT_NE(c1, c4);
      EXPECT_NE(c2, c4);

    */
}

#[test] fn i_value_test_dict_equality_different_order() {
    todo!();
    /*
    
      auto d1 = Dict<string, i64>();
      d1.insert("one", 1);
      d1.insert("two", 2);
      auto d2 = Dict<string, i64>();
      d2.insert("two", 2);
      d2.insert("one", 1);

      EXPECT_EQ(d1, d2);

    */
}

#[test] fn i_value_test_list_nested_equality() {
    todo!();
    /*
    
      IValue c1 = vector<vector<i64>>({{0}, {0, 1}, {0, 1, 2}});
      IValue c2 = vector<vector<i64>>({{0}, {0, 1}, {0, 1, 2}});
      IValue c3 = vector<vector<i64>>({{1}, {0, 1}, {0, 1, 2}});
      EXPECT_EQ(c1, c1);
      EXPECT_EQ(c1, c2);
      EXPECT_NE(c1, c3);
      EXPECT_NE(c2, c3);

    */
}

#[test] fn i_value_test_stream_equality() {
    todo!();
    /*
    
      Device device1 =  Device(kCUDA, 0);
      Device device2 = Device(kCUDA, 1);
      Stream stream1 = Stream(Stream::Default::DEFAULT, device1);
      Stream stream2 = Stream(Stream::Default::DEFAULT, device2);
      IValue lhs(stream1);
      IValue rhs_different(stream2);
      IValue rhs_same(stream1);
      EXPECT_FALSE(lhs.equals(rhs_different).toBool());
      EXPECT_TRUE(lhs.equals(rhs_same).toBool());

    */
}

#[test] fn i_value_test_enum_equality() {
    todo!();
    /*
    
      auto cu = make_shared<CompilationUnit>();
      IValue int_ivalue_1(1);
      IValue int_ivalue_2(2);
      IValue str_ivalue_1("1");
      auto int_enum_type1 = EnumType::create(
          "enum_class_1",
          IntType::get(),
          {{"enum_name_1", int_ivalue_1}, {"enum_name_2", int_ivalue_2}},
          cu);
      auto int_enum_type2 = EnumType::create(
          "enum_class_2",
          IntType::get(),
          {{"enum_name_1", int_ivalue_1}, {"enum_name_2", int_ivalue_2}},
          cu);
      auto string_enum_type = EnumType::create(
          "enum_class_3", StringType::get(), {{"enum_name_1", str_ivalue_1}}, cu);

      EXPECT_EQ(
          IValue(make_intrusive<EnumHolder>(
              int_enum_type1, "enum_name_1", int_ivalue_1)),
          IValue(make_intrusive<EnumHolder>(
              int_enum_type1, "enum_name_1", int_ivalue_1))
      );

      EXPECT_NE(
          IValue(make_intrusive<EnumHolder>(
              int_enum_type1, "enum_name_1", int_ivalue_1)),
          IValue(make_intrusive<EnumHolder>(
              int_enum_type2, "enum_name_1", int_ivalue_1))
      );

      EXPECT_NE(
          IValue(make_intrusive<EnumHolder>(
              int_enum_type1, "enum_name_1", int_ivalue_1)),
          IValue(make_intrusive<EnumHolder>(
              int_enum_type1, "enum_name_2", int_ivalue_2))
      );

      EXPECT_NE(
          IValue(make_intrusive<EnumHolder>(
              int_enum_type1, "enum_name_1", int_ivalue_1)),
          IValue(make_intrusive<EnumHolder>(
              string_enum_type, "enum_name_1", str_ivalue_1))
      );

    */
}

#[test] fn i_value_test_is_ptr_type() {
    todo!();
    /*
    
      IValue tensor(rand({3, 4}));
      IValue undefinedTensor((Tensor()));
      IValue integer(42);
      IValue str("hello");

      EXPECT_TRUE(tensor.isPtrType());
      EXPECT_FALSE(undefinedTensor.isPtrType());
      EXPECT_FALSE(integer.isPtrType());
      EXPECT_TRUE(str.isPtrType());

    */
}

#[test] fn i_value_test_is_alias_of() {
    todo!();
    /*
    
      auto sampleIValues = makeSampleIValues();
      for (auto& iv: sampleIValues) {
        for (auto& iv2: sampleIValues) {
          if (&iv == &iv2 && iv.isPtrType()) {
            EXPECT_TRUE(iv.isAliasOf(iv2));
          } else {
            EXPECT_FALSE(iv.isAliasOf(iv2));
          }
        }
      }

    */
}

#[test] fn i_value_test_internal_to_pointer() {
    todo!();
    /*
    
      IValue tensor(rand({3, 4}));
      IValue str("hello");

      EXPECT_EQ(tensor.internalToPointer(), tensor.unsafeToTensorImpl());
      EXPECT_NE(str.internalToPointer(), nullptr);

      IValue nullStr((intrusive_ptr<ConstantString>()));
      ASSERT_TRUE(nullStr.isString());
      EXPECT_EQ(nullStr.internalToPointer(), nullptr);

    */
}

#[test] fn i_value_test_identity_comparison_and_hashing() {
    todo!();
    /*
    
      Tensor t1 = rand({3, 4});
      Tensor t2 = rand({3, 4});
      IValue tv1(t1), tv2(t2);
      IValue tv1b(t1);

      EXPECT_EQ(tv1.hash(), tv1b.hash());
      EXPECT_NE(tv1.hash(), tv2.hash());

      EXPECT_TRUE(tv1.is(tv1));
      EXPECT_TRUE(tv1.is(tv1b));
      EXPECT_TRUE(tv1b.is(tv1));
      EXPECT_TRUE(tv2.is(tv2));

      EXPECT_FALSE(tv1.is(tv2));
      EXPECT_FALSE(tv2.is(tv1));

      IValue none;
      IValue undefinedTensor((Tensor()));

      EXPECT_TRUE(none.is(undefinedTensor));
      EXPECT_TRUE(undefinedTensor.is(none));

      // Is this a bug? We should probably have a is b => a.hash() == b.hash()
      EXPECT_NE(none.hash(), undefinedTensor.hash());

      auto sampleIValues = makeSampleIValues();
      auto sampleIValues2 = makeSampleIValues();
      auto moreSampleIValues = makeMoreSampleIValues();

      ASSERT_EQ(sampleIValues.size(), moreSampleIValues.size());
      for (int ii = 0; ii < sampleIValues.size(); ++ii) {
        // Constant strings will have the same pointer value.
        if (sampleIValues[ii].isPtrType() && !sampleIValues[ii].isString()) {
          EXPECT_NE(sampleIValues[ii].hash(), sampleIValues2[ii].hash());
        } else {
          EXPECT_EQ(sampleIValues[ii].hash(), sampleIValues2[ii].hash());
        }
        EXPECT_NE(sampleIValues[ii].hash(), moreSampleIValues[ii].hash());
      }

    */
}

#[test] fn i_value_test_get_sub_values() {
    todo!();
    /*
    
      // Scalars have no subvalues.
      IValue integer(42), float_(1.5), complex(complex<double>(2, 3));

      IValue::HashAliasedIValues subvalues;

      integer.getSubValues(subvalues);
      EXPECT_TRUE(subvalues.empty());

      subvalues.clear();

      float_.getSubValues(subvalues);
      EXPECT_TRUE(subvalues.empty());

      subvalues.clear();

      complex.getSubValues(subvalues);
      EXPECT_TRUE(subvalues.empty());

      subvalues.clear();

      Tensor t1(rand({3, 4})), t2(rand({3, 4}));
      IValue tv1(t1), tv2(t2);
      IValue list(vector<Tensor>{t1, t2});
      IValue tuple(Tuple::create({tv1, tv2}));

      Dict<i64, Tensor> m;
      m.insert(1, t1);
      m.insert(2, t2);

      IValue dict(move(m));

      auto objType = ClassType::create(nullopt, {});
      objType->addAttribute("t1", tv1.type());
      objType->addAttribute("t2", tv2.type());

      auto o = Object::create(StrongTypePtr(nullptr, objType), 2);
      o->setSlot(0, tv1);
      o->setSlot(1, tv2);

      IValue object(o);
      tv1.getSubValues(subvalues);
      EXPECT_EQ(subvalues.size(), 1);
      EXPECT_EQ(subvalues.count(tv1), 1);

      subvalues.clear();

      for (auto& container: {list, tuple, dict, object}) {
        container.getSubValues(subvalues);
        EXPECT_EQ(subvalues.size(), 3);
        EXPECT_EQ(subvalues.count(container), 1);
        EXPECT_EQ(subvalues.count(tv1), 1);
        EXPECT_EQ(subvalues.count(tv2), 1);

        subvalues.clear();
      }

    */
}

#[test] fn i_value_test_scalar_bool() {
    todo!();
    /*
    
      Scalar expected(true);
      IValue v(expected);
      Scalar actual = v.toScalar();
      EXPECT_TRUE(actual.isBoolean());
      EXPECT_TRUE(actual.toBool());

    */
}

// TODO(gmagogsfm): Add type conversion test?
