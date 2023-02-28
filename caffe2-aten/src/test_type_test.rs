crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/test/type_test.cpp]

#[test] fn type_custom_printer_basic() {
    todo!();
    /*
    
      TypePrinter printer =
          [](const ConstTypePtr& t) -> optional<string> {
        if (auto tensorType = t->cast<TensorType>()) {
          return "CustomTensor";
        }
        return nullopt;
      };

      // Tensor types should be rewritten
      TorchTensor iv = Torchrand({2, 3});
      const auto type = TensorType::create(iv);
      EXPECT_EQ(type->annotation_str(), "Tensor");
      EXPECT_EQ(type->annotation_str(printer), "CustomTensor");

      // Unrelated types shoudl not be affected
      const auto intType = IntType::get();
      EXPECT_EQ(intType->annotation_str(printer), intType->annotation_str());

    */
}

#[test] fn type_custom_printer_contained_types() {
    todo!();
    /*
    
      TypePrinter printer =
          [](const ConstTypePtr& t) -> optional<string> {
        if (auto tensorType = t->cast<TensorType>()) {
          return "CustomTensor";
        }
        return nullopt;
      };
      TorchTensor iv = Torchrand({2, 3});
      const auto type = TensorType::create(iv);

      // Contained types should work
      const auto tupleType = TupleType::create({type, IntType::get(), type});
      EXPECT_EQ(tupleType->annotation_str(), "Tuple[Tensor, int, Tensor]");
      EXPECT_EQ(
          tupleType->annotation_str(printer), "Tuple[CustomTensor, int, CustomTensor]");
      const auto dictType = DictType::create(IntType::get(), type);
      EXPECT_EQ(dictType->annotation_str(printer), "Dict[int, CustomTensor]");
      const auto listType = ListType::create(tupleType);
      EXPECT_EQ(
          listType->annotation_str(printer),
          "List[Tuple[CustomTensor, int, CustomTensor]]");

    */
}

#[test] fn type_custom_printer_named_tuples() {
    todo!();
    /*
    
      TypePrinter printer =
          [](const ConstTypePtr& t) -> optional<string> {
        if (auto tupleType = t->cast<TupleType>()) {
          // Rewrite only namedtuples
          if (tupleType->name()) {
            return "Rewritten";
          }
        }
        return nullopt;
      };
      TorchTensor iv = Torchrand({2, 3});
      const auto type = TensorType::create(iv);

      const auto namedTupleType = TupleType::createNamed(
          "my.named.tuple", {"foo", "bar"}, {type, IntType::get()});
      EXPECT_EQ(namedTupleType->annotation_str(printer), "Rewritten");

      // Put it inside another tuple, should still work
      const auto outerTupleType = TupleType::create({IntType::get(), namedTupleType});
      EXPECT_EQ(outerTupleType->annotation_str(printer), "Tuple[int, Rewritten]");

    */
}

pub fn import_type(
    cu:        Arc<CompilationUnit>,
    qual_name: &String,
    src:       &String) -> TypePtr {
    
    todo!();
        /*
            vector<IValue> constantTable;
      auto source = make_shared<TorchJitSource>(src);
      TorchJitSourceImporter si(
          cu,
          &constantTable,
          [&](const string& name) -> shared_ptr<TorchJitSource> {
            return source;
          },
          /*version=*/2);
      return si.loadType(qual_name);
        */
}

#[test] fn type_equality_class_basic() {
    todo!();
    /*
    
      // Even if classes have the same name across two compilation units, they
      // should not compare equal.
      auto cu = make_shared<CompilationUnit>();
      const auto src = R"JIT(
    class First:
        def one(self, x: Tensor, y: Tensor) -> Tensor:
          return x
    )JIT";

      auto classType = importType(cu, "__torch__.First", src);
      auto classType2 = cu->get_type("__torch__.First");
      // Trivially these should be equal
      EXPECT_EQ(*classType, *classType2);

    */
}

#[test] fn type_equality_class_inequality() {
    todo!();
    /*
    
      // Even if classes have the same name across two compilation units, they
      // should not compare equal.
      auto cu = make_shared<CompilationUnit>();
      const auto src = R"JIT(
    class First:
        def one(self, x: Tensor, y: Tensor) -> Tensor:
          return x
    )JIT";

      auto classType = importType(cu, "__torch__.First", src);

      auto cu2 = make_shared<CompilationUnit>();
      const auto src2 = R"JIT(
    class First:
        def one(self, x: Tensor, y: Tensor) -> Tensor:
          return y
    )JIT";

      auto classType2 = importType(cu2, "__torch__.First", src2);
      EXPECT_NE(*classType, *classType2);

    */
}

#[test] fn type_equality_interface() {
    todo!();
    /*
    
      // Interfaces defined anywhere should compare equal, provided they share a
      // name and interface
      auto cu = make_shared<CompilationUnit>();
      const auto interfaceSrc = R"JIT(
    class OneForward(Interface):
        def one(self, x: Tensor, y: Tensor) -> Tensor:
            pass
        def forward(self, x: Tensor) -> Tensor:
            pass
    )JIT";
      auto interfaceType = importType(cu, "__torch__.OneForward", interfaceSrc);

      auto cu2 = make_shared<CompilationUnit>();
      auto interfaceType2 = importType(cu2, "__torch__.OneForward", interfaceSrc);

      EXPECT_EQ(*interfaceType, *interfaceType2);

    */
}

#[test] fn type_equality_interface_inequality() {
    todo!();
    /*
    
      // Interfaces must match for them to compare equal, even if they share a name
      auto cu = make_shared<CompilationUnit>();
      const auto interfaceSrc = R"JIT(
    class OneForward(Interface):
        def one(self, x: Tensor, y: Tensor) -> Tensor:
            pass
        def forward(self, x: Tensor) -> Tensor:
            pass
    )JIT";
      auto interfaceType = importType(cu, "__torch__.OneForward", interfaceSrc);

      auto cu2 = make_shared<CompilationUnit>();
      const auto interfaceSrc2 = R"JIT(
    class OneForward(Interface):
        def two(self, x: Tensor, y: Tensor) -> Tensor:
            pass
        def forward(self, x: Tensor) -> Tensor:
            pass
    )JIT";
      auto interfaceType2 = importType(cu2, "__torch__.OneForward", interfaceSrc2);

      EXPECT_NE(*interfaceType, *interfaceType2);

    */
}

#[test] fn type_equality_tuple() {
    todo!();
    /*
    
      // Tuples should be structurally typed
      auto type = TupleType::create({IntType::get(), TensorType::get(), FloatType::get(), ComplexType::get()});
      auto type2 = TupleType::create({IntType::get(), TensorType::get(), FloatType::get(), ComplexType::get()});

      EXPECT_EQ(*type, *type2);

    */
}

#[test] fn type_equality_named_tuple() {
    todo!();
    /*
    
      // Named tuples should compare equal if they share a name and field names
      auto type = TupleType::createNamed(
          "MyNamedTuple",
          {"a", "b", "c", "d"},
          {IntType::get(), TensorType::get(), FloatType::get(), ComplexType::get()});
      auto type2 = TupleType::createNamed(
          "MyNamedTuple",
          {"a", "b", "c", "d"},
          {IntType::get(), TensorType::get(), FloatType::get(), ComplexType::get()});
      EXPECT_EQ(*type, *type2);

      auto differentName = TupleType::createNamed(
          "WowSoDifferent",
          {"a", "b", "c", "d"},
          {IntType::get(), TensorType::get(), FloatType::get(), ComplexType::get()});
      EXPECT_NE(*type, *differentName);

      auto differentField = TupleType::createNamed(
          "MyNamedTuple",
          {"wow", "so", "very", "different"},
          {IntType::get(), TensorType::get(), FloatType::get(), ComplexType::get()});
      EXPECT_NE(*type, *differentField);

    */
}
