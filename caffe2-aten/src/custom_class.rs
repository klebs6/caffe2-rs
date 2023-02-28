crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/custom_class.cpp]

pub fn custom_classes() -> &mut HashMap<String,ClassTypePtr> {
    
    todo!();
        /*
            static unordered_map<string, ClassTypePtr> customClasses;
      return customClasses;
        */
}


pub fn register_custom_class(class_type: ClassTypePtr)  {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT(class_type->name());
      auto name = class_type->name()->qualifiedName();
      TORCH_CHECK(
          !customClasses().count(name),
          "Custom class with name ",
          name,
          " is already registered. Ensure that registration with torch::class_ is only called once.");
      customClasses()[name] = move(class_type);
        */
}


pub fn get_custom_class(name: &String) -> ClassTypePtr {
    
    todo!();
        /*
            return customClasses().count(name) ? customClasses()[name] : nullptr;
        */
}


pub fn is_custom_class(v: &IValue) -> bool {
    
    todo!();
        /*
            return v.isObject() && v.toObject()->type()->name() &&
          getCustomClass(v.toObject()->type()->name()->qualifiedName());
        */
}

pub fn custom_class_methods() -> &mut Vec<Box<JitFunction>> {
    
    todo!();
        /*
            static vector<unique_ptr<JitFunction>> customClassMethods;
      return customClassMethods;
        */
}

pub fn register_custom_class_method(fn_: Box<JitFunction>)  {
    
    todo!();
        /*
            customClassMethods().emplace_back(move(fn));
        */
}

pub fn custom_class_schemas_for_bcc_heck() -> Vec<FunctionSchema> {
    
    todo!();
        /*
            auto& methods = customClassMethods();
        return fmap(methods, [](const unique_ptr<JitFunction>& fn) {
          return fn->getSchema();
        });
        */
}

impl ClassBase {
    
    pub fn new(
        namespace_name:              &String,
        class_name:                  &String,
        doc_string:                  String,
        intrusive_ptr_class_typeid:  &type_info::TypeId,
        tagged_capsule_class_typeid: &type_info::TypeId) -> Self {
    
        todo!();
        /*


            : qualClassName("__torch__.torch.classes." + namespaceName + '.' + className),
          classTypePtr(ClassType::create(
                           QualifiedName(qualClassName),
                           weak_ptr<JitCompilationUnit>(),
                           /*is_module=*/false,
                           move(doc_string)))

        detail::checkValidIdent(namespaceName, "Namespace name");
        detail::checkValidIdent(className, "Class name");
        classTypePtr->addAttribute("capsule", CapsuleType::get());
        getCustomClassTypeMap().insert(
            {type_index(intrusivePtrClassTypeid), classTypePtr});
        getCustomClassTypeMap().insert(
            {type_index(taggedCapsuleClassTypeid), classTypePtr});

        registerCustomClass(classTypePtr);
        */
    }
    
    pub fn with_new_arguments(&mut self, 
        schema:       &FunctionSchema,
        default_args: InitializerList<Arg>) -> FunctionSchema {
        
        todo!();
        /*
            const auto& old_args = schema.arguments();
      vector<Argument> new_args;
      new_args.reserve(old_args.size());

      new_args.emplace_back(old_args[0]);
      // Skip self.
      usize argIdx = 1;
      for (const auto& default_arg : default_args) {
        auto& old_arg = old_args[argIdx++];
        new_args.emplace_back(
            default_arg.name_,
            old_arg.type(),
            old_arg.N(),
            default_arg.value_);
      }
      return schema.cloneWithArguments(move(new_args));
        */
    }
}
