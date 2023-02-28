crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/function_schema.h]

/**
  | schema as used in the compiler for resolving
  | function calls and reporting errors.
  |
  | These
  | objects should be constructed from C10 schema
  | once those are available.
  */
pub struct Argument {

    name:          String,
    ty:            TypePtr,

    /**
      | for list types, an optional statically
      | known length for the list e.g. for int[3]:
      | type = ListType::ofInts(), N = 3
      | 
      | If present, this will allow scalars
      | to be broadcast to this length to become
      | a list.
      |
      */
    N:             Option<i32>,

    default_value: Option<IValue>,

    /**
      | is this only specifiable as a keyword
      | argument?
      |
      */
    kwarg_only:    bool,

    alias_info:    Option<AliasInfo>,
}

impl Argument {
    
    pub fn new(
        name:          String,
        ty:            TypePtr,
        N:             Option<i32>,
        default_value: Option<IValue>,
        kwarg_only:    bool,
        alias_info:    Option<AliasInfo>) -> Self {

        let name: String = name.unwrap_or("");

        let ty: TypePtr = ty.unwrap_or(nullptr);

        let N: Option<i32> = N.unwrap_or(nullopt);

        let default_value: Option<IValue> =
                 default_value.unwrap_or(nullopt);

        let kwarg_only: bool = kwarg_only.unwrap_or(false);

        let alias_info: Option<AliasInfo> =
                 alias_info.unwrap_or(nullopt);
        todo!();
        /*


            : name_(move(name)),
            type_(type ? type : TensorType::get()),
            N_(move(N)),
            default_value_(move(default_value)),
            kwarg_only_(kwarg_only),
            alias_info_(move(alias_info))
        */
    }
    
    pub fn name(&self) -> &String {
        
        todo!();
        /*
            return name_;
        */
    }
    
    pub fn ty(&self) -> &TypePtr {
        
        todo!();
        /*
            return type_;
        */
    }
    
    pub fn N(&self) -> Option<i32> {
        
        todo!();
        /*
            return N_;
        */
    }
    
    pub fn default_value(&self) -> &Option<IValue> {
        
        todo!();
        /*
            return default_value_;
        */
    }
    
    pub fn kwarg_only(&self) -> bool {
        
        todo!();
        /*
            return kwarg_only_;
        */
    }
    
    pub fn alias_info(&self) -> &Option<AliasInfo> {
        
        todo!();
        /*
            return alias_info_;
        */
    }
    
    pub fn is_inferred_type(&self) -> bool {
        
        todo!();
        /*
            bool is_inferred_type = false;
        TORCH_INTERNAL_ASSERT(type_);
        if (auto pt = type_->cast<TensorType>()) {
          if (pt->isInferredType()) {
            is_inferred_type = true;
          }
        }
        return is_inferred_type;
        */
    }
    
    pub fn format_type_mismatch_msg(&self, actual_type: &String) -> String {
        
        todo!();
        /*
            string inferred_type_hint;
        if (is_inferred_type()) {
          inferred_type_hint = str(
              "Inferred '",
              name(),
              "' to be of type 'Tensor' ",
              "because it was not annotated with an explicit type.\n");
        }
        return str(
            "Expected a value of type '",
            type()->repr_str(),
            "' for argument '",
            name(),
            "' but instead found type '",
            actual_type,
            "'.\n",
            inferred_type_hint);
        */
    }
    
    pub fn clone_with_type(&self, new_type: TypePtr) -> Argument {
        
        todo!();
        /*
            return Argument(
            name_,
            move(new_type),
            N_,
            default_value_,
            kwarg_only_,
            alias_info_);
        */
    }

    /**
      | this function checks whether this Argument is
      | backward compatible with the old one. we
      | consider the following cases are backward
      | compatible:
      |
      |   1) two arguments are equal
      |
      |   2) this arg's type should be subtype of old
      |
      |   3) this arg must provide the same default
      |   value if old arg has one,
      |
      */
    pub fn is_backward_compatible_with(&self, 
        old:     &Argument,
        why_not: *mut std::io::BufWriter) -> bool {
        let why_not: *mut std::io::BufWriter = why_not.unwrap_or(nullptr);

        todo!();
        /*
        
        */
    }
}

impl PartialEq<Argument> for Argument {
    
    fn eq(&self, other: &Argument) -> bool {
        todo!();
        /*
            return lhs.name() == rhs.name()
              && *lhs.type() == *rhs.type()
              && lhs.N() == rhs.N()
              && lhs.default_value() == rhs.default_value()
              && lhs.kwarg_only() == rhs.kwarg_only()
              && lhs.alias_info() == rhs.alias_info();
        */
    }
}

//-----------------------------
pub struct FunctionSchema {
    name:       OperatorName,
    arguments:  Vec<Argument>,
    returns:    Vec<Argument>,

    /**
      | if true then this schema takes an arbitrary
      | number of additional arguments after
      | the argument specified in arguments
      | currently this is used primarily to
      | represent 'primitive' operators whose
      | arguments are not checked by schema
      |
      */
    is_vararg:  bool,

    is_varret:  bool,

    /**
      | if no alias information is directly
      | specified, what kind of "default" alias
      | information should we infer?
      | 
      | NB: due to alias analysis kind merging,
      | this may be nullopt. Eventually this
      | should always be set no matter what
      |
      */
    alias_kind: Option<AliasAnalysisKind>,
}

impl FunctionSchema {

    pub fn new(
        name:          String,
        overload_name: String,
        arguments:     Vec<Argument>,
        returns:       Vec<Argument>,
        is_vararg:     bool,
        is_varret:     bool) -> Self {

        let is_vararg: bool = is_vararg.unwrap_or(false);
        let is_varret: bool = is_varret.unwrap_or(false);

        todo!();
        /*


            : name_({move(name), move(overload_name)}),
            arguments_(move(arguments)),
            returns_(move(returns)),
            is_vararg_(is_vararg),
            is_varret_(is_varret) 
        checkSchema();
        */
    }
    
    pub fn new(
        name:          Symbol,
        overload_name: String,
        arguments:     Vec<Argument>,
        returns:       Vec<Argument>,
        is_vararg:     bool,
        is_varret:     bool) -> Self {

        let is_vararg: bool = is_vararg.unwrap_or(false);
        let is_varret: bool = is_varret.unwrap_or(false);

        todo!();
        /*


            : FunctionSchema(
                name.toQualString(),
                move(overload_name),
                move(arguments),
                move(returns),
                is_vararg,
                is_varret) 
        checkSchema();
        */
    }

    /**
      | Checks whether this schema is backward
      | compatible with the old one. The following
      | conditions must be true:
      |
      | [Function structure] The new schema's name,
      |      overload-name, varargs, and return arity
      |      are the same.
      |
      | [Output Narrowing] The new schema's output
      |      type must be the same class or inherit
      |      from the old schema's output type.
      |
      | [Argument count] The new schema must have at
      |      least as many arguments as the old
      |      schema (considering the list of
      |      positional and kwargs).
      |
      | [Arg Compatibility] Every argument in the old
      |      schema has a corresponding argument in
      |      the new schema that:
      |
      |        * is at the same position.
      |
      |        * has the same name.
      |
      |        * is either positional, or kwarg and
      |        the old argument was kwarg.
      |
      |        * has the same type, or the old
      |          argument's type inherits from the
      |          new argument's type.
      |
      | [Default Values] Every new argument must have
      | a default value.
      |
      | E.g.
      |   OK    f_new(a, b, c=1) => f_old(a, b)
      |   NOK   f_new(a, c=1, *, b) => f_old(a, *, b)
      |   OK    f_new(a, b, *, c) => f_old(a, *, b, c)
      |   NOK   f_new(a, *, b, c) -> f_old(a, b, *, c)
      |   NOK   f_new(a, *, c, b) => f_old(a, *, b, c)
      |   OK    f_new(a, *, b, c, d=1) => f_old(a, *, b, c)
      */
    pub fn is_backward_compatible_with(&self, 
        old:     &FunctionSchema,
        why_not: *mut std::io::BufWriter) -> bool {

        let why_not: *mut std::io::BufWriter = why_not.unwrap_or(nullptr);

        todo!();
        /*
        
        */
    }
    
    pub fn check_arg(&self, 
        value:    &IValue,
        argument: &Argument,
        pos:      Option<usize>)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn check_schema(&self)  {
        
        todo!();
        /*
            bool seen_default_arg = false;
        for (const auto& arg : arguments()) {
          if (arg.default_value()) {
            seen_default_arg = true;
          } else {
            // we have historically serialized broadcasting lists wo/default values,
            // so to not break BC allow lists here
            if (arg.type()->kind() == ListType::Kind) {
              continue;
            }
            TORCH_INTERNAL_ASSERT(
                !seen_default_arg || arg.kwarg_only(),
                "Non-default positional argument follows default argument. Parameter ",
                arg.name(),
                " in ",
                *this);
          }
        }
        */
    }
    
    pub fn dump(&self)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn operator_name(&self) -> &OperatorName {
        
        todo!();
        /*
            return name_;
        */
    }
    
    pub fn name(&self) -> &String {
        
        todo!();
        /*
            return name_.name;
        */
    }
    
    pub fn overload_name(&self) -> &String {
        
        todo!();
        /*
            return name_.overload_name;
        */
    }
    
    pub fn arguments(&self) -> &Vec<Argument> {
        
        todo!();
        /*
            return arguments_;
        */
    }
    
    pub fn returns(&self) -> &Vec<Argument> {
        
        todo!();
        /*
            return returns_;
        */
    }
    
    pub fn is_vararg(&self) -> bool {
        
        todo!();
        /*
            return is_vararg_;
        */
    }
    
    pub fn is_varret(&self) -> bool {
        
        todo!();
        /*
            return is_varret_;
        */
    }
    
    pub fn is_mutable(&self) -> bool {
        
        todo!();
        /*
            return any_of(
            arguments_.cbegin(), arguments_.cend(), [](const Argument& arg) {
              const auto& aliasInfo = arg.alias_info();
              return aliasInfo && aliasInfo.value().isWrite();
            });
        */
    }
    
    pub fn argument_index_with_name(&self, name: &String) -> Option<i32> {
        
        todo!();
        /*
            for(usize i = 0; i < arguments().size(); ++i) {
          if(name == arguments()[i].name())
            return i;
        }
        return nullopt;
        */
    }
    
    pub fn clone_with_name(&self, 
        name:          String,
        overload_name: String) -> FunctionSchema {
        
        todo!();
        /*
            return FunctionSchema(
            move(name),
            move(overload_name),
            arguments(),
            returns(),
            is_vararg(),
            is_varret()
            );
        */
    }
    
    pub fn clone_with_arguments(&self, new_arguments: Vec<Argument>) -> FunctionSchema {
        
        todo!();
        /*
            return FunctionSchema(
            name(),
            overload_name(),
            move(new_arguments),
            returns(),
            is_vararg(),
            is_varret());
        */
    }
    
    pub fn clone_with_returns(&self, new_returns: Vec<Argument>) -> FunctionSchema {
        
        todo!();
        /*
            return FunctionSchema(
            name(),
            overload_name(),
            arguments(),
            move(new_returns),
            is_vararg(),
            is_varret());
        */
    }
    
    pub fn format_type_mismatch_msg(&self, 
        expected:    &Argument,
        actual_type: &String,
        position:    Option<usize>,
        value:       Option<String>) -> String {

        let position: Option<usize> = position.unwrap_or(nullopt);
        let value: Option<String> = value.unwrap_or(nullopt);

        todo!();
        /*
        
        */
    }
    
    pub fn clone_with_remapped_types(&self, type_map: fn(_0: TypePtr) -> TypePtr) -> FunctionSchema {
        
        todo!();
        /*
        
        */
    }

    /**
      | Check that inputs have the correct types
      | and appends any missing default values.
      |
      */
    pub fn check_and_normalize_inputs(&self, 
        inputs: &mut Vec<IValue>,
        kwargs: &HashMap<String,IValue>)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn find_error_in_kwargs(&self, kwargs: &Vec<String>) -> String {
        
        todo!();
        /*
        
        */
    }
    
    pub fn has_any_alias_info(&self) -> bool {
        
        todo!();
        /*
            for (const auto& arg : arguments_) {
          if (arg.alias_info().has_value()) {
            return true;
          }
        }
        for (const auto& ret : returns_) {
          if (ret.alias_info().has_value()) {
            return true;
          }
        }
        return false;
        */
    }
    
    /**
      | TODO remove the mutation here
      |
      */
    pub fn is_default_alias_analysis_kind(&self) -> bool {
        
        todo!();
        /*
            return !alias_kind_;
        */
    }
    
    pub fn alias_analysis(&self) -> AliasAnalysisKind {
        
        todo!();
        /*
            return alias_kind_.value_or(AliasAnalysisKind::CONSERVATIVE);
        */
    }
    
    pub fn set_alias_analysis(&mut self, v: AliasAnalysisKind)  {
        
        todo!();
        /*
            alias_kind_ = v;
        */
    }
    
    pub fn get_namespace(&self) -> Option<StringView> {
        
        todo!();
        /*
            return name_.getNamespace();
        */
    }
    
    /**
      | Returns true if we successfully set
      | the namespace (as there was none set,
      | and false otherwise)
      |
      */
    pub fn set_namespace_if_not_set(&mut self, ns: *const u8) -> bool {
        
        todo!();
        /*
            return name_.setNamespaceIfNotSet(ns);
        */
    }

    /**
      | can a function with this schema be
      | substituted for a function of rhs's schema
      | and have the program typecheck?
      |
      | as_method - if true, treat this schema as
      | a method and ignore the first argument, which
      | will be the object in both cases
      */
    pub fn is_subtype_of(&self, 
        rhs:       &FunctionSchema,
        as_method: bool,
        why_not:   *mut std::io::BufWriter) -> bool {
        let why_not: *mut std::io::BufWriter = why_not.unwrap_or(nullptr);

        todo!();
        /*
        
        */
    }
}

impl PartialEq<FunctionSchema> for FunctionSchema {
    
    fn eq(&self, other: &FunctionSchema) -> bool {
        todo!();
        /*
            return lhs.name() == rhs.name()
         && lhs.overload_name() == rhs.overload_name()
         && lhs.arguments() == rhs.arguments()
         && lhs.returns() == rhs.returns()
         && lhs.is_vararg() == rhs.is_vararg()
         && lhs.is_varret() == rhs.is_varret();
        */
    }
}

/**
  | print out Argument, which is compatible with
  | FunctionSchema parser full format: Type(alias)?
  | name=default_value
  |
  */
impl fmt::Display for &mut Argument {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            // for adjusting the ? position.
      // in schema, we have Tensor?(a!) input, and t(a!)?.
      // however, t?(a!) doesn't work with schema parser.
      // so we always use Type(alias)? format
      auto type = arg.type();
      bool is_opt = type->kind() == OptionalType::Kind;
      auto unopt_type = is_opt ? type->castRaw<OptionalType>()->getElementType() : type;

      if (unopt_type->kind() == ListType::Kind && arg.N()) {
        // sized lists get size N from arg, not type
        auto list = unopt_type->cast<ListType>();
        out << list->getElementType()->str() << "[" << *arg.N() << "]";
      } else {
        out << unopt_type->str();
      }

      if (arg.alias_info()) {
        out << arg.alias_info().value();
      }

      if (is_opt) {
        out << "?";
      }

      if (!arg.name().empty()) {
        out << " " << arg.name();
      }

      if (arg.default_value()) {
        out << "=";
        if (type->kind() == TypeKind::StringType || (unopt_type->kind() == TypeKind::StringType && !arg.default_value().value().isNone())) {
          printQuotedString(out, arg.default_value().value().toStringRef());
        } else {
          out << arg.default_value().value();
        }
      }

      return out;
        */
    }
}

#[inline] pub fn to_string(schema: &FunctionSchema) -> String {
    
    todo!();
        /*
            ostringstream str;
      str << schema;
      return str.str();
        */
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/function_schema.cpp]

impl FunctionSchema {
    
    pub fn dump(&self)  {
        
        todo!();
        /*
            cout << *this << "\n";
        */
    }
}
