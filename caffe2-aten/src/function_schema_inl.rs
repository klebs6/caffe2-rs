crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/function_schema_inl.h]

/**
  | note: windows build doesn't find symbols
  | in operator files unless this is a header
  | file
  |
  */
impl fmt::Display for FunctionSchema {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            // eventually this should look almost identical to python arg parser, but
      // it is simpler for now to work directly on this schema

      out << schema.name();
      if (schema.overload_name() != "") {
        out << "." << schema.overload_name();
      }
      out << "(";

      bool seen_kwarg_only = false;
      for(usize i = 0; i < schema.arguments().size(); ++i) {
        if (i > 0) out << ", ";
        if (schema.arguments()[i].kwarg_only() && !seen_kwarg_only) {
          out << "*, ";
          seen_kwarg_only = true;
        }
        out << schema.arguments()[i];
      }

      if(schema.is_vararg()) {
        if(schema.arguments().size() > 0)
          out << ", ";
        out << "...";
      }

      out << ") -> ";

      const auto& returns = schema.returns();
      out << "(";
      for(usize i = 0; i < returns.size(); ++i) {
        if (i > 0) {
          out << ", ";
        }
        out << returns.at(i);
      }
      if (schema.is_varret()) {
        if (returns.size() != 0) {
          out << ", ";
        }
        out << "...";
      }
      out << ")";
      return out;
        */
    }
}

impl Argument {
    
    #[inline] pub fn is_backward_compatible_with(&self, 
        old:     &Argument,
        why_not: *mut std::io::BufWriter) -> bool {
        
        todo!();
        /*
            const Argument* lhs = this;
        const Argument* rhs = &old;
        if (!(lhs->name() == rhs->name()
            && lhs->N() == rhs->N()
            && lhs->alias_info() == rhs->alias_info())) {
          return false;
        }
        if (lhs->kwarg_only() && !rhs->kwarg_only()) {
          return false;
        }
        if (!rhs->type()->isSubtypeOfExt(lhs->type(), why_not)) {
          return false;
        }
        if (rhs->default_value().has_value() &&
            lhs->default_value() != rhs->default_value()) {
          return false;
        }
        return true;
        */
    }
}

impl FunctionSchema {
    
    #[inline] pub fn format_type_mismatch_msg(&self, 
        expected:    &Argument,
        actual_type: &String,
        position:    Option<usize>,
        value:       Option<String>) -> String {
        
        todo!();
        /*
            std::string position_str;
      if (position) {
        position_str = c10::str("Position: ", *position, "\n");
      }
      std::string value_str;
      if (value) {
        value_str = c10::str("Value: ", *value, "\n");
      }
      return c10::str(
          name(),
          "() ",
          expected.formatTypeMismatchMsg(actual_type),
          position_str,
          value_str,
          "Declaration: ",
          *this);
        */
    }
    
    #[inline] pub fn is_backward_compatible_with(&self, 
        old:     &FunctionSchema,
        why_not: *mut std::io::BufWriter) -> bool {
        
        todo!();
        /*
            if (!(name() == old.name()
            && overload_name() == old.overload_name()
            // we are conservative on is_vararg and is_varret,
            // since they are only used by internal operators
            && is_vararg() == old.is_vararg()
            && is_varret() == old.is_varret()
            && returns().size() == old.returns().size()
            && arguments().size() >= old.arguments().size())) {
        return false;
      }
      for (usize i = 0; i < returns().size(); ++i) {
        // Backwards compatibility requires covariance on argument types
        // (i.e. more generic), and contravariance on return types (i.e.
        //  more specific).
        if (!old.returns().at(i).isBackwardCompatibleWith(
              returns().at(i),
              why_not)) {
          return false;
        }
      }

      // Make sure that all the old arguments have their corresponding backward
      // compatible arguments in this schema.
      for (usize i = 0; i < old.arguments().size(); ++i) {
        if (!arguments().at(i).isBackwardCompatibleWith(
              old.arguments().at(i), why_not)) {
          return false;
        }
      }

      // Validate that all new arguments provided a default value.
      for (usize i = old.arguments().size(); i < arguments().size(); ++i) {
        if (!arguments().at(i).default_value()) {
          if (why_not) {
            *why_not
                << "Function schema not backward compatible since the new argument '"
                << arguments().at(i).name() << "' of type "
                << arguments().at(i).type()->str()
                << " did not provide a default value.";
          }
          return false;
        }
      }

      return true;
        */
    }
    
    #[inline] pub fn check_arg(&self, 
        value:    &IValue,
        argument: &Argument,
        pos:      Option<usize>)  {
        
        todo!();
        /*
            if (value.isTensor() && argument.type() == TensorType::get()) {
        // Fast-path for the common case
        return;
      }
      if (!value.type()->isSubtypeOf(argument.type())) {
        TORCH_CHECK(
            false,
            formatTypeMismatchMsg(
                argument, value.type()->repr_str(), pos));
      }
        */
    }
    
    #[inline] pub fn find_error_in_kwargs(&self, kwargs: &Vec<String>) -> String {
        
        todo!();
        /*
            // First check if any of the kwargs are unknown, i.e. don't match the name of
      // any argument in the schema.
      for (const auto& kwarg : kwargs) {
        if (!std::count_if(
                arguments().begin(),
                arguments().end(),
                [&kwarg](const Argument& argument) {
                  return argument.name() == kwarg;
                })) {
          return c10::str(
              "Unknown keyword argument '",
              kwarg,
              "' for operator '",
              name(),
              "'. Schema: ",
              *this);
        }
      }
      // If there are unconsumed kwargs but none of them were unknown, the first
      // positional argument present in the kwargs is duplicated.
      for (const auto& argument : arguments()) {
        if (std::find(kwargs.begin(), kwargs.end(), argument.name()) != kwargs.end()) {
          AT_ASSERT(!argument.default_value());
          return c10::str(
              "Argument '",
              argument.name(),
              "' specified both as positional and ",
              "keyword argument. Schema: ",
              *this);
        }
      }
      return "";
        */
    }
    
    #[inline] pub fn check_and_normalize_inputs(&self, 
        inputs: &mut Vec<IValue>,
        kwargs: &HashMap<String,IValue>)  {
        
        todo!();
        /*
            // Do we have more inputs than the schema accepts?
      TORCH_CHECK(
          inputs.size() <= arguments().size(),
          "Expected at most ",
          arguments().size(),
          " argument(s) for operator '",
          name(),
          "', but received ",
          inputs.size(),
          " argument(s). Declaration: ",
          *this);

      usize consumed_kwargs = 0;
      for (usize pos = 0; pos < arguments().size(); ++pos) {
        const auto& argument = arguments()[pos];
        if (pos < inputs.size()) {
          checkArg(inputs[pos], argument, pos);
          continue;
        }
        auto it = kwargs.find(argument.name());
        if (it != kwargs.end()) {
          checkArg(it->second, argument, nullopt);
          inputs.push_back(it->second);
          consumed_kwargs++;
          continue;
        }
        if (argument.default_value()) {
          inputs.push_back(*argument.default_value());
          continue;
        }
        AT_ERROR(
            name(),
            "() is missing value for argument '",
            argument.name(),
            "'. Declaration: ",
            *this);
      }
      if (consumed_kwargs != kwargs.size()) {
        std::vector<std::string> names;
        for(const auto& k : kwargs) {
          names.emplace_back(k.first);
        }
        throw std::runtime_error(findErrorInKwargs(names));
      }
        */
    }
    
    #[inline] pub fn clone_with_remapped_types(&self, type_map: fn(_0: TypePtr) -> TypePtr) -> FunctionSchema {
        
        todo!();
        /*
            auto update_args = [&](const std::vector<Argument>& args) {
        std::vector<Argument> new_args;
        new_args.reserve(args.size());
        for(const Argument& arg : args) {
          new_args.emplace_back(arg.cloneWithType(type_map(arg.type())));
        }
        return new_args;
      };
      return FunctionSchema(
          name(),
          overload_name(),
          update_args(arguments()),
          update_args(returns()),
          is_vararg(),
          is_varret());
        */
    }
}

/**
  | covariant subtyping of list of Arguments
  |
  */
#[inline] pub fn is_subtype_of_list(
        child:   &[Argument],
        parent:  &[Argument],
        why_not: *mut std::io::BufWriter) -> bool {
    
    todo!();
        /*
            if (child.size() != parent.size()) {
        return false;
      }
      for (usize i = 0; i < child.size(); ++i) {
        const Argument& c = child[i];
        const Argument& p = parent[i];
        if (c.name() != p.name()) {
          return false;
        }
        if (!c.type()->isSubtypeOfExt(p.type(), why_not)) {
          return false;
        }
      }
      return true;
        */
}

impl FunctionSchema {
    
    #[inline] pub fn is_subtype_of(&self, 
        rhs:       &FunctionSchema,
        as_method: bool,
        why_not:   *mut std::io::BufWriter) -> bool {
        
        todo!();
        /*
            usize start = as_method ? 1 : 0;
      // functions are contravariant in arguments but covariant in returns
      return isSubtypeOfList(
                 ArrayRef<Argument>(rhs.arguments()).slice(start),
                 ArrayRef<Argument>(arguments()).slice(start),
                 why_not) &&
          isSubtypeOfList(returns(), rhs.returns(), why_not);
        */
    }
}
