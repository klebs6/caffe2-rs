crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/operator_name.h]

/**
  | TODO: consider storing namespace separately
  | too
  |
  */
pub struct OperatorName {
    name:          String,
    overload_name: String,
}

impl OperatorName {

    pub fn new(
        name:          String,
        overload_name: String) -> Self {
    
        todo!();
        /*
        : name(move(name)),
        : overload_name(move(overload_name)),
        */
    }

    /**
      | TODO: These two functions below are slow!
      | Fix internal data structures so
      | I don't have to manually reconstruct the
      | namespaces!
      |
      | Return the namespace of this OperatorName, if
      | it exists.
      |
      | The returned string_view is only live as long
      | as the OperatorName exists and name is not
      | mutated
      |
      */
    pub fn get_namespace(&self) -> Option<StringView> {
        
        todo!();
        /*
            auto pos = name.find("::");
        if (pos == string::npos) {
          return nullopt;
        } else {
          return make_optional(string_view(name.data(), pos));
        }
        */
    }

    /**
      | Returns true if we successfully set
      | the namespace
      |
      */
    pub fn set_namespace_if_not_set(&mut self, ns: *const u8) -> bool {
        
        todo!();
        /*
            ostringstream oss;
        if (!getNamespace().has_value()) {
          oss << ns << "::" << name;
          name = oss.str();
          return true;
        } else {
          return false;
        }
        */
    }
}

/**
  | Non-owning view of an OperatorName.
  |
  | Unlike OperatorName, most of its functions are
  | constexpr, so it can be used for compile time
  | computations
  |
  */
pub struct OperatorNameView {
    name:          StringView,
    overload_name: StringView,
}

impl OperatorNameView {

    pub fn new(
        name:          StringView,
        overload_name: StringView) -> Self {
    
        todo!();
        /*
        : name(name),
        : overload_name(overload_name),

        
        */
    }

    /// Parses strings like "foo.overload" and
    /// also "foo"
    ///
    pub const fn parse(full_name: StringView) -> OperatorNameView {
        
        todo!();
        /*
            auto i = full_name.find('.');
        if (i == string_view::npos) {
          return OperatorNameView(full_name, string_view());
        } else {
          return OperatorNameView(full_name.substr(0, i), full_name.substr(i + 1));
        }
        */
    }
}


impl PartialEq<OperatorName> for OperatorName {
    
    fn eq(&self, other: &OperatorName) -> bool {
        todo!();
        /*
            return lhs.name == rhs.name && lhs.overload_name == rhs.overload_name;
        */
    }
}

lazy_static!{
    /*
    namespace std {

      template <>
      struct hash<::OperatorName> {
        usize operator()(const ::OperatorName& x) const {
          return hash<string>()(x.name) ^ (~ hash<string>()(x.overload_name));
        }
      };
    }
    */
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/operator_name.cpp]

pub fn to_string(op_name: &OperatorName) -> String {
    
    todo!();
        /*
            ostringstream oss;
      oss << opName;
      return oss.str();
        */
}

impl fmt::Display for OperatorName {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            os << opName.name;
      if (opName.overload_name.size() != 0) {
        os << "." << opName.overload_name;
      }
      return os;
        */
    }
}
