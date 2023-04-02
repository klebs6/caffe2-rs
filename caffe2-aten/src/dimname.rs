crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/Dimname.cpp]

lazy_static!{
    /*
    static Symbol kWildcard = Symbol::dimname("*");
    */
}

pub struct Dimname {
    //TODO: this is a scaffold -- what data does this need?

}

impl fmt::Display for Dimname {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            if (dimname.type() == NameType::WILDCARD) {
        out << "None";
      } else {
        out << "'" << dimname.symbol().toUnqualString() << "'";
      }
      return out;
        */
    }
}

pub fn check_valid_identifier(name: &String)  {
    
    todo!();
        /*
            TORCH_CHECK(
          Dimname::isValidName(name),
          "Invalid name: a valid identifier contains only digits, alphabetical "
          "characters, and/or underscore and starts with a non-digit. got: '",
          name, "'.");
        */
}

impl Dimname {
    
    pub fn is_valid_name(&mut self, name: &str) -> bool {
        
        todo!();
        /*
            // allow valid ASCII python identifiers: "uppercase and lowercase
      // letters A through Z, the underscore _ and, except for the first
      // character, the digits 0 through 9" (at least length 1)
      // https://docs.python.org/3/reference/lexical_analysis.html#identifiers
      if (name.length() == 0) {
        return false;
      }
      for (auto it = name.begin(); it != name.end(); ++it) {
        if (std::isalpha(*it) || *it == '_') {
          continue;
        } else if (it != name.begin() && std::isdigit(*it)) {
          continue;
        }
        return false;
      }
      return true;
        */
    }

    pub fn from_symbol(&mut self, name: Symbol) -> Dimname {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(name.is_dimname());
      if (name == kWildcard) {
        return Dimname::wildcard();
      }
      check_valid_identifier(name.toUnqualString());
      return Dimname(name);
        */
    }
    
    pub fn wildcard(&mut self) -> Dimname {
        
        todo!();
        /*
            static Dimname result(kWildcard, NameType::WILDCARD);
      return result;
        */
    }
    
    pub fn unify(&self, other: Dimname) -> Option<Dimname> {
        
        todo!();
        /*
            if (other.type() == NameType::WILDCARD) {
        return *this;
      }
      if (type_ == NameType::WILDCARD) {
        return other;
      }
      if (name_ == other.symbol()) {
        return *this;
      }
      return c10::nullopt;
        */
    }
    
    pub fn matches(&self, other: Dimname) -> bool {
        
        todo!();
        /*
            return unify(other).has_value();
        */
    }
}
