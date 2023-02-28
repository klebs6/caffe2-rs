crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/qualified_name.h]

/**
  | Represents a name of the form "foo.bar.baz"
  |
  */
pub struct QualifiedName {

    /**
      | The actual list of names, like "{foo,
      | bar, baz}"
      |
      */
    atoms:          Vec<String>,

    /**
      | Cached accessors, derived from `atoms_`.
      |
      */
    qualified_name: String,

    prefix:         String,
    name:           String,
}

impl PartialEq<QualifiedName> for QualifiedName {
    
    #[inline] fn eq(&self, other: &QualifiedName) -> bool {
        todo!();
        /*
            return this->qualifiedName_ == other.qualifiedName_;
        */
    }
}

impl QualifiedName {

    pub const DELIMITER: u8 = '.';

    /**
      | `name` can be a dotted string, like "foo.bar.baz",
      | or just a bare name.
      |
      */
    pub fn new(name: &String) -> Self {
    
        todo!();
        /*


            TORCH_CHECK(!name.empty());
        // split the string into its atoms.
        usize startSearchFrom = 0;
        usize pos = name.find(delimiter_, startSearchFrom);

        while (pos != string::npos) {
          auto atom = name.substr(startSearchFrom, pos - startSearchFrom);
          TORCH_INTERNAL_ASSERT(
              atom.size() > 0, "Invalid name for qualified name: '", name, "'");
          atoms_.push_back(move(atom));
          startSearchFrom = pos + 1;
          pos = name.find(delimiter_, startSearchFrom);
        }

        auto finalAtom = name.substr(startSearchFrom, pos - startSearchFrom);
        TORCH_INTERNAL_ASSERT(
            finalAtom.size() > 0, "Invalid name for qualified name: '", name, "'");
        atoms_.emplace_back(move(finalAtom));

        cacheAccessors();
        */
    }
    
    pub fn new(atoms: Vec<String>) -> Self {
    
        todo!();
        /*


            for (const auto& atom : atoms) {
          TORCH_CHECK(!atom.empty(), "Atom cannot be empty");
          TORCH_CHECK(
              atom.find(delimiter_) == string::npos,
              "Delimiter not allowed in atom");
        }
        atoms_ = move(atoms);
        cacheAccessors();
        */
    }

    /**
      | Unnecessary copy. Ideally we'd use
      | something like string_view.
      |
      */
    pub fn new(name: *const u8) -> Self {
    
        todo!();
        /*


            : QualifiedName(string(name))
        */
    }

    /**
      | `name` must be a bare name (no dots!)
      |
      */
    pub fn new(
        prefix: &QualifiedName,
        name:   String) -> Self {
    
        todo!();
        /*


            TORCH_INTERNAL_ASSERT(!name.empty());
        TORCH_INTERNAL_ASSERT(name.find(delimiter_) == string::npos);
        atoms_.insert(atoms_.begin(), prefix.atoms_.begin(), prefix.atoms_.end());
        atoms_.push_back(move(name));

        cacheAccessors();
        */
    }

    /**
      | Is `this` a prefix of `other`?
      |
      | For example, "foo.bar" is a prefix of
      | "foo.bar.baz"
      */
    pub fn is_prefix_of(&self, other: &QualifiedName) -> bool {
        
        todo!();
        /*
            const auto& thisAtoms = atoms_;
        const auto& otherAtoms = other.atoms_;

        if (thisAtoms.size() > otherAtoms.size()) {
          // Can't be a prefix if it's bigger
          return false;
        }
        for (usize i = 0; i < thisAtoms.size(); i++) {
          if (thisAtoms[i] != otherAtoms[i]) {
            return false;
          }
        }
        return true;
        */
    }

    /**
      | The fully qualified name, like "foo.bar.baz"
      |
      */
    pub fn qualified_name(&self) -> &String {
        
        todo!();
        /*
            return qualifiedName_;
        */
    }

    /// The leading qualifier, like "foo.bar"
    ///
    pub fn prefix(&self) -> &String {
        
        todo!();
        /*
            return prefix_;
        */
    }

    /// The base name, like "baz"
    ///
    pub fn name(&self) -> &String {
        
        todo!();
        /*
            return name_;
        */
    }
    
    pub fn atoms(&self) -> &Vec<String> {
        
        todo!();
        /*
            return atoms_;
        */
    }

    /// Helper for cacheAccessors() below.
    pub fn join<T>(&mut self, 
        delimiter: u8,
        v:         &T) -> String {
    
        todo!();
        /*
            string out;
        usize reserve = 0;
        for (const auto& e : v) {
          reserve += e.size() + 1;
        }
        out.reserve(reserve);
        for (usize i = 0; i < v.size(); ++i) {
          if (i != 0) {
            out.push_back(delimiter);
          }
          out.append(v[i]);
        }
        return out;
        */
    }
    
    pub fn cache_accessors(&mut self)  {
        
        todo!();
        /*
            qualifiedName_ = join(delimiter_, atoms_);
        if (atoms_.size() > 1) {
          ArrayRef<string> view(atoms_);
          const auto prefixView = view.slice(0, view.size() - 1);
          prefix_ = join(delimiter_, prefixView);
        }

        if (atoms_.size() >= 1) {
          name_ = atoms_.back();
        }
        */
    }
}

lazy_static!{
    /*
    namespace std {
        struct hash<QualifiedName> {
          usize operator()(const QualifiedName& n) const noexcept {
            return hash<string>()(n.qualifiedName());
          }
        };
    } // namespace std
    */
}
