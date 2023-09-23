crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/interned_strings.cpp]

pub fn domain_prefix<'a>() -> &'a String {
    
    todo!();
        /*
            static const string _domain_prefix = "org.pytorch.";
      return _domain_prefix;
        */
}

impl InternedStrings {
    
    pub fn symbol(&mut self, s: &String) -> Symbol {
        
        todo!();
        /*
            lock_guard<mutex> guard(mutex_);
      return _symbol(s);
        */
    }
    
    pub fn string(&mut self, sym: Symbol) -> Pair<*const u8,*const u8> {
        
        todo!();
        /*
            // Builtin Symbols are also in the maps, but
      // we can bypass the need to acquire a lock
      // to read the map for Builtins because we already
      // know their string value
    #if defined C10_MOBILE
      return customString(sym);
    #else
      switch (sym) {
    #define DEFINE_CASE(ns, s) \
      case static_cast<unique_t>(ns::s): \
        return {#ns "::" #s, #s};
        FORALL_NS_SYMBOLS(DEFINE_CASE)
    #undef DEFINE_CASE
        default:
          return customString(sym);
      }
    #endif
        */
    }
    
    pub fn ns(&mut self, sym: Symbol) -> Symbol {
        
        todo!();
        /*
            #if defined C10_MOBILE
      lock_guard<mutex> guard(mutex_);
      return sym_to_info_.at(sym).ns;
    #else
      switch (sym) {
    #define DEFINE_CASE(ns, s) \
      case static_cast<unique_t>(ns::s): \
        return namespaces::ns;
        FORALL_NS_SYMBOLS(DEFINE_CASE)
    #undef DEFINE_CASE
        default: {
          lock_guard<mutex> guard(mutex_);
          return sym_to_info_.at(sym).ns;
        }
      }
    #endif
        */
    }
    
    pub fn symbol(&mut self, s: &String) -> Symbol {
        
        todo!();
        /*
            auto it = string_to_sym_.find(s);
      if (it != string_to_sym_.end())
        return it->second;

      auto pos = s.find("::");
      if (pos == string::npos) {
        stringstream ss;
        ss << "all symbols must have a namespace, <namespace>::<string>, but found: " << s;
        throw runtime_error(ss.str());
      }
      Symbol ns = _symbol("namespaces::" + s.substr(0, pos));

      Symbol sym(sym_to_info_.size());
      string_to_sym_[s] = sym;
      sym_to_info_.push_back({ns, s, s.substr(pos + strlen("::"))});
      return sym;
        */
    }
    
    pub fn custom_string(&mut self, sym: Symbol) -> Pair<*const u8,*const u8> {
        
        todo!();
        /*
            lock_guard<mutex> guard(mutex_);
      SymbolInfo& s = sym_to_info_.at(sym);
      return {s.qual_name.c_str(), s.unqual_name.c_str()};
        */
    }
}

pub fn global_strings<'a>() -> &'a mut InternedStrings {
    
    todo!();
        /*
            static InternedStrings s;
      return s;
        */
}

impl Symbol {
    
    pub fn from_qual_string(&mut self, s: &String) -> Symbol {
        
        todo!();
        /*
            return globalStrings().symbol(s);
        */
    }
    
    pub fn to_unqual_string(&self) -> *const u8 {
        
        todo!();
        /*
            return globalStrings().string(*this).second;
        */
    }
    
    pub fn to_qual_string(&self) -> *const u8 {
        
        todo!();
        /*
            return globalStrings().string(*this).first;
        */
    }
    
    pub fn to_display_string(&self) -> *const u8 {
        
        todo!();
        /*
            // TODO: Make this actually return something that's "user friendly".
      // The trouble is that, for this to be usable in printf-style assert
      // statements, this has to return a const char* (whose lifetime is
      // global), so we can't actually assemble a string on the fly.
      return toQualString();
        */
    }
    
    pub fn ns(&self) -> Symbol {
        
        todo!();
        /*
            return globalStrings().ns(*this);
        */
    }
    
    pub fn domain_string(&self) -> String {
        
        todo!();
        /*
            return domain_prefix() + ns().toUnqualString();
        */
    }
    
    pub fn from_domain_and_unqual_string(&mut self, 
        d: &String,
        s: &String) -> Symbol {
        
        todo!();
        /*
            if (d.compare(0, domain_prefix().size(), domain_prefix()) != 0) {
        ostringstream ss;
        ss << "Symbol: domain string is expected to be prefixed with '"
           << domain_prefix() << "', e.g. 'org.pytorch.aten'";
        throw runtime_error(ss.str());
      }
      string qualString = d.substr(domain_prefix().size()) + "::" + s;
      return fromQualString(qualString);
        */
    }
}
