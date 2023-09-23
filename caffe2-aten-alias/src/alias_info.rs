crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/alias_info.h]

pub struct Unknown {}

pub type Symbol = Unknown;

/**
  | class AliasInfo
  | 
  | Data structure to hold aliasing information
  | for an `Argument`. They can be nested
  | to represent aliasing information
  | on contained types.
  | 
  | There is a `beforeSet` which describes
  | the aliasing information before the
  | operator executes, and an `afterSet`
  | that describes aliasing info after
  | execution.
  |
  */
pub struct AliasInfo {
    before_sets:     HashSet<Symbol>,
    after_sets:      HashSet<Symbol>,
    contained_types: Vec<AliasInfo>,
    is_write:        bool, // default = false
}

impl AliasInfo {

    /**
      | Symbol for the set that can alias anything
      |
      */
    pub fn wildcard_set() -> Symbol {
        
        todo!();
        /*
            static const Symbol wc = Symbol::fromQualString("alias::*");
        return wc;
        */
    }
    
    pub fn set_is_write(&mut self, is_write: bool)  {
        
        todo!();
        /*
            isWrite_ = isWrite;
        */
    }
    
    pub fn is_write(&self) -> bool {
        
        todo!();
        /*
            return isWrite_;
        */
    }
    
    pub fn add_before_set(&mut self, alias_set: Symbol)  {
        
        todo!();
        /*
            beforeSets_.insert(aliasSet);
        */
    }
    
    pub fn add_after_set(&mut self, alias_set: Symbol)  {
        
        todo!();
        /*
            afterSets_.insert(aliasSet);
        */
    }
    
    pub fn before_sets(&self) -> &HashSet<Symbol> {
        
        todo!();
        /*
            return beforeSets_;
        */
    }
    
    pub fn after_sets(&self) -> &HashSet<Symbol> {
        
        todo!();
        /*
            return afterSets_;
        */
    }
    
    pub fn before_set(&self) -> Symbol {
        
        todo!();
        /*
            AT_ASSERT(beforeSets_.size() == 1);
        return *beforeSets_.begin();
        */
    }
    
    pub fn is_wildcard_before(&self) -> bool {
        
        todo!();
        /*
            return beforeSets_.count(wildcardSet()) != 0;
        */
    }
    
    pub fn is_wildcard_after(&self) -> bool {
        
        todo!();
        /*
            return afterSets_.count(wildcardSet()) != 0;
        */
    }

    /**
      | the alias info for the contained types
      | of the type e.g. if this is an annotation
      | on List[T], `sets` refers to the alias
      | sets that the list may be in while containedTypes()[0]
      | refers to the sets that members of the
      | list may be in
      |
      */
    pub fn add_contained_type(&mut self, alias_info: AliasInfo)  {
        
        todo!();
        /*
            containedTypes_.push_back(std::move(aliasInfo));
        */
    }
    
    pub fn contained_types(&self) -> &Vec<AliasInfo> {
        
        todo!();
        /*
            return containedTypes_;
        */
    }
}

impl PartialEq<AliasInfo> for AliasInfo {
    
    fn eq(&self, other: &AliasInfo) -> bool {
        todo!();
        /*
            return lhs.isWrite() == rhs.isWrite()
          && lhs.beforeSets() == rhs.beforeSets()
          && lhs.afterSets() == rhs.afterSets()
          && lhs.containedTypes() == rhs.containedTypes();
        */
    }
}

impl fmt::Display for AliasInfo {
    
    /**
      | this does match the way things are represented
      | in the schema
      |
      */
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            out << "(";
      bool first = true;
      for (const auto& set : aliasInfo.beforeSets()) {
        if (first) {
          first = false;
        } else {
          out << "|";
        }
        out << set.toUnqualString();
      }
      if (aliasInfo.isWrite()) {
        out << "!";
      }
      if (aliasInfo.beforeSets() != aliasInfo.afterSets()) {
        out << " -> ";
        first = true;
        for (const auto& set : aliasInfo.afterSets()) {
          if (first) {
            first = false;
          } else {
            out << "|";
          }
          out << set.toUnqualString();
        }
      }
      out << ")";
      return out;
        */
    }
}
