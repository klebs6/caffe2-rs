crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/TensorNames.h]

/**
  | TensorName and TensorNames are wrappers around
  | Dimname and DimnameList that contain helper
  | functions to make writing name inference rules
  | easier.
  |
  | A TensorName represents a Dimname associated
  | with some DimnameList (from a Tensor). This
  | encapsulates all the information that is needed
  | to check if names *match* and to *unify* names.
  |
  | Definition: Two names in two tensors *match* if
  | they are equal, or if at least one of them is
  | a wildcard that can be *refined* to the other
  | name.
  |
  | Definition: unify(name, other) fails if the
  | names do not match. Otherwise, it returns the
  | most refined of name and other.
  |
  | Here is an example of checking if two names
  | match.
  |
  | tensor: Tensor[A, None]
  |
  | other: Tensor[A]
  |
  | Let's say we wish to check if tensor.names[-1]
  | matches other.names[-1].
  |
  | None (in tensor) cannot match A (in other)
  | because if the None were refined to A, `tensor`
  | would have duplicate names [A, A]. Therefore we
  | need to check tensor.names [A, None] for the
  | existence of A.
  |
  */
pub struct TensorName {
    origin:     &[Dimname],
    name:       Dimname,

    /**
      | A named tensor can have at most 64 dims.
      |
      */
    origin_idx: i32,
}

impl TensorName {
    
    pub fn new(
        origin:     &[Dimname],
        origin_idx: i32) -> Self {
    
        todo!();
        /*


            : origin_(origin),
          name_(origin[maybe_wrap_dim(origin_idx, origin.size())]),
          origin_idx_(origin_idx)
        */
    }

    /**
      | op_name is only used for error reporting.
      |
      */
    pub fn unify(&self, 
        other:   &TensorName,
        op_name: *const u8) -> &TensorName {
        
        todo!();
        /*
        
        */
    }
    
    pub fn to_dimname(&self) -> Dimname {
        
        todo!();
        /*
        
        */
    }
}

pub type TensorNameVec = SmallVector<TensorName,10>;

pub struct TensorNames {
    names: TensorNameVec,
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/TensorNames.cpp]
impl TensorNames {
    
    pub fn new(names: &[Dimname]) -> Self {
    
        todo!();
        /*
        
        */
    }

    /**
      | Create TensorNames from
      | names[start:end]. Each individual TensorName
      | stores `names`, NOT names[start:end], because
      | the original tensor's names are `names`.
      |
      */
    pub fn new(
        names: &[Dimname],
        start: i64,
        end:   i64) -> Self {
    
        todo!();
        /*


        
        */
    }

    /**
      | op_name is only used for error reporting.
      |
      */
    pub fn unify_from_right_inplace(&mut self, 
        other:   &TensorNames,
        op_name: *const u8) -> &mut TensorNames {
        let op_name: *const u8 = op_name.unwrap_or("unify");

        todo!();
        /*
        
        */
    }
    
    pub fn check_unique(&self, op_name: *const u8)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn append(&mut self, name: TensorName)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn to_dimname_vec(&self) -> Vec<Dimname> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn new(names: TensorNameVec) -> Self {
    
        todo!();
        /*
        : names(names),

            }{
        */
    }
    
    pub fn new(names: &[Dimname]) -> Self {
    
        todo!();
        /*


            names_.reserve(names.size());
      for (i64 idx = 0; idx < names.size(); ++idx) {
        names_.emplace_back(names, idx);
      }
        */
    }
    
    pub fn new(
        names: &[Dimname],
        start: i64,
        end:   i64) -> Self {
    
        todo!();
        /*


            start = maybe_wrap_dim(start, names.size());
      end = maybe_wrap_dim(end, names.size());
      names_.reserve(end - start);
      for (i64 idx = start; idx < end; ++idx) {
        names_.emplace_back(names, idx);
      }
        */
    }
    
    pub fn unify_from_right_inplace(&mut self, 
        other:   &TensorNames,
        op_name: *const u8) -> &mut TensorNames {
        
        todo!();
        /*
      usize size_diff = labs(names_.size() - other.names_.size());

      if (names_.size() > other.names_.size()) {
        for (usize idx = size_diff; idx < names_.size(); ++idx) {
          names_[idx] = names_[idx].unify(other.names_[idx - size_diff], op_name);
        }
      } else {
        // pad names_ to the same length as other.names_ before unification
        names_.insert(
            names_.begin(),
            other.names_.begin(),
            other.names_.begin() + size_diff);
        for (i64 idx = size_diff; idx < names_.size(); ++idx) {
          names_[idx] = names_[idx].unify(other.names_[idx], op_name);
        }
      }

      return *this;
        */
    }
    
    pub fn append(&mut self, name: TensorName)  {
        
        todo!();
        /*
            names_.emplace_back(name);
        */
    }
    
    pub fn check_unique(&self, op_name: *const u8)  {
        
        todo!();
        /*
            // O(N^2), but named tensors can have at most N = 64 dimensions, so this
      // doesn't matter unless benchmarking tells us it does. The alternative is
      // to create some sort of set data structure but the overhead of that
      // might dominate for small sizes.
      for (auto it = names_.begin(); it != names_.end(); ++it) {
        const auto name = it->toDimname();
        if (name.isWildcard()) continue;

        auto dup = find_if(it + 1, names_.end(),
            [&](const TensorName& other) { return other.toDimname() == name; });
        TORCH_CHECK(dup == names_.end(),
            op_name, ": ",
            "Attempted to propagate dims ", *it, " and ", *dup, " to the output, ",
            "but that would create a tensor with duplicate names [", toDimnameVec(),
            "]. Please rename your inputs with Tensor.rename to prevent this.");
      }
        */
    }
    
    pub fn to_dimname_vec(&self) -> Vec<Dimname> {
        
        todo!();
        /*
            vector<Dimname> result;
      result.reserve(names_.size());
      for (const auto& tensor_name : names_) {
        result.emplace_back(tensor_name.toDimname());
      }
      return result;
        */
    }
}

impl fmt::Display for TensorName {
    
    /**
      | Let's say the TensorName represents 'C' in
      | ['N', 'C', 'H, 'W'].
      |
      | It should print like:
      |
      | 'C' (index 1 of ['N', 'C', 'H', 'W'])
      */
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            out << tensorname.name_ << " (index ";
      out << tensorname.origin_idx_ << " of ";
      out << tensorname.origin_ << ")";
      return out;
        */
    }
}

impl TensorName {
    
    pub fn to_dimname(&self) -> Dimname {
        
        todo!();
        /*
            return name_;
        */
    }
    
    pub fn unify(&self, 
        other:   &TensorName,
        op_name: *const u8) -> &TensorName {
        
        todo!();
        /*
            // unify(None, None)
      if (name_.isWildcard() && other.name_.isWildcard()) {
        return *this;
      }

      // unify(A, A)
      if (name_ == other.name_) {
        return *this;
      }

      // unify(A, None)
      if (other.name_.isWildcard()) {
        const auto it = find(other.origin_.begin(), other.origin_.end(), name_);
        TORCH_CHECK(it == other.origin_.end(),
            op_name, ":",
            " Cannot match ", *this, " with ", other,
            " because the latter names already have ", name_, ".",
            " Are your tensors misaligned?");
        return *this;
      }

      // unify(None, A)
      if (name_.isWildcard()) {
        return other.unify(*this, op_name);
      }

      // unify(A, B)
      TORCH_CHECK(name_ == other.name_,
          op_name, ":",
          " Expected ", *this,
          " to match ", other,
          " but they do not match.");
      return *this;
        */
    }
}

