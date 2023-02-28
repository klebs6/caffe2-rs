crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/util/Bitset.h]

pub type BitsetType = i64;

/**
  | This is a simple bitset class with sizeof(long
  | long int) bits.
  | 
  | You can set bits, unset bits, query bits
  | by index, and query for the first set
  | bit.
  | 
  | Before using this class, please also
  | take a look at bitset, which has more
  | functionality and is more generic.
  | It is probably a better fit for your use
  | case. The sole reason for utils::bitset
  | to exist is that bitset misses a find_first_set()
  | method.
  |
  */
#[derive(Debug,PartialEq,Eq)]
pub struct BitSet {
    bitset: BitsetType,
}

impl BitSet {

    pub fn NUM_BITS() -> usize {
        
        todo!();
        /*
            return 8 * sizeof(bitset_type);
        */
    }
    
    pub fn new() -> Self {
    
        todo!();
        /*
        : bitset(0),

        
        */
    }
    
    
    pub fn set(&mut self, index: usize)  {
        
        todo!();
        /*
            bitset_ |= (static_cast<long long int>(1) << index);
        */
    }
    
    
    pub fn unset(&mut self, index: usize)  {
        
        todo!();
        /*
            bitset_ &= ~(static_cast<long long int>(1) << index);
        */
    }
    
    
    pub fn get(&self, index: usize) -> bool {
        
        todo!();
        /*
            return bitset_ & (static_cast<long long int>(1) << index);
        */
    }
    
    
    pub fn is_entirely_unset(&self) -> bool {
        
        todo!();
        /*
            return 0 == bitset_;
        */
    }

    /**
      | Call the given functor with the index
      | of each bit that is set
      |
      */
    pub fn for_each_set_bit<Func>(&self, func: Func)  {
    
        todo!();
        /*
            bitset cur = *this;
        size_t index = cur.find_first_set();
        while (0 != index) {
          // -1 because find_first_set() is not one-indexed.
          index -= 1;
          func(index);
          cur.unset(index);
          index = cur.find_first_set();
        }
        */
    }
 
    /**
      | Return the index of the first set bit. The
      | returned index is one-indexed (i.e. if the
      | very first bit is set, this function
      | returns '1'), and a return of '0' means
      | that there was no bit set.
      |
      */
    pub fn find_first_set(&self) -> usize {
        
        todo!();
        /*
            #if defined(_MSC_VER) && defined(_M_X64)
        unsigned long result;
        bool has_bits_set = (0 != _BitScanForward64(&result, bitset_));
        if (!has_bits_set) {
          return 0;
        }
        return result + 1;
    #elif defined(_MSC_VER) && defined(_M_IX86)
        unsigned long result;
        if (static_cast<uint32_t>(bitset_) != 0) {
          bool has_bits_set =
              (0 != _BitScanForward(&result, static_cast<uint32_t>(bitset_)));
          if (!has_bits_set) {
            return 0;
          }
          return result + 1;
        } else {
          bool has_bits_set =
              (0 != _BitScanForward(&result, static_cast<uint32_t>(bitset_ >> 32)));
          if (!has_bits_set) {
            return 32;
          }
          return result + 33;
        }
    #else
        return __builtin_ffsll(bitset_);
    #endif
        */
    }
}

