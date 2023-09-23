/*!
 | Note [Philox Engine implementation]
 | ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 | Originally implemented in PyTorch's fusion compiler
 | Refer to: http://www.thesalmons.org/john/random123/papers/random123sc11.pdf
 | for details regarding the engine.
 |
 | Note that currently this implementation of the
 | philox engine is not used anywhere except for
 | tests in cpu_generator_test.cpp. However, this
 | engine will replace curandStatePhilox4_32_10_t
 | in the future.
 |
 | The philox engine takes a seed value,
 | a subsequeunce for starting the generation and
 | an offset for the subsequence.
 |
 | Think of this engine as an algorithm producing
 | a huge array. We are parallelizing this array
 | by partitioning the huge array and assigning
 | a thread index to each partition. 
 |
 | In other words, each seed value (there are 2^64
 | possible seed values) gives a sub array of size
 | 2^128 (each element in that array is a 128 bit
 | number). 
 |
 | Reasoning behind the array being of size 2^128
 | is, there are 2^64 possible thread index value
 | and there is an array of size 2^64 for each of
 | those thread index. Hence 2^64 * 2^64 = 2^128
 | for each seed value.
 |
 | In short, this generator can produce 2^64 (seed
 | values) * 2^128 (number of elements in an array
 | given by a seed value) = 2^192 values.
 |
 | Arguments:
 |
 | seed:        Seed values could be any number
 |              from 0 to 2^64-1.
 |
 | subsequence: Subsequence is just the cuda
 |              thread indexing with: - blockIdx.x
 |              * blockDim.x + threadIdx.x
 |
 | offset:      The offset variable in
 |              PhiloxEngine  decides how many
 |              128-bit random numbers to skip
 |              (i.e. how many groups of 4, 32-bit
 |              numbers to skip) and hence really
 |              decides the total number of
 |              randoms that can be achieved
 |              for the given subsequence.
 */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/PhiloxRNGEngine.h]

/// typedefs for holding vector data
pub type UINT4   = [u32; 4];
pub type UINT2   = [u32; 2];
pub type DOUBLE2 = [f64; 2];
pub type FLOAT2  = [f32; 2];

pub struct PhiloxEngine {
    counter: UINT4,
    output:  UINT4,
    key:     UINT2,
    state:   u32,
}

pub mod philox_engine {

    pub const PHILOX10A: u32 = 0x9E3779B9;
    pub const PHILOX10B: u32 = 0xBB67AE85;
    pub const PHILOXSA:  u32 = 0xD2511F53;
    pub const PHILOXSB:  u32 = 0xCD9E8D57;
}

pub type Philox4_32_10 = PhiloxEngine;

impl PhiloxEngine {
    
    pub fn new(
        seed:        u64,
        subsequence: u64,
        offset:      u64) -> Self {

        let seed:        u64 = seed.unwrap_or(67280421310721);
        let subsequence: u64 = subsequence.unwrap_or(0);
        let offset:      u64 = offset.unwrap_or(0);

        todo!();
        /*


            key[0] = static_cast<u32>(seed);
        key[1] = static_cast<u32>(seed >> 32);
        counter = UINT4(0);
        counter[2] = static_cast<u32>(subsequence);
        counter[3] = static_cast<u32>(subsequence >> 32);
        STATE = 0;
        incr_n(offset);
        */
    }

    /**
      | Produces a unique 32-bit pseudo random
      | number on every invocation
      |
      */
    #[inline] pub fn invoke(&mut self) -> u32 {
        
        todo!();
        /*
            if(STATE == 0) {
          UINT4 counter_ = counter;
          UINT2 key_ = key;

          counter_ = single_round(counter_, key_);
          key_[0] += (kPhilox10A); key_[1] += (kPhilox10B);
          counter_ = single_round(counter_, key_);
          key_[0] += (kPhilox10A); key_[1] += (kPhilox10B);
          counter_ = single_round(counter_, key_);
          key_[0] += (kPhilox10A); key_[1] += (kPhilox10B);
          counter_ = single_round(counter_, key_);
          key_[0] += (kPhilox10A); key_[1] += (kPhilox10B);
          counter_ = single_round(counter_, key_);
          key_[0] += (kPhilox10A); key_[1] += (kPhilox10B);
          counter_ = single_round(counter_, key_);
          key_[0] += (kPhilox10A); key_[1] += (kPhilox10B);
          counter_ = single_round(counter_, key_);
          key_[0] += (kPhilox10A); key_[1] += (kPhilox10B);
          counter_ = single_round(counter_, key_);
          key_[0] += (kPhilox10A); key_[1] += (kPhilox10B);
          counter_ = single_round(counter_, key_);
          key_[0] += (kPhilox10A); key_[1] += (kPhilox10B);

          output = single_round(counter_, key_);
          incr();
        }
        u32 ret = output[STATE];
        STATE = (STATE + 1) & 3;
        return ret;
        */
    }

    /**
      | Function that Skips N 128 bit numbers
      | in a subsequence
      |
      */
    #[inline] pub fn incr_n(&mut self, n: u64)  {
        
        todo!();
        /*
            u32 nlo = static_cast<u32>(n);
        u32 nhi = static_cast<u32>(n >> 32);
        counter[0] += nlo;
        // if overflow in x has occurred, carry over to nhi
        if (counter[0] < nlo) {
          nhi++;
          // if overflow in nhi has occurred during carry over,
          // propagate that overflow to y and exit to increment z
          // otherwise return
          counter[1] += nhi;
          if(nhi != 0) {
            if (nhi <= counter[1]) {
              return;
            }
          }
        } else {
          // if overflow in y has occurred during addition,
          // exit to increment z
          // otherwise return
          counter[1] += nhi;
          if (nhi <= counter[1]) {
            return;
          }
        }
        if (++counter[2])
          return;
        ++counter[3];
        */
    }

    /**
      | Function that Skips one 128 bit number
      | in a subsequence
      |
      */
    #[inline] pub fn incr(&mut self)  {
        
        todo!();
        /*
            if (++counter[0])
          return;
        if (++counter[1])
          return;
        if (++counter[2]) {
          return;
        }
        ++counter[3];
        */
    }
    
    #[inline] pub fn mulhilo32(&mut self, 
        a:           u32,
        b:           u32,
        result_high: *mut u32) -> u32 {
        
        todo!();
        /*
            #ifdef __CUDA_ARCH__
          *result_high = __umulhi(a, b);
          return a*b;
        #else
          const u64 product = static_cast<u64>(a) * b;
          *result_high = static_cast<u32>(product >> 32);
          return static_cast<u32>(product);
        #endif
        */
    }
    
    #[inline] pub fn single_round(&mut self, 
        ctr:    UINT4,
        in_key: UINT2) -> UINT4 {
        
        todo!();
        /*
            u32 hi0;
        u32 hi1;
        u32 lo0 = mulhilo32(kPhiloxSA, ctr[0], &hi0);
        u32 lo1 = mulhilo32(kPhiloxSB, ctr[2], &hi1);
        UINT4 ret;
        ret[0] = hi1 ^ ctr[1] ^ in_key[0];
        ret[1] = lo1;
        ret[2] = hi0 ^ ctr[3] ^ in_key[1];
        ret[3] = lo0;
        return ret;
        */
    }
}
