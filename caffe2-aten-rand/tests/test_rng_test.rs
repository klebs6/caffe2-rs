crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/test/rng_test.h]

pub const INT64_MIN_VAL: Auto = i64::lowest;
pub const INT64_MAX_VAL: Auto = i64::max;

pub fn min_val_with_float<T: Float>() -> i64 {
    
    todo!();
        /*
            return int64_min_val;
        */
}

pub fn min_val_with_int<T: Integer>() -> i64 {
    
    todo!();
        /*
            return static_cast<i64>(numeric_limits<T>::lowest());
        */
}

pub fn min_from_with_float<T: Float>() -> i64 {
    
    todo!();
        /*
            return -(static_cast<i64>(1) << numeric_limits<T>::digits);
        */
}

pub fn min_from_with_int<T: Integer>() -> i64 {
    
    todo!();
        /*
            return _min_val<T>();
        */
}

pub fn max_val_with_float<T: Float>() -> i64 {
    
    todo!();
        /*
            return int64_max_val;
        */
}

pub fn max_val_with_int<T: Integer>() -> i64 {
    
    todo!();
        /*
            return static_cast<i64>(T::max);
        */
}

pub fn max_to_with_float<T: Float>() -> i64 {
    
    todo!();
        /*
            return static_cast<i64>(1) << numeric_limits<T>::digits;
        */
}

pub fn max_to_with_int<T: Integer>() -> i64 {
    
    todo!();
        /*
            return _max_val<T>();
        */
}

pub fn test_random_from_to<RNG, const S: ScalarType, T>(device: &Device)  {

    todo!();
        /*
            constexpr i64 min_val = _min_val<T>();
      constexpr i64 min_from = _min_from<T>();

      constexpr i64 max_val = _max_val<T>();
      constexpr i64 max_to = _max_to<T>();

      constexpr auto uint64_max_val = u64::max;

      vector<i64> froms;
      vector<optional<i64>> tos;
      if (is_same<T, bool>::value) {
        froms = {
          0L
        };
        tos = {
          1L,
          static_cast<optional<i64>>(nullopt)
        };
      } else if (is_signed<T>::value) {
        froms = {
          min_from,
          -42L,
          0L,
          42L
        };
        tos = {
          optional<i64>(-42L),
          optional<i64>(0L),
          optional<i64>(42L),
          optional<i64>(max_to),
          static_cast<optional<i64>>(nullopt)
        };
      } else {
        froms = {
          0L,
          42L
        };
        tos = {
          optional<i64>(42L),
          optional<i64>(max_to),
          static_cast<optional<i64>>(nullopt)
        };
      }

      const vector<u64> vals = {
        0L,
        42L,
        static_cast<u64>(max_val),
        static_cast<u64>(max_val) + 1,
        uint64_max_val
      };

      bool full_64_bit_range_case_covered = false;
      bool from_to_case_covered = false;
      bool from_case_covered = false;
      for (const i64 from : froms) {
        for (const optional<i64> to : tos) {
          if (!to.has_value() || from < *to) {
            for (const u64 val : vals) {
              auto gen = make_generator<RNG>(val);

              auto actual = Torchempty({3, 3}, TorchTensorOptions().dtype(S).device(device));
              actual.random_(from, to, gen);

              T exp;
              u64 range;
              if (!to.has_value() && from == int64_min_val) {
                exp = static_cast<i64>(val);
                full_64_bit_range_case_covered = true;
              } else {
                if (to.has_value()) {
                  range = static_cast<u64>(*to) - static_cast<u64>(from);
                  from_to_case_covered = true;
                } else {
                  range = static_cast<u64>(max_to) - static_cast<u64>(from) + 1;
                  from_case_covered = true;
                }
                if (range < (1ULL << 32)) {
                  exp = static_cast<T>(static_cast<i64>((static_cast<u32>(val) % range + from)));
                } else {
                  exp = static_cast<T>(static_cast<i64>((val % range + from)));
                }
              }
              ASSERT_TRUE(from <= exp);
              if (to.has_value()) {
                ASSERT_TRUE(static_cast<i64>(exp) < *to);
              }
              const auto expected = Torchfull_like(actual, exp);
              if (is_same<T, bool>::value) {
                ASSERT_TRUE(Torchallclose(actual.toType(TorchkInt), expected.toType(TorchkInt)));
              } else {
                ASSERT_TRUE(Torchallclose(actual, expected));
              }
            }
          }
        }
      }
      if (is_same<T, i64>::value) {
        ASSERT_TRUE(full_64_bit_range_case_covered);
      }
      ASSERT_TRUE(from_to_case_covered);
      ASSERT_TRUE(from_case_covered);
        */
}

pub fn test_random<RNG, const S: ScalarType, T>(device: &Device)  {

    todo!();
        /*
            const auto max_val = _max_val<T>();
      const auto uint64_max_val = u64::max;

      const vector<u64> vals = {
        0L,
        42L,
        static_cast<u64>(max_val),
        static_cast<u64>(max_val) + 1,
        uint64_max_val
      };

      for (const u64 val : vals) {
        auto gen = make_generator<RNG>(val);

        auto actual = Torchempty({3, 3}, TorchTensorOptions().dtype(S).device(device));
        actual.random_(gen);

        u64 range;
        if (is_floating_point<T>::value) {
          range = static_cast<u64>((1ULL << numeric_limits<T>::digits) + 1);
        } else if (is_same<T, bool>::value) {
          range = 2;
        } else {
          range = static_cast<u64>(T::max) + 1;
        }
        T exp;
        if (is_same<T, double>::value || is_same<T, i64>::value) {
          exp = val % range;
        } else {
          exp = static_cast<u32>(val) % range;
        }

        ASSERT_TRUE(0 <= static_cast<i64>(exp));
        ASSERT_TRUE(static_cast<u64>(exp) < range);

        const auto expected = Torchfull_like(actual, exp);
        if (is_same<T, bool>::value) {
          ASSERT_TRUE(Torchallclose(actual.toType(TorchkInt), expected.toType(TorchkInt)));
        } else {
          ASSERT_TRUE(Torchallclose(actual, expected));
        }
      }
        */
}
