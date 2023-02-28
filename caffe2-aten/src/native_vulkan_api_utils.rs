crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/api/Utils.h]

#[cfg(USE_VULKAN_API)]
mod vulkan_api {
    use super::*;

    #[inline] pub fn align_down<Type>(
            number:   Type,
            multiple: Type) -> Type {

        todo!();
            /*
                return (number / multiple) * multiple;
            */
    }

    #[inline] pub fn align_up<Type>(
            number:   Type,
            multiple: Type) -> Type {

        todo!();
            /*
                return align_down(number + multiple - 1, multiple);
            */
    }

    #[inline] pub fn div_up<Type>(
            numerator:   Type,
            denominator: Type) -> Type {

        todo!();
            /*
                return (numerator + denominator - 1) / denominator;
            */
    }

    #[inline] pub fn safe_downcast<To, From>(v: From) -> To {

        todo!();
            /*
                typedef common_type_t<From, To> Type;
          constexpr Type min{static_cast<Type>(numeric_limits<To>::lowest())};
          constexpr Type max{static_cast<Type>(To::max)};
          TORCH_CHECK(min <= v && v <= max, "Cast failed: out of range!");
          return static_cast<To>(v);
            */
    }

    #[inline] pub fn is_signed_to_unsigned<To, From>() -> bool {

        todo!();
            /*
                return is_signed<From>::value && is_unsigned<To>::value;
            */
    }

    //template < typename To, typename From, enable_if_t<is_signed_to_unsigned<To, From>(), bool> = true>
    #[inline] pub fn safe_downcast(v: From) -> To {
        
        todo!();
            /*
                TORCH_CHECK(v >= From{}, "Cast failed: negative signed to unsigned!");
          return safe_downcast<To, From>(v);
            */
    }

    //template < typename To, typename From, enable_if_t<!is_signed_to_unsigned<To, From>(), bool> = true>
    #[inline] pub fn safe_downcast(v: From) -> To {
        
        todo!();
            /*
                return safe_downcast<To, From>(v);
            */
    }
}
