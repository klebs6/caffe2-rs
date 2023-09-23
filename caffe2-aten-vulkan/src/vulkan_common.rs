crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/VulkanCommon.h]

#[macro_export] macro_rules! up_div {
    ($x:ident, $y:ident) => {
        /*
                (((x) + (y) - (1)) / (y))
        */
    }
}

#[macro_export] macro_rules! round_up {
    ($x:ident, $y:ident) => {
        /*
                (((x) + (y) - (1)) / (y) * (y))
        */
    }
}

#[macro_export] macro_rules! align_up4 {
    ($x:ident) => {
        /*
                ROUND_UP((x), 4)
        */
    }
}

pub struct ContextConv2D {
    weight_prepacked_vulkan: Tensor,
    bias_vulkan:             Option<Tensor>,
    weight_size:             [i64; 4],
    padding:                 [i64; 2],
    stride:                  [i64; 2],
    dilation:                [i64; 2],
    groups:                  i64,
    output_min:              f32,
    output_max:              f32,
}

pub mod context_conv2d {

    pub const K_MIN: f32 = -f32::infinity;
    pub const K_MAX: f32 = f32::infinity;
}

impl ContextConv2D {
    
    pub fn new(
        weight_prepacked_vulkan: Tensor,
        bias_vulkan:             Option<Tensor>,
        weight_size:             [i64; 4],
        padding:                 [i64; 2],
        stride:                  [i64; 2],
        dilation:                [i64; 2],
        groups:                  i64,
        output_min:              f32,
        output_max:              f32) -> Self {
    
        todo!();
        /*


            : weight_prepacked_vulkan_(move(weight_prepacked_vulkan)),
            bias_vulkan_(move(bias_vulkan)),
            weight_size_(weight_size),
            padding_(padding),
            stride_(stride),
            dilation_(dilation),
            groups_(groups),
            output_min_(output_min),
            output_max_(output_max)
        */
    }
}

#[inline] pub fn safe_downcast_internal<To, From>(v: From) -> To {

    todo!();
        /*
            typedef common_type_t<From, To> Type;
      constexpr Type min{static_cast<Type>(numeric_limits<To>::lowest())};
      constexpr Type max{static_cast<Type>(To::max)};
      TORCH_CHECK(min <= v && v <= max, "Cast failed: out of range");
      return static_cast<To>(v);
        */
}

#[inline] pub fn is_signed_to_unsigned<To, From>() -> bool {

    todo!();
        /*
            return is_signed<From>::value && is_unsigned<To>::value;
        */
}

#[inline] pub fn safe_downcast(v: From) -> To {
    
    //template < typename To, typename From, enable_if_t<!is_signed_to_unsigned<To, From>(), bool> = true>
    todo!();
    /*
        return safe_downcast_internal<To, From>(v);
    */

    //template < typename To, typename From, enable_if_t<is_signed_to_unsigned<To, From>(), bool> = true>
    #[inline] pub fn safe_downcast(v: From) -> To {
        
        todo!();
            /*
                TORCH_CHECK(v >= From{}, "Cast failed: negative signed to unsigned");
          return safe_downcast_internal<To, From>(v);
            */
    }
}
