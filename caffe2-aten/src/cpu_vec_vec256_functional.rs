/*!
  | DO NOT DEFINE STATIC DATA IN THIS HEADER!
  | 
  | See Note [Do not compile initializers
  | with AVX]
  |
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/cpu/vec/vec256/functional.h]

// TODO: Make this more efficient
#[inline] pub fn vec_reduce_all<Scalar, Op>(
        vec_fun: &Op,
        acc_vec: Vectorized<Scalar>,
        size:    i64) -> Scalar {

    todo!();
        /*
            using Vec = vec::Vectorized<Scalar>;
      Scalar acc_arr[Vec::size()];
      acc_vec.store(acc_arr);
      for (i64 i = 1; i < size; i++) {
        std::array<Scalar, Vec::size()> acc_arr_next = {0};
        acc_arr_next[0] = acc_arr[i];
        Vec acc_vec_next = Vec::loadu(acc_arr_next.data());
        acc_vec = vec_fun(acc_vec, acc_vec_next);
      }
      acc_vec.store(acc_arr);
      return acc_arr[0];
        */
}

#[inline] pub fn reduce_all<Scalar, Op>(
        vec_fun: &Op,
        data:    *const Scalar,
        size:    i64) -> Scalar {

    todo!();
        /*
            using Vec = vec::Vectorized<Scalar>;
      if (size < Vec::size())
        return vec_reduce_all(vec_fun, Vec::loadu(data, size), size);
      i64 d = Vec::size();
      Vec acc_vec = Vec::loadu(data);
      for (; d < size - (size % Vec::size()); d += Vec::size()) {
        Vec data_vec = Vec::loadu(data + d);
        acc_vec = vec_fun(acc_vec, data_vec);
      }
      if (size - d > 0) {
        Vec data_vec = Vec::loadu(data + d, size - d);
        acc_vec = Vec::set(acc_vec, vec_fun(acc_vec, data_vec), size - d);
      }
      return vec_reduce_all(vec_fun, acc_vec, Vec::size());
        */
}

/**
  | similar to reduce_all, but reduces
  | into two outputs
  |
  */
#[inline] pub fn reduce2_all<Scalar, Op1, Op2>(
        vec_fun1: &Op1,
        vec_fun2: &Op2,
        data:     *const Scalar,
        size:     i64) -> (Scalar,Scalar) {

    todo!();
        /*
            using Vec = vec::Vectorized<Scalar>;
      if (size < Vec::size()) {
        auto loaded_data = Vec::loadu(data, size);
        return std::pair<Scalar, Scalar>(
          vec_reduce_all(vec_fun1, loaded_data, size),
          vec_reduce_all(vec_fun2, loaded_data, size));
      }
      i64 d = Vec::size();
      Vec acc_vec1 = Vec::loadu(data);
      Vec acc_vec2 = Vec::loadu(data);
      for (; d < size - (size % Vec::size()); d += Vec::size()) {
        Vec data_vec = Vec::loadu(data + d);
        acc_vec1 = vec_fun1(acc_vec1, data_vec);
        acc_vec2 = vec_fun2(acc_vec2, data_vec);
      }
      if (size - d > 0) {
        Vec data_vec = Vec::loadu(data + d, size - d);
        acc_vec1 = Vec::set(acc_vec1, vec_fun1(acc_vec1, data_vec), size - d);
        acc_vec2 = Vec::set(acc_vec2, vec_fun2(acc_vec2, data_vec), size - d);
      }
      return std::pair<Scalar, Scalar>(
        vec_reduce_all(vec_fun1, acc_vec1, Vec::size()),
        vec_reduce_all(vec_fun2, acc_vec2, Vec::size()));
        */
}

#[inline] pub fn map_reduce_all<Scalar, MapOp, ReduceOp>(
        map_fun: &MapOp,
        red_fun: &ReduceOp,
        data:    *mut Scalar,
        size:    i64) -> Scalar {

    todo!();
        /*
            using Vec = vec::Vectorized<Scalar>;
      if (size < Vec::size())
        return vec_reduce_all(red_fun, map_fun(Vec::loadu(data, size)), size);
      i64 d = Vec::size();
      Vec acc_vec = map_fun(Vec::loadu(data));
      for (; d < size - (size % Vec::size()); d += Vec::size()) {
        Vec data_vec = Vec::loadu(data + d);
        data_vec = map_fun(data_vec);
        acc_vec = red_fun(acc_vec, data_vec);
      }
      if (size - d > 0) {
        Vec data_vec = Vec::loadu(data + d, size - d);
        data_vec = map_fun(data_vec);
        acc_vec = Vec::set(acc_vec, red_fun(acc_vec, data_vec), size - d);
      }
      return vec_reduce_all(red_fun, acc_vec, Vec::size());
        */
}

#[inline] pub fn map2_reduce_all<Scalar, MapOp, ReduceOp>(
        map_fun: &MapOp,
        red_fun: &ReduceOp,
        data:    *const Scalar,
        data2:   *const Scalar,
        size:    i64) -> Scalar {

    todo!();
        /*
            using Vec = vec::Vectorized<Scalar>;
      if (size < Vec::size()) {
        Vec data_vec = Vec::loadu(data, size);
        Vec data2_vec = Vec::loadu(data2, size);
        data_vec = map_fun(data_vec, data2_vec);
        return vec_reduce_all(red_fun, data_vec, size);
      }
      i64 d = Vec::size();
      Vec acc_vec = map_fun(Vec::loadu(data), Vec::loadu(data2));
      for (; d < size - (size % Vec::size()); d += Vec::size()) {
        Vec data_vec = Vec::loadu(data + d);
        Vec data2_vec = Vec::loadu(data2 + d);
        data_vec = map_fun(data_vec, data2_vec);
        acc_vec = red_fun(acc_vec, data_vec);
      }
      if (size - d > 0) {
        Vec data_vec = Vec::loadu(data + d, size - d);
        Vec data2_vec = Vec::loadu(data2 + d, size - d);
        data_vec = map_fun(data_vec, data2_vec);
        acc_vec = Vec::set(acc_vec, red_fun(acc_vec, data_vec), size - d);
      }
      return vec_reduce_all(red_fun, acc_vec, Vec::size());
        */
}

#[inline] pub fn map3_reduce_all<Scalar, MapOp, ReduceOp>(
        map_fun: &MapOp,
        red_fun: &ReduceOp,
        data:    *const Scalar,
        data2:   *const Scalar,
        data3:   *const Scalar,
        size:    i64) -> Scalar {

    todo!();
        /*
            using Vec = vec::Vectorized<Scalar>;
      if (size < Vec::size()) {
        Vec data_vec = Vec::loadu(data, size);
        Vec data2_vec = Vec::loadu(data2, size);
        Vec data3_vec = Vec::loadu(data3, size);
        data_vec = map_fun(data_vec, data2_vec, data3_vec);
        return vec_reduce_all(red_fun, data_vec, size);
      }

      i64 d = Vec::size();
      Vec acc_vec = map_fun(Vec::loadu(data), Vec::loadu(data2), Vec::loadu(data3));
      for (; d < size - (size % Vec::size()); d += Vec::size()) {
        Vec data_vec = Vec::loadu(data + d);
        Vec data2_vec = Vec::loadu(data2 + d);
        Vec data3_vec = Vec::loadu(data3 + d);
        data_vec = map_fun(data_vec, data2_vec, data3_vec);
        acc_vec = red_fun(acc_vec, data_vec);
      }
      if (size - d > 0) {
        Vec data_vec = Vec::loadu(data + d, size - d);
        Vec data2_vec = Vec::loadu(data2 + d, size - d);
        Vec data3_vec = Vec::loadu(data3 + d, size - d);
        data_vec = map_fun(data_vec, data2_vec, data3_vec);
        acc_vec = Vec::set(acc_vec, red_fun(acc_vec, data_vec), size - d);
      }
      return vec_reduce_all(red_fun, acc_vec, Vec::size());
        */
}

#[inline] pub fn map<Scalar, Op>(
        vec_fun:     &Op,
        output_data: *mut Scalar,
        input_data:  *const Scalar,
        size:        i64)  {

    todo!();
        /*
            using Vec = vec::Vectorized<Scalar>;
      i64 d = 0;
      for (; d < size - (size % Vec::size()); d += Vec::size()) {
        Vec output_vec = vec_fun(Vec::loadu(input_data + d));
        output_vec.store(output_data + d);
      }
      if (size - d > 0) {
        Vec output_vec = vec_fun(Vec::loadu(input_data + d, size - d));
        output_vec.store(output_data + d, size - d);
      }
        */
}

#[inline] pub fn map2<Scalar, Op>(
        vec_fun:     &Op,
        output_data: *mut Scalar,
        input_data:  *const Scalar,
        input_data2: *const Scalar,
        size:        i64)  {

    todo!();
        /*
            using Vec = vec::Vectorized<Scalar>;
      i64 d = 0;
      for (; d < size - (size % Vec::size()); d += Vec::size()) {
        Vec data_vec = Vec::loadu(input_data + d);
        Vec data_vec2 = Vec::loadu(input_data2 + d);
        Vec output_vec = vec_fun(data_vec, data_vec2);
        output_vec.store(output_data + d);
      }
      if (size - d > 0) {
        Vec data_vec = Vec::loadu(input_data + d, size - d);
        Vec data_vec2 = Vec::loadu(input_data2 + d, size - d);
        Vec output_vec = vec_fun(data_vec, data_vec2);
        output_vec.store(output_data + d, size - d);
      }
        */
}

#[inline] pub fn map3<Scalar, Op>(
        vec_fun:     &Op,
        output_data: *mut Scalar,
        input_data1: *const Scalar,
        input_data2: *const Scalar,
        input_data3: *const Scalar,
        size:        i64)  {

    todo!();
        /*
            using Vec = vec::Vectorized<Scalar>;
      i64 d = 0;
      for (; d < size - (size % Vec::size()); d += Vec::size()) {
        Vec data_vec1 = Vec::loadu(input_data1 + d);
        Vec data_vec2 = Vec::loadu(input_data2 + d);
        Vec data_vec3 = Vec::loadu(input_data3 + d);
        Vec output_vec = vec_fun(data_vec1, data_vec2, data_vec3);
        output_vec.store(output_data + d);
      }
      if (size - d > 0) {
        Vec data_vec1 = Vec::loadu(input_data1 + d, size - d);
        Vec data_vec2 = Vec::loadu(input_data2 + d, size - d);
        Vec data_vec3 = Vec::loadu(input_data3 + d, size - d);
        Vec output_vec = vec_fun(data_vec1, data_vec2, data_vec3);
        output_vec.store(output_data + d, size - d);
      }
        */
}

#[inline] pub fn map4<Scalar, Op>(
        vec_fun:     &Op,
        output_data: *mut Scalar,
        input_data1: *const Scalar,
        input_data2: *const Scalar,
        input_data3: *const Scalar,
        input_data4: *const Scalar,
        size:        i64)  {

    todo!();
        /*
            using Vec = vec::Vectorized<Scalar>;
      i64 d = 0;
      for (; d < size - (size % Vec::size()); d += Vec::size()) {
        Vec data_vec1 = Vec::loadu(input_data1 + d);
        Vec data_vec2 = Vec::loadu(input_data2 + d);
        Vec data_vec3 = Vec::loadu(input_data3 + d);
        Vec data_vec4 = Vec::loadu(input_data4 + d);
        Vec output_vec = vec_fun(data_vec1, data_vec2, data_vec3, data_vec4);
        output_vec.store(output_data + d);
      }
      if (size - d > 0) {
        Vec data_vec1 = Vec::loadu(input_data1 + d, size - d);
        Vec data_vec2 = Vec::loadu(input_data2 + d, size - d);
        Vec data_vec3 = Vec::loadu(input_data3 + d, size - d);
        Vec data_vec4 = Vec::loadu(input_data4 + d, size - d);
        Vec output_vec = vec_fun(data_vec1, data_vec2, data_vec3, data_vec4);
        output_vec.store(output_data + d, size - d);
      }
        */
}
