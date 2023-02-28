/*!
  | This file contains functionality to
  | take a C++ function and infer its
  | FunctionSchema.
  |
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/op_registration/infer_schema.h]

/**
  | The templated inference code creates
  | `ArgumentDef` instead of `Argument`, because
  | that can be constructed at compile time and
  | has a much smaller binary size than having
  | calls to `Argument` constructors in the
  | template.
  |
  | Creating `Argument` objects from `ArgumentDef`
  | can then be done at runtime in a non-templated
  | way.
  */
lazy_static!{
    /*
    struct ArgumentDef final {

      using GetTypeFn = TypePtr();
      GetTypeFn* getTypeFn;
      constexpr ArgumentDef(): getTypeFn(nullptr) {}
      explicit constexpr ArgumentDef(GetTypeFn *getTypeFn): getTypeFn(getTypeFn) {}
    };
    */
}

lazy_static!{
    /*
    template<bool V>
    struct bool_t {};
    template<> struct bool_t<true> : true_type {};
    template<> struct bool_t<false> : false_type {};

    /// Checks the static C++ types `Types` for correctness to catch common error cases.
    template <class... Types>
    constexpr int checkStaticTypes() {
     // Give nice error messages for some of the common error cases.
     // Use a LOUD ERROR MESSAGE SO USERS SEE THE STATIC_ASSERT
     static_assert(conjunction<
         bool_t<!is_integral<Types>::value || is_same<Types, i64>::value || is_same<Types, bool>::value>...
       >::value, "INVALID TYPE: Only i64 and bool are supported as an integral argument type");
     static_assert(conjunction<
         bool_t<!is_same<Types, float>::value>...
       >::value, "INVALID TYPE: float is not supported as an argument type, use double instead");
     return 0;
    }

    template <typename... Ts, usize... Is>
    constexpr array<ArgumentDef, sizeof...(Ts)> createArgumentVectorFromTypes(index_sequence<Is...>) {
      return (
        // Check types for common errors
        checkStaticTypes<Ts...>(),

        // Create the return value
        array<ArgumentDef, sizeof...(Ts)>{ArgumentDef(&getTypePtr_<decay_t<Ts>>::call)...}
      );
    }

    /// Creates a vector of `ArgumentDef` from a list of C++ types that are specified
    /// as template arguments.
    template<class ParameterTypes> struct createArguments final {};
    template<class... ParameterTypes>
    struct createArguments<typelist::typelist<ParameterTypes...>> final {
      static constexpr array<ArgumentDef, sizeof...(ParameterTypes)> call() {
        return createArgumentVectorFromTypes<ParameterTypes...>(
            make_index_sequence<sizeof...(ParameterTypes)>()
        );
      }
    };

    /// Creates a vector of `ArgumentDef` from a list of C++ types that are specified
    /// as a tuple (i.e. in the way c10 kernels return values).
    /// It can be a tuple<A, B, C> if there's three output arguments with types A, B, C.
    /// It can be an empty tuple<>, or void for kernels that don't return anything.
    /// It can be a single type A (i.e. no tuple) for the case where a kernel just
    /// returns one value.
    template<class ReturnTypeTuple, class Enable = void> struct createReturns final {};

    template<class... ReturnTypes>
    struct createReturns<tuple<ReturnTypes...>, void> final {
      static constexpr array<ArgumentDef, sizeof...(ReturnTypes)> call() {
        return createArgumentVectorFromTypes<ReturnTypes...>(
            make_index_sequence<sizeof...(ReturnTypes)>()
        );
      }
    };

    template<class ReturnType>
    struct createReturns<ReturnType, enable_if_t<!is_same<void, ReturnType>::value && !is_instantiation_of<tuple, ReturnType>::value>> final {
      static constexpr array<ArgumentDef, 1> call() {
        return createReturns<tuple<ReturnType>>::call();
      }
    };

    template<>
    struct createReturns<void, void> final {
      static constexpr array<ArgumentDef, 0> call() {
        return createReturns<tuple<>>::call();
      }
    };
    */
}

pub struct CreateSingleReturn<ReturnType> {

}

impl CreateSingleReturn<ReturnType> {
    
    pub fn call() -> [ArgumentDef; 1] {
        
        todo!();
        /*
            return createArgumentVectorFromTypes<ReturnType>(make_index_sequence<1>());
        */
    }
}

/**
  | Creates a `FunctionSchema` object from
  | a `FunctionTraits` type for
  | a function. Flattens tuple returns into
  | multiple return types
  */
pub fn create_function_schema_from_traits_flattened_returns<FunctionTraits>() -> FunctionSchema {

    todo!();
        /*
            using ReturnType = typename FunctionTraits::return_type;
     using ParameterTypes = typename FunctionTraits::parameter_types;

     // arguments and returns are computed into a array at compile time and embedded into the binary.
     // The only code executed at runtime here is the one that creates a vector
     // of the arguments/returns from the array.
     constexpr auto arguments = createArguments<ParameterTypes>::call();
     constexpr auto returns = createReturns<ReturnType>::call();

     return make_function_schema(arguments, returns);
        */
}

/**
  | Creates a `FunctionSchema` object from
  | a `FunctionTraits` type for
  | a function. Preserves tuple returns as a Tuple
  | return type
  */
pub fn create_function_schema_from_traits_single_return<FunctionTraits>(
        name:          String,
        overload_name: String) -> FunctionSchema {

    todo!();
        /*
            using ReturnType = typename FunctionTraits::return_type;
     using ParameterTypes = typename FunctionTraits::parameter_types;

     // arguments and returns are computed into a array at compile time and embedded into the binary.
     // The only code executed at runtime here is the one that creates a vector
     // of the arguments/returns from the array.
     constexpr auto arguments = createArguments<ParameterTypes>::call();
     constexpr auto returns = createSingleReturn<ReturnType>::call();

     return make_function_schema(move(name), move(overload_name), arguments, returns);
        */
}

pub fn infer_function_schema_flattened_returns<FuncType>() -> FunctionSchema {

    todo!();
        /*
            return infer_schema::createFunctionSchemaFromTraitsFlattenedReturns<infer_function_traits_t<FuncType>>();
        */
}

pub fn infer_function_schema_single_return<FuncType>(
        name:          String,
        overload_name: String) -> FunctionSchema {

    todo!();
        /*
            return infer_schema::createFunctionSchemaFromTraitsSingleReturn<infer_function_traits_t<FuncType>>(move(name), move(overload_name));
        */
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/op_registration/infer_schema.cpp]

pub fn create_argument_vector(args: &[ArgumentDef]) -> Vec<Argument> {
    
    todo!();
        /*
            vector<Argument> result;
      result.reserve(args.size());
      for (usize i = 0; i < args.size(); ++i) {
        // Arguments are named "_<index>"
        result.push_back(Argument("_" + to_string(i), (*args[i].getTypeFn)()));
      }
      return result;
        */
}

/**
  | This is intentionally a separate function and
  | in a .cpp file because then the template is
  | smaller and that benefits binary size
  |
  */
pub fn make_function_schema_a(
    name:          String,
    overload_name: String,
    arguments:     &[ArgumentDef],
    returns:       &[ArgumentDef]) -> FunctionSchema {
    
    todo!();
        /*
            return FunctionSchema(move(name), move(overload_name), createArgumentVector(arguments), createArgumentVector(returns));
        */
}

pub fn make_function_schema_b(
    arguments: &[ArgumentDef],
    returns:   &[ArgumentDef]) -> FunctionSchema {
    
    todo!();
        /*
            return make_function_schema("", "", arguments, returns);
        */
}

pub fn find_schema_differences(
    lhs: &FunctionSchema,
    rhs: &FunctionSchema) -> Option<String> {

    todo!();
        /*
      if (lhs.arguments().size() != rhs.arguments().size()) {
        return "The number of arguments is different. " + to_string(lhs.arguments().size()) +
                 " vs " + to_string(rhs.arguments().size()) + ".";
      }
      if (lhs.returns().size() != rhs.returns().size()) {
        return "The number of returns is different. " + to_string(lhs.returns().size()) +
                 " vs " + to_string(rhs.returns().size());
      }

      for (usize i = 0; i < lhs.arguments().size(); ++i) {
        if (*lhs.arguments()[i].type() != *rhs.arguments()[i].type()) {
          return "Type mismatch in argument " + to_string(i+1) + ": " + lhs.arguments()[i].type()->str() +
                   " vs " + rhs.arguments()[i].type()->str();
        }
      }

      for (usize i = 0; i < lhs.returns().size(); ++i) {
        if (*lhs.returns()[i].type() != *rhs.returns()[i].type()) {
          return "Type mismatch in return " + to_string(i+1) + ": " + lhs.returns()[i].type()->str() +
                   " vs " + rhs.returns()[i].type()->str();
        }
      }

      // no differences found
      return nullopt;
        */
}
