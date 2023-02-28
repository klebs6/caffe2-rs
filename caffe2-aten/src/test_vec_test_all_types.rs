// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/test/vec_test_all_types.h]

lazy_static!{
    /*
    #define CACHE_LINE 32
    #if defined(__GNUC__)
    #define CACHE_ALIGN __attribute__((aligned(CACHE_LINE)))
    #define not_inline __attribute__((noinline))
    #elif defined(_WIN32)
    #define CACHE_ALIGN __declspec(align(CACHE_LINE))
    #define not_inline __declspec(noinline)
    #else
    CACHE_ALIGN #define
    #define not_inline
    #endif
    #if defined(CPU_CAPABILITY_DEFAULT) || defined(_MSC_VER)
    #define TEST_AGAINST_DEFAULT 1
    #elif !defined(target_feature = "avx") &&  !defined(target_feature = "avx2") && !defined(CPU_CAPABILITY_VSX)
    #define TEST_AGAINST_DEFAULT 1
    #else
    #undef TEST_AGAINST_DEFAULT
    #endif
    #undef NAME_INFO
    #define STRINGIFY(x) #x
    #define TOSTRING(x) STRINGIFY(x)
    #define NAME_INFO(name) TOSTRING(name) " " TOSTRING(__FILE__) ":" TOSTRING(__LINE__)

    #define RESOLVE_OVERLOAD(...)                                  \
      [](auto&&... args) -> decltype(auto) {                       \
        return __VA_ARGS__(forward<decltype(args)>(args)...); \
      }

    #if defined(CPU_CAPABILITY_VSX) || defined(target_feature = "avx2") && (defined(__GNUC__) || defined(__GNUG__))
    #undef CHECK_DEQUANT_WITH_LOW_PRECISION
    #define CHECK_WITH_FMA 1
    #elif !defined(CPU_CAPABILITY_VSX) && !defined(target_feature = "avx2")
    #undef CHECK_DEQUANT_WITH_LOW_PRECISION
    #undef CHECK_WITH_FMA
    #else
    #define CHECK_DEQUANT_WITH_LOW_PRECISION 1
    #undef CHECK_WITH_FMA
    #endif

    template<typename T>
    using Complex = typename complex<T>;

    template <typename T>
    using VecType = typename vec::Vectorized<T>;

    using vfloat = VecType<float>;
    using vdouble = VecType<double>;
    using vcomplex = VecType<Complex<float>>;
    using vcomplexDbl = VecType<Complex<double>>;
    using vlong = VecType<i64>;
    using vint = VecType<i32>;
    using vshort = VecType<i16>;
    using vqint8 = VecType<qint8>;
    using vquint8 = VecType<quint8>;
    using vqint = VecType<qint32>;

    template <typename T>
    using ValueType = typename T::value_type;

    template <int N>
    struct BitStr
    {
        using type = uintmax_t;
    };

    template <>
    struct BitStr<8>
    {
        using type = u64;
    };

    template <>
    struct BitStr<4>
    {
        using type = u32;
    };

    template <>
    struct BitStr<2>
    {
        using type = u16;
    };

    template <>
    struct BitStr<1>
    {
        using type = u8;
    };

    template <typename T>
    using BitType = typename BitStr<sizeof(T)>::type;

    template<typename T>
    struct VecTypeHelper {
        using holdType = typename T::value_type;
        using memStorageType = typename T::value_type;
        static constexpr int holdCount = T::size();
        static constexpr int unitStorageCount = 1;
    };

    template<>
    struct VecTypeHelper<vcomplex> {
        using holdType = Complex<float>;
        using memStorageType = float;
        static constexpr int holdCount = vcomplex::size();
        static constexpr int unitStorageCount = 2;
    };

    template<>
    struct VecTypeHelper<vcomplexDbl> {
        using holdType = Complex<double>;
        using memStorageType = double;
        static constexpr int holdCount = vcomplexDbl::size();
        static constexpr int unitStorageCount = 2;
    };

    template<>
    struct VecTypeHelper<vqint8> {
        using holdType = qint8;
        using memStorageType = typename qint8::underlying;
        static constexpr int holdCount = vqint8::size();
        static constexpr int unitStorageCount = 1;
    };

    template<>
    struct VecTypeHelper<vquint8> {
        using holdType = quint8;
        using memStorageType = typename quint8::underlying;
        static constexpr int holdCount = vquint8::size();
        static constexpr int unitStorageCount = 1;
    };

    template<>
    struct VecTypeHelper<vqint> {
        using holdType = qint32;
        using memStorageType = typename qint32::underlying;
        static constexpr int holdCount = vqint::size();
        static constexpr int unitStorageCount = 1;
    };

    template <typename T>
    using UholdType = typename VecTypeHelper<T>::holdType;

    template <typename T>
    using UvalueType = typename VecTypeHelper<T>::memStorageType;

    template <class T, usize N>
    constexpr usize size(T(&)[N]) {
        return N;
    }

    template <typename Filter, typename T>
    typename enable_if_t<is_same<Filter, nullptr_t>::value, void>
    call_filter(Filter filter, T& val) {}

    template <typename Filter, typename T>
    typename enable_if_t< is_same<Filter, nullptr_t>::value, void>
    call_filter(Filter filter, T& first, T& second) { }

    template <typename Filter, typename T>
    typename enable_if_t< is_same<Filter, nullptr_t>::value, void>
    call_filter(Filter filter, T& first, T& second, T& third) {  }

    template <typename Filter, typename T>
    typename enable_if_t<
        !is_same<Filter, nullptr_t>::value, void>
        call_filter(Filter filter, T& val) {
        return filter(val);
    }

    template <typename Filter, typename T>
    typename enable_if_t<
        !is_same<Filter, nullptr_t>::value, void>
        call_filter(Filter filter, T& first, T& second) {
        return filter(first, second);
    }

    template <typename Filter, typename T>
    typename enable_if_t<
        !is_same<Filter, nullptr_t>::value, void>
        call_filter(Filter filter, T& first, T& second, T& third) {
        return filter(first, second, third);
    }

    template <typename T>
    struct DomainRange {
        T start;  // start [
        T end;    // end is not included. one could use  nextafter for including his end case for tests
    };

    template <typename T>
    struct CustomCheck {
        vector<UholdType<T>> Args;
        UholdType<T> expectedResult;
    };

    template <typename T>
    struct CheckWithinDomains {
        // each argument takes domain Range
        vector<DomainRange<T>> ArgsDomain;
        // check with error tolerance
        bool CheckWithTolerance = false;
        T ToleranceError = (T)0;
    };

    template <typename T>
    ostream& operator<<(ostream& stream, const CheckWithinDomains<T>& dmn) {
        stream << "Domain: ";
        if (dmn.ArgsDomain.size() > 0) {
            for (const DomainRange<T>& x : dmn.ArgsDomain) {
                if (is_same<T, i8>::value || is_same<T, u8>::value) {
                    stream << "\n{ " << static_cast<int>(x.start) << ", " << static_cast<int>(x.end) << " }";
                }
                else {
                    stream << "\n{ " << x.start << ", " << x.end << " }";
                }
            }
        }
        else {
            stream << "default range";
        }
        if (dmn.CheckWithTolerance) {
            stream << "\nError tolerance: " << dmn.ToleranceError;
        }
        return stream;
    }

    template <class To, class From>
    typename enable_if<
        (sizeof(To) == sizeof(From)) && is_trivially_copyable<From>::value&&
        is_trivial<To>::value,
        // this implementation requires that To is trivially default constructible
        To>::type
        bit_cast(const From& src) noexcept {
        To dst;
        memcpy(&dst, &src, sizeof(To));
        return dst;
    }

    template <class To, class T>
    To bit_cast_ptr(T* p, usize N = sizeof(To)) noexcept {
        unsigned char p1[sizeof(To)] = {};
        memcpy(p1, p, min(N, sizeof(To)));
        return bit_cast<To>(p1);
    }

    template <typename T>
    enable_if_t<is_floating_point<T>::value, bool> check_both_nan(T x,
        T y) {
        return isnan(x) && isnan(y);
    }

    template <typename T>
    enable_if_t<!is_floating_point<T>::value, bool> check_both_nan(T x,
        T y) {
        return false;
    }

    template <typename T>
    enable_if_t<is_floating_point<T>::value, bool> check_both_inf(T x,
        T y) {
        return isinf(x) && isinf(y);
    }

    template <typename T>
    enable_if_t<!is_floating_point<T>::value, bool> check_both_inf(T x,
        T y) {
        return false;
    }

    template<typename T>
    enable_if_t<!is_floating_point<T>::value, bool> check_both_big(T x, T y) {
        return false;
    }

    template<typename T>
    enable_if_t<is_floating_point<T>::value, bool> check_both_big(T x, T y) {
        T cmax = is_same<T, float>::value ? static_cast<T>(1e+30) : static_cast<T>(1e+300);
        T cmin = is_same<T, float>::value ? static_cast<T>(-1e+30) : static_cast<T>(-1e+300);
        //only allow when one is inf
        bool x_inf = isinf(x);
        bool y_inf = isinf(y);
        bool px = x > 0;
        bool py = y > 0;
        return (px && x_inf && y >= cmax) || (py && y_inf && x >= cmax) ||
            (!px && x_inf && y <= cmin) || (!py && y_inf && x <= cmin);
    }

    template<class T> struct is_complex : false_type {};

    template<class T> struct is_complex<Complex<T>> : true_type {};

    template<typename T>
    T safe_fpt_division(T f1, T f2)
    {
        //code was taken from boost
        // Avoid overflow.
        if ((f2 < static_cast<T>(1)) && (f1 > f2 * T::max)) {
            return T::max;
        }
        // Avoid underflow.
        if ((f1 == static_cast<T>(0)) ||
            ((f2 > static_cast<T>(1)) && (f1 < f2 * numeric_limits<T>::min()))) {
            return static_cast<T>(0);
        }
        return f1 / f2;
    }

    template<class T>
    enable_if_t<is_floating_point<T>::value, bool>
    nearlyEqual(T a, T b, T tolerance) {
        if (check_both_nan<T>(a, b)) return true;
        if (check_both_big(a, b)) return true;
        T absA = abs(a);
        T absB = abs(b);
        T diff = abs(a - b);
        if (diff <= tolerance) {
            return true;
        }
        T d1 = safe_fpt_division<T>(diff, absB);
        T d2 = safe_fpt_division<T>(diff, absA);
        return (d1 <= tolerance) || (d2 <= tolerance);
    }

    template<class T>
    enable_if_t<!is_floating_point<T>::value, bool>
    nearlyEqual(T a, T b, T tolerance) {
        return a == b;
    }

    template <typename T>
    T reciprocal(T x) {
        return 1 / x;
    }

    template <typename T>
    T rsqrt(T x) {
        return 1 / sqrt(x);
    }

    template <typename T>
    T frac(T x) {
      return x - trunc(x);
    }

    template <class T>
    T maximum(const T& a, const T& b) {
        return (a > b) ? a : b;
    }

    template <class T>
    T minimum(const T& a, const T& b) {
        return (a < b) ? a : b;
    }

    template <class T>
    T clamp(const T& a, const T& min, const T& max) {
        return a < min ? min : (a > max ? max : a);
    }

    template <class T>
    T clamp_max(const T& a, const T& max) {
        return a > max ? max : a;
    }

    template <class T>
    T clamp_min(const T& a, const T& min) {
        return a < min ? min : a;
    }

    template <class VT, usize N>
    void copy_interleave(VT(&vals)[N], VT(&interleaved)[N]) {
        static_assert(N % 2 == 0, "should be even");
        auto ptr1 = vals;
        auto ptr2 = vals + N / 2;
        for (usize i = 0; i < N; i += 2) {
            interleaved[i] = *ptr1++;
            interleaved[i + 1] = *ptr2++;
        }
    }

    template <typename T>
    enable_if_t<is_floating_point<T>::value, bool> is_zero(T val) {
        return fpclassify(val) == FP_ZERO;
    }

    template <typename T>
    enable_if_t<!is_floating_point<T>::value, bool> is_zero(T val) {
        return val == 0;
    }

    template <typename T>
    void filter_clamp(T& f, T& s, T& t) {
        if (t < s) {
            swap(s, t);
        }
    }

    template <typename T>
    enable_if_t<is_floating_point<T>::value, void> filter_fmod(T& a, T& b) {
        // This is to make sure fmod won't cause overflow when doing the div
        if (abs(b) < (T)1) {
          b = b < (T)0 ? (T)-1 : T(1);
        }
    }

    template <typename T>
    enable_if_t<is_floating_point<T>::value, void> filter_fmadd(T& a, T& b, T& c) {
        // This is to setup a limit to make sure fmadd (a * b + c) won't overflow
        T max = sqrt(T::max) / T(2.0);
        T min = ((T)0 - max);

        if (a > max) a = max;
        else if (a < min) a = min;

        if (b > max) b = max;
        else if (b < min) b = min;

        if (c > max) c = max;
        else if (c < min) c = min;
    }

    template <typename T>
    void filter_zero(T& val) {
        val = is_zero(val) ? (T)1 : val;
    }
    template <typename T>
    enable_if_t<is_complex<Complex<T>>::value, void> filter_zero(Complex<T>& val) {
        T rr = val.real();
        T ii = val.imag();
        rr = is_zero(rr) ? (T)1 : rr;
        ii = is_zero(ii) ? (T)1 : ii;
        val = Complex<T>(rr, ii);
    }

    template <typename T>
    void filter_int_minimum(T& val) {
        if (!is_integral<T>::value) return;
        if (val == numeric_limits<T>::min()) {
            val = 0;
        }
    }

    template <typename T>
    enable_if_t<is_complex<T>::value, void> filter_add_overflow(T& a, T& b)
    {
        //missing for complex
    }

    template <typename T>
    enable_if_t<is_complex<T>::value, void> filter_sub_overflow(T& a, T& b)
    {
        //missing for complex
    }

    template <typename T>
    enable_if_t < !is_complex<T>::value, void> filter_add_overflow(T& a, T& b) {
        if (is_integral<T>::value == false) return;
        T max = T::max;
        T min = numeric_limits<T>::min();
        // min <= (a +b) <= max;
        // min - b <= a  <= max - b
        if (b < 0) {
            if (a < min - b) {
                a = min - b;
            }
        }
        else {
            if (a > max - b) {
                a = max - b;
            }
        }
    }

    template <typename T>
    enable_if_t < !is_complex<T>::value, void> filter_sub_overflow(T& a, T& b) {
        if (is_integral<T>::value == false) return;
        T max = T::max;
        T min = numeric_limits<T>::min();
        // min <= (a-b) <= max;
        // min + b <= a  <= max +b
        if (b < 0) {
            if (a > max + b) {
                a = max + b;
            }
        }
        else {
            if (a < min + b) {
                a = min + b;
            }
        }
    }

    template <typename T>
    enable_if_t<is_complex<T>::value, void>
    filter_mult_overflow(T& val1, T& val2) {
        //missing
    }

    template <typename T>
    enable_if_t<is_complex<T>::value, void>
    filter_div_ub(T& val1, T& val2) {
        //missing
        //at least consdier zero division
        auto ret = abs(val2);
        if (ret == 0) {
            val2 = T(1, 2);
        }
    }

    template <typename T>
    enable_if_t<!is_complex<T>::value, void>
    filter_mult_overflow(T& val1, T& val2) {
        if (is_integral<T>::value == false) return;
        if (!is_zero(val2)) {
            T c = (T::max - 1) / val2;
            if (abs(val1) >= c) {
                // correct first;
                val1 = c;
            }
        }  // is_zero
    }

    template <typename T>
    enable_if_t<!is_complex<T>::value, void>
    filter_div_ub(T& val1, T& val2) {
        if (is_zero(val2)) {
            val2 = 1;
        }
        else if (is_integral<T>::value && val1 == numeric_limits<T>::min() && val2 == -1) {
            val2 = 1;
        }
    }

    struct TestSeed {
        TestSeed() : seed(chrono::high_resolution_clock::now().time_since_epoch().count()) {
        }
        TestSeed(u64 seed) : seed(seed) {
        }
        u64 getSeed() {
            return seed;
        }
        operator u64 () const {
            return seed;
        }

        TestSeed add(u64 index) {
            return TestSeed(seed + index);
        }

        u64 seed;
    };

    template <typename T, bool is_floating_point = is_floating_point<T>::value, bool is_complex = is_complex<T>::value>
    struct ValueGen
    {
        uniform_int_distribution<i64> dis;
        mt19937 gen;
        ValueGen() : ValueGen(numeric_limits<T>::min(), T::max)
        {
        }
        ValueGen(u64 seed) : ValueGen(numeric_limits<T>::min(), T::max, seed)
        {
        }
        ValueGen(T start, T stop, u64 seed = TestSeed())
        {
            gen = mt19937(seed);
            dis = uniform_int_distribution<i64>(start, stop);
        }
        T get()
        {
            return static_cast<T>(dis(gen));
        }
    };

    template <typename T>
    struct ValueGen<T, true, false>
    {
        mt19937 gen;
        normal_distribution<T> normal;
        uniform_int_distribution<int> roundChance;
        T _start;
        T _stop;
        bool use_sign_change = false;
        bool use_round = true;
        ValueGen() : ValueGen(numeric_limits<T>::min(), T::max)
        {
        }
        ValueGen(u64 seed) : ValueGen(numeric_limits<T>::min(), T::max, seed)
        {
        }
        ValueGen(T start, T stop, u64 seed = TestSeed())
        {
            gen = mt19937(seed);
            T mean = start * static_cast<T>(0.5) + stop * static_cast<T>(0.5);
            //make it  normal +-3sigma
            T divRange = static_cast<T>(6.0);
            T stdev = abs(stop / divRange - start / divRange);
            normal = normal_distribution<T>{ mean, stdev };
            // in real its hard to get rounded value
            // so we will force it by  uniform chance
            roundChance = uniform_int_distribution<int>(0, 5);
            _start = start;
            _stop = stop;
        }
        T get()
        {
            T a = normal(gen);
            //make rounded value ,too
            auto rChoice = roundChance(gen);
            if (rChoice == 1)
                a = round(a);
            if (a < _start)
                return nextafter(_start, _stop);
            if (a >= _stop)
                return nextafter(_stop, _start);
            return a;
        }
    };

    template <typename T>
    struct ValueGen<Complex<T>, false, true>
    {
        mt19937 gen;
        normal_distribution<T> normal;
        uniform_int_distribution<int> roundChance;
        T _start;
        T _stop;
        bool use_sign_change = false;
        bool use_round = true;
        ValueGen() : ValueGen(numeric_limits<T>::min(), T::max)
        {
        }
        ValueGen(u64 seed) : ValueGen(numeric_limits<T>::min(), T::max, seed)
        {
        }
        ValueGen(T start, T stop, u64 seed = TestSeed())
        {
            gen = mt19937(seed);
            T mean = start * static_cast<T>(0.5) + stop * static_cast<T>(0.5);
            //make it  normal +-3sigma
            T divRange = static_cast<T>(6.0);
            T stdev = abs(stop / divRange - start / divRange);
            normal = normal_distribution<T>{ mean, stdev };
            // in real its hard to get rounded value
            // so we will force it by  uniform chance
            roundChance = uniform_int_distribution<int>(0, 5);
            _start = start;
            _stop = stop;
        }
        Complex<T> get()
        {
            T a = normal(gen);
            T b = normal(gen);
            //make rounded value ,too
            auto rChoice = roundChance(gen);
            rChoice = rChoice & 3;
            if (rChoice & 1)
                a = round(a);
            if (rChoice & 2)
                b = round(b);
            if (a < _start)
                a = nextafter(_start, _stop);
            else if (a >= _stop)
                a = nextafter(_stop, _start);
            if (b < _start)
                b = nextafter(_start, _stop);
            else if (b >= _stop)
                b = nextafter(_stop, _start);
            return Complex<T>(a, b);
        }
    };

    template<class T>
    int getTrialCount(int test_trials, int domains_size) {
        int trialCount;
        int trial_default = 1;
        if (sizeof(T) <= 2) {
            //half coverage for byte
            trial_default = 128;
        }
        else {
            //2*65536
            trial_default = 2 * u16::max;
        }
        trialCount = test_trials < 1 ? trial_default : test_trials;
        if (domains_size > 1) {
            trialCount = trialCount / domains_size;
            trialCount = trialCount < 1 ? 1 : trialCount;
        }
        return trialCount;
    }

    template <typename T, typename U = UvalueType<T>>
    class TestCaseBuilder;

    template <typename T, typename U = UvalueType<T>>
    class TestingCase {

        friend class TestCaseBuilder<T, U>;
        static TestCaseBuilder<T, U> getBuilder() { return TestCaseBuilder<T, U>{}; }
        bool checkSpecialValues() const {
            //this will be used to check nan, infs, and other special cases
            return specialCheck;
        }
        usize getTrialCount() const { return trials; }
        bool isBitwise() const { return bitwise; }
        const vector<CheckWithinDomains<U>>& getDomains() const {
            return domains;
        }
        const vector<CustomCheck<T>>& getCustomChecks() const {
            return customCheck;
        }
        TestSeed getTestSeed() const {
            return testSeed;
        }

        // if domains is empty we will test default
        vector<CheckWithinDomains<U>> domains;
        vector<CustomCheck<T>> customCheck;
        // its not used for now
        bool specialCheck = false;
        bool bitwise = false;  // test bitlevel
        usize trials = 0;
        TestSeed testSeed;
    };

    template <typename T, typename U >
    class TestCaseBuilder {

        TestingCase<T, U> _case;

        TestCaseBuilder<T, U>& set(bool bitwise, bool checkSpecialValues) {
            _case.bitwise = bitwise;
            _case.specialCheck = checkSpecialValues;
            return *this;
        }
        TestCaseBuilder<T, U>& setTestSeed(TestSeed seed) {
            _case.testSeed = seed;
            return *this;
        }
        TestCaseBuilder<T, U>& setTrialCount(usize trial_count) {
            _case.trials = trial_count;
            return *this;
        }
        TestCaseBuilder<T, U>& addDomain(const CheckWithinDomains<U>& domainCheck) {
            _case.domains.emplace_back(domainCheck);
            return *this;
        }
        TestCaseBuilder<T, U>& addCustom(const CustomCheck<T>& customArgs) {
            _case.customCheck.emplace_back(customArgs);
            return *this;
        }
        TestCaseBuilder<T, U>& checkSpecialValues() {
            _case.specialCheck = true;
            return *this;
        }
        TestCaseBuilder<T, U>& compareBitwise() {
            _case.bitwise = true;
            return *this;
        }
        operator TestingCase<T, U> && () { return move(_case); }
    };

    template <typename T>
    typename enable_if_t<!is_complex<T>::value&& is_unsigned<T>::value, T>
    correctEpsilon(const T& eps)
    {
        return eps;
    }
    template <typename T>
    typename enable_if_t<!is_complex<T>::value && !is_unsigned<T>::value, T>
    correctEpsilon(const T& eps)
    {
        return abs(eps);
    }
    template <typename T>
    typename enable_if_t<is_complex<Complex<T>>::value, T>
    correctEpsilon(const Complex<T>& eps)
    {
        return abs(eps);
    }

    template <typename T>
    class AssertVectorized
    {

        AssertVectorized(const string& info, TestSeed seed, const T& expected, const T& actual, const T& input0)
            : additionalInfo(info), testSeed(seed), exp(expected), act(actual), arg0(input0), argSize(1)
        {
        }
        AssertVectorized(const string& info, TestSeed seed, const T& expected, const T& actual, const T& input0, const T& input1)
            : additionalInfo(info), testSeed(seed), exp(expected), act(actual), arg0(input0), arg1(input1), argSize(2)
        {
        }
        AssertVectorized(const string& info, TestSeed seed, const T& expected, const T& actual, const T& input0, const T& input1, const T& input2)
            : additionalInfo(info), testSeed(seed), exp(expected), act(actual), arg0(input0), arg1(input1), arg2(input2), argSize(3)
        {
        }
        AssertVectorized(const string& info, TestSeed seed, const T& expected, const T& actual) : additionalInfo(info), testSeed(seed), exp(expected), act(actual)
        {
        }
        AssertVectorized(const string& info, const T& expected, const T& actual) : additionalInfo(info), exp(expected), act(actual), hasSeed(false)
        {
        }

        string getDetail(int index) const
        {
            using UVT = UvalueType<T>;
            stringstream stream;
            stream.precision(numeric_limits<UVT>::max_digits10);
            stream << "Failure Details:\n";
            stream << additionalInfo << "\n";
            if (hasSeed)
            {
                stream << "Test Seed to reproduce: " << testSeed << "\n";
            }
            if (argSize > 0)
            {
                stream << "Arguments:\n";
                stream << "#\t " << arg0 << "\n";
                if (argSize == 2)
                {
                    stream << "#\t " << arg1 << "\n";
                }
                if (argSize == 3)
                {
                    stream << "#\t " << arg2 << "\n";
                }
            }
            stream << "Expected:\n#\t" << exp << "\nActual:\n#\t" << act;
            stream << "\nFirst mismatch Index: " << index;
            return stream.str();
        }

        bool check(bool bitwise = false, bool checkWithTolerance = false, ValueType<T> toleranceEps = {}) const
        {
            using UVT = UvalueType<T>;
            using BVT = BitType<UVT>;
            UVT absErr = correctEpsilon(toleranceEps);
            constexpr int sizeX = VecTypeHelper<T>::holdCount * VecTypeHelper<T>::unitStorageCount;
            constexpr int unitStorageCount = VecTypeHelper<T>::unitStorageCount;
            CACHE_ALIGN UVT expArr[sizeX];
            CACHE_ALIGN UVT actArr[sizeX];
            exp.store(expArr);
            act.store(actArr);
            if (bitwise)
            {
                for (int i = 0; i < sizeX; i++)
                {
                    BVT b_exp = bit_cast<BVT>(expArr[i]);
                    BVT b_act = bit_cast<BVT>(actArr[i]);
                    EXPECT_EQ(b_exp, b_act) << getDetail(i / unitStorageCount);
                    if (::testing::Test::HasFailure())
                        return true;
                }
            }
            else if (checkWithTolerance)
            {
                for (int i = 0; i < sizeX; i++)
                {
                    EXPECT_EQ(nearlyEqual<UVT>(expArr[i], actArr[i], absErr), true) << expArr[i] << "!=" << actArr[i] << "\n" << getDetail(i / unitStorageCount);
                    if (::testing::Test::HasFailure())
                        return true;
                }
            }
            else
            {
                for (int i = 0; i < sizeX; i++)
                {
                    if (is_same<UVT, float>::value)
                    {
                        if (!check_both_nan(expArr[i], actArr[i])) {
                            EXPECT_FLOAT_EQ(expArr[i], actArr[i]) << getDetail(i / unitStorageCount);
                        }
                    }
                    else if (is_same<UVT, double>::value)
                    {
                        if (!check_both_nan(expArr[i], actArr[i]))
                        {
                            EXPECT_DOUBLE_EQ(expArr[i], actArr[i]) << getDetail(i / unitStorageCount);
                        }
                    }
                    else
                    {
                        EXPECT_EQ(expArr[i], actArr[i]) << getDetail(i / unitStorageCount);
                    }
                    if (::testing::Test::HasFailure())
                        return true;
                }
            }
            return false;
        }


        string additionalInfo;
        TestSeed testSeed;
        T exp;
        T act;
        T arg0;
        T arg1;
        T arg2;
        int argSize = 0;
        bool hasSeed = true;
    };

    template< typename T, typename Op1, typename Op2, typename Filter = nullptr_t>
    void test_unary(
        string testNameInfo,
        Op1 expectedFunction,
        Op2 actualFunction, const TestingCase<T>& testCase, Filter filter = {}) {
        using vec_type = T;
        using VT = ValueType<T>;
        using UVT = UvalueType<T>;
        constexpr int el_count = vec_type::size();
        CACHE_ALIGN VT vals[el_count];
        CACHE_ALIGN VT expected[el_count];
        bool bitwise = testCase.isBitwise();
        UVT default_start = is_floating_point<UVT>::value ? numeric_limits<UVT>::lowest() : numeric_limits<UVT>::min();
        UVT default_end = UVT::max;
        auto domains = testCase.getDomains();
        auto domains_size = domains.size();
        auto test_trials = testCase.getTrialCount();
        int trialCount = getTrialCount<UVT>(test_trials, domains_size);
        TestSeed seed = testCase.getTestSeed();
        u64 changeSeedBy = 0;
        for (const CheckWithinDomains<UVT>& dmn : domains) {
            usize dmn_argc = dmn.ArgsDomain.size();
            UVT start = dmn_argc > 0 ? dmn.ArgsDomain[0].start : default_start;
            UVT end = dmn_argc > 0 ? dmn.ArgsDomain[0].end : default_end;
            ValueGen<VT> generator(start, end, seed.add(changeSeedBy));
            for (int trial = 0; trial < trialCount; trial++) {
                for (int k = 0; k < el_count; k++) {
                    vals[k] = generator.get();
                    call_filter(filter, vals[k]);
                    //map operator
                    expected[k] = expectedFunction(vals[k]);
                }
                // test
                auto input = vec_type::loadu(vals);
                auto actual = actualFunction(input);
                auto vec_expected = vec_type::loadu(expected);
                AssertVectorized<vec_type> vecAssert(testNameInfo, seed, vec_expected, actual, input);
                if (vecAssert.check(bitwise, dmn.CheckWithTolerance, dmn.ToleranceError)) return;

            }// trial
            //inrease Seed
            changeSeedBy += 1;
        }
        for (auto& custom : testCase.getCustomChecks()) {
            auto args = custom.Args;
            if (args.size() > 0) {
                auto input = vec_type{ args[0] };
                auto actual = actualFunction(input);
                auto vec_expected = vec_type{ custom.expectedResult };
                AssertVectorized<vec_type> vecAssert(testNameInfo, seed, vec_expected, actual, input);
                if (vecAssert.check()) return;
            }
        }
    }

    template< typename T, typename Op1, typename Op2, typename Filter = nullptr_t>
    void test_binary(
        string testNameInfo,
        Op1 expectedFunction,
        Op2 actualFunction, const TestingCase<T>& testCase, Filter filter = {}) {
        using vec_type = T;
        using VT = ValueType<T>;
        using UVT = UvalueType<T>;
        constexpr int el_count = vec_type::size();
        CACHE_ALIGN VT vals0[el_count];
        CACHE_ALIGN VT vals1[el_count];
        CACHE_ALIGN VT expected[el_count];
        bool bitwise = testCase.isBitwise();
        UVT default_start = is_floating_point<UVT>::value ? numeric_limits<UVT>::lowest() : numeric_limits<UVT>::min();
        UVT default_end = UVT::max;
        auto domains = testCase.getDomains();
        auto domains_size = domains.size();
        auto test_trials = testCase.getTrialCount();
        int trialCount = getTrialCount<UVT>(test_trials, domains_size);
        TestSeed seed = testCase.getTestSeed();
        u64 changeSeedBy = 0;
        for (const CheckWithinDomains<UVT>& dmn : testCase.getDomains()) {
            usize dmn_argc = dmn.ArgsDomain.size();
            UVT start0 = dmn_argc > 0 ? dmn.ArgsDomain[0].start : default_start;
            UVT end0 = dmn_argc > 0 ? dmn.ArgsDomain[0].end : default_end;
            UVT start1 = dmn_argc > 1 ? dmn.ArgsDomain[1].start : default_start;
            UVT end1 = dmn_argc > 1 ? dmn.ArgsDomain[1].end : default_end;
            ValueGen<VT> generator0(start0, end0, seed.add(changeSeedBy));
            ValueGen<VT> generator1(start1, end1, seed.add(changeSeedBy + 1));
            for (int trial = 0; trial < trialCount; trial++) {
                for (int k = 0; k < el_count; k++) {
                    vals0[k] = generator0.get();
                    vals1[k] = generator1.get();
                    call_filter(filter, vals0[k], vals1[k]);
                    //map operator
                    expected[k] = expectedFunction(vals0[k], vals1[k]);
                }
                // test
                auto input0 = vec_type::loadu(vals0);
                auto input1 = vec_type::loadu(vals1);
                auto actual = actualFunction(input0, input1);
                auto vec_expected = vec_type::loadu(expected);
                AssertVectorized<vec_type> vecAssert(testNameInfo, seed, vec_expected, actual, input0, input1);
                if (vecAssert.check(bitwise, dmn.CheckWithTolerance, dmn.ToleranceError))return;
            }// trial
            changeSeedBy += 1;
        }
        for (auto& custom : testCase.getCustomChecks()) {
            auto args = custom.Args;
            if (args.size() > 0) {
                auto input0 = vec_type{ args[0] };
                auto input1 = args.size() > 1 ? vec_type{ args[1] } : vec_type{ args[0] };
                auto actual = actualFunction(input0, input1);
                auto vec_expected = vec_type(custom.expectedResult);
                AssertVectorized<vec_type> vecAssert(testNameInfo, seed, vec_expected, actual, input0, input1);
                if (vecAssert.check()) return;
            }
        }
    }

    template< typename T, typename Op1, typename Op2, typename Filter = nullptr_t>
    void test_ternary(
        string testNameInfo,
        Op1 expectedFunction,
        Op2 actualFunction, const TestingCase<T>& testCase, Filter filter = {}) {
        using vec_type = T;
        using VT = ValueType<T>;
        using UVT = UvalueType<T>;
        constexpr int el_count = vec_type::size();
        CACHE_ALIGN VT vals0[el_count];
        CACHE_ALIGN VT vals1[el_count];
        CACHE_ALIGN VT vals2[el_count];
        CACHE_ALIGN VT expected[el_count];
        bool bitwise = testCase.isBitwise();
        UVT default_start = is_floating_point<UVT>::value ? numeric_limits<UVT>::lowest() : numeric_limits<UVT>::min();
        UVT default_end = UVT::max;
        auto domains = testCase.getDomains();
        auto domains_size = domains.size();
        auto test_trials = testCase.getTrialCount();
        int trialCount = getTrialCount<UVT>(test_trials, domains_size);
        TestSeed seed = testCase.getTestSeed();
        u64 changeSeedBy = 0;
        for (const CheckWithinDomains<UVT>& dmn : testCase.getDomains()) {
            usize dmn_argc = dmn.ArgsDomain.size();
            UVT start0 = dmn_argc > 0 ? dmn.ArgsDomain[0].start : default_start;
            UVT end0 = dmn_argc > 0 ? dmn.ArgsDomain[0].end : default_end;
            UVT start1 = dmn_argc > 1 ? dmn.ArgsDomain[1].start : default_start;
            UVT end1 = dmn_argc > 1 ? dmn.ArgsDomain[1].end : default_end;
            UVT start2 = dmn_argc > 2 ? dmn.ArgsDomain[2].start : default_start;
            UVT end2 = dmn_argc > 2 ? dmn.ArgsDomain[2].end : default_end;
            ValueGen<VT> generator0(start0, end0, seed.add(changeSeedBy));
            ValueGen<VT> generator1(start1, end1, seed.add(changeSeedBy + 1));
            ValueGen<VT> generator2(start2, end2, seed.add(changeSeedBy + 2));

            for (int trial = 0; trial < trialCount; trial++) {
                for (int k = 0; k < el_count; k++) {
                    vals0[k] = generator0.get();
                    vals1[k] = generator1.get();
                    vals2[k] = generator2.get();
                    call_filter(filter, vals0[k], vals1[k], vals2[k]);
                    //map operator
                    expected[k] = expectedFunction(vals0[k], vals1[k], vals2[k]);
                }
                // test
                auto input0 = vec_type::loadu(vals0);
                auto input1 = vec_type::loadu(vals1);
                auto input2 = vec_type::loadu(vals2);
                auto actual = actualFunction(input0, input1, input2);
                auto vec_expected = vec_type::loadu(expected);
                AssertVectorized<vec_type> vecAssert(testNameInfo, seed, vec_expected, actual, input0, input1, input2);
                if (vecAssert.check(bitwise, dmn.CheckWithTolerance, dmn.ToleranceError)) return;
            }// trial
            changeSeedBy += 1;
        }
    }

    template <typename T, typename Op>
    T func_cmp(Op call, T v0, T v1) {
        using bit_rep = BitType<T>;
        constexpr bit_rep mask = bit_rep::max;
        bit_rep  ret = call(v0, v1) ? mask : 0;
        return bit_cast<T>(ret);
    }

    struct PreventFma
    {
        not_inline float sub(float a, float b)
        {
            return a - b;
        }
        not_inline double sub(double a, double b)
        {
            return a - b;
        }
        not_inline float add(float a, float b)
        {
            return a + b;
        }
        not_inline double add(double a, double b)
        {
            return a + b;
        }
    };

    template <typename T>
    enable_if_t<!is_complex<T>::value, T> local_log2(T x) {
        return log2(x);
    }

    template <typename T>
    enable_if_t<is_complex<Complex<T>>::value, Complex<T>> local_log2(Complex<T> x) {
        T ret = log(x);
        T real = ret.real() / log(static_cast<T>(2));
        T imag = ret.imag() / log(static_cast<T>(2));
        return Complex<T>(real, imag);
    }

    template <typename T>
    enable_if_t<!is_complex<T>::value, T> local_abs(T x) {
        return abs(x);
    }

    template <typename T>
    enable_if_t<is_complex<Complex<T>>::value, Complex<T>> local_abs(Complex<T> x) {
    #if defined(TEST_AGAINST_DEFAULT)
        return abs(x);
    #else
        PreventFma noFma;
        T real = x.real();
        T imag = x.imag();
        T rr = real * real;
        T ii = imag * imag;
        T abs = sqrt(noFma.add(rr, ii));
        return Complex<T>(abs, 0);
    #endif
    }

    template <typename T>
    enable_if_t<!is_complex<T>::value, T> local_multiply(T x, T y) {
        return x * y;
    }

    template <typename T>
    enable_if_t<is_complex<Complex<T>>::value, Complex<T>> local_multiply(Complex<T> x, Complex<T> y) {
    #if defined(TEST_AGAINST_DEFAULT)
        return x * y;
    #else
        //(a + bi)  * (c + di) = (ac - bd) + (ad + bc)i
        T x_real = x.real();
        T x_imag = x.imag();
        T y_real = y.real();
        T y_imag = y.imag();
    #if defined(CPU_CAPABILITY_VSX)
        //check multiplication considerin swap and fma
        T rr = x_real * y_real;
        T ii = x_imag * y_real;
        T neg_imag = -y_imag;
        rr = fma(x_imag, neg_imag, rr);
        ii = fma(x_real, y_imag, ii);
    #else
        // replicate order
        PreventFma noFma;
        T ac = x_real * y_real;
        T bd = x_imag * y_imag;
        T ad = x_real * y_imag;
        T bc = x_imag * (-y_real);
        T rr = noFma.sub(ac, bd);
        T ii = noFma.sub(ad, bc);
    #endif
        return Complex<T>(rr, ii);
    #endif
    }

    template <typename T>
    enable_if_t<!is_complex<T>::value, T> local_division(T x, T y) {
        return x / y;
    }

    template <typename T>
    enable_if_t<is_complex<Complex<T>>::value, Complex<T>> local_division(Complex<T> x, Complex<T> y) {
    #if defined(TEST_AGAINST_DEFAULT)
        return x / y;
    #else
        //re = (ac + bd)/abs_2()
        //im = (bc - ad)/abs_2()
        T x_real = x.real();
        T x_imag = x.imag();
        T y_real = y.real();
        T y_imag = y.imag();
        PreventFma noFma;
    #if defined(CPU_CAPABILITY_VSX)
        //check multiplication considerin swap and fma
        T rr = x_real * y_real;
        T ii = x_imag * y_real;
        T neg_imag = -y_imag;
        rr = fma(x_imag, y_imag, rr);
        ii = fma(x_real, neg_imag, ii);
        //b.abs_2
    #else
        T ac = x_real * y_real;
        T bd = x_imag * y_imag;
        T ad = x_real * y_imag;
        T bc = x_imag * y_real;
        T rr = noFma.add(ac, bd);
        T ii = noFma.sub(bc, ad);
    #endif
        //b.abs_2()
        T abs_rr = y_real * y_real;
        T abs_ii = y_imag * y_imag;
        T abs_2 = noFma.add(abs_rr, abs_ii);
        rr = rr / abs_2;
        ii = ii / abs_2;
        return Complex<T>(rr, ii);
    #endif
    }

    template <typename T>
    enable_if_t<!is_complex<T>::value, T> local_fmadd(T a, T b, T c) {
        PreventFma noFma;
        T ab = a * b;
        return noFma.add(ab, c);
    }

    template <typename T>
    enable_if_t<!is_complex<T>::value, T> local_sqrt(T x) {
        return sqrt(x);
    }

    template <typename T>
    enable_if_t<is_complex<Complex<T>>::value, Complex<T>> local_sqrt(Complex<T> x) {
        return sqrt(x);
    }

    template <typename T>
    enable_if_t<!is_complex<T>::value, T> local_asin(T x) {
        return asin(x);
    }

    template <typename T>
    enable_if_t<is_complex<Complex<T>>::value, Complex<T>> local_asin(Complex<T> x) {
        return asin(x);
    }

    template <typename T>
    enable_if_t<!is_complex<T>::value, T> local_acos(T x) {
        return acos(x);
    }

    template <typename T>
    enable_if_t<is_complex<Complex<T>>::value, Complex<T>> local_acos(Complex<T> x) {
        return acos(x);
    }

    template<typename T>
    enable_if_t<!is_complex<T>::value, T>
    local_and(const T& val0, const T& val1) {
        using bit_rep = BitType<T>;
        bit_rep ret = bit_cast<bit_rep>(val0) & bit_cast<bit_rep>(val1);
        return bit_cast<T> (ret);
    }

    template <typename T>
    enable_if_t<is_complex<Complex<T>>::value, Complex<T>>
    local_and(const Complex<T>& val0, const Complex<T>& val1)
    {
        using bit_rep = BitType<T>;
        T real1 = val0.real();
        T imag1 = val0.imag();
        T real2 = val1.real();
        T imag2 = val1.imag();
        bit_rep real_ret = bit_cast<bit_rep>(real1) & bit_cast<bit_rep>(real2);
        bit_rep imag_ret = bit_cast<bit_rep>(imag1) & bit_cast<bit_rep>(imag2);
        return Complex<T>(bit_cast<T>(real_ret), bit_cast<T>(imag_ret));
    }

    template<typename T>
    enable_if_t<!is_complex<T>::value, T>
    local_or(const T& val0, const T& val1) {
        using bit_rep = BitType<T>;
        bit_rep ret = bit_cast<bit_rep>(val0) | bit_cast<bit_rep>(val1);
        return bit_cast<T> (ret);
    }

    template<typename T>
    enable_if_t<is_complex<Complex<T>>::value, Complex<T>>
    local_or(const Complex<T>& val0, const Complex<T>& val1) {
        using bit_rep = BitType<T>;
        T real1 = val0.real();
        T imag1 = val0.imag();
        T real2 = val1.real();
        T imag2 = val1.imag();
        bit_rep real_ret = bit_cast<bit_rep>(real1) | bit_cast<bit_rep>(real2);
        bit_rep imag_ret = bit_cast<bit_rep>(imag1) | bit_cast<bit_rep>(imag2);
        return Complex<T>(bit_cast<T> (real_ret), bit_cast<T>(imag_ret));
    }

    template<typename T>
    enable_if_t<!is_complex<T>::value, T>
    local_xor(const T& val0, const T& val1) {
        using bit_rep = BitType<T>;
        bit_rep ret = bit_cast<bit_rep>(val0) ^ bit_cast<bit_rep>(val1);
        return bit_cast<T> (ret);
    }

    template<typename T>
    enable_if_t<is_complex<Complex<T>>::value, Complex<T>>
    local_xor(const Complex<T>& val0, const Complex<T>& val1) {
        using bit_rep = BitType<T>;
        T real1 = val0.real();
        T imag1 = val0.imag();
        T real2 = val1.real();
        T imag2 = val1.imag();
        bit_rep real_ret = bit_cast<bit_rep>(real1) ^ bit_cast<bit_rep>(real2);
        bit_rep imag_ret = bit_cast<bit_rep>(imag1) ^ bit_cast<bit_rep>(imag2);
        return Complex<T>(bit_cast<T> (real_ret), bit_cast<T>(imag_ret));
    }

    template <typename T>
    T quantize_val(float scale, i64 zero_point, float value) {
        i64 qvalue;
        constexpr i64 qmin = numeric_limits<T>::min();
        constexpr i64 qmax = T::max;
        float inv_scale = 1.0f / scale;
        qvalue = static_cast<i64>(zero_point + native::round_impl<float>(value * inv_scale));
        qvalue = max<i64>(qvalue, qmin);
        qvalue = min<i64>(qvalue, qmax);
        return static_cast<T>(qvalue);
    }

    template <typename T>
    #if defined(TEST_AGAINST_DEFAULT)
    T requantize_from_int(float multiplier, i32 zero_point, i32 src) {
        auto xx = static_cast<float>(src) * multiplier;
        double xx2 = nearbyint(xx);
        i32 quantize_down = xx2 + zero_point;
    #else
    T requantize_from_int(float multiplier, i64 zero_point, i64 src) {
        i64 quantize_down = static_cast<i64>(zero_point + lrintf(src * multiplier));
    #endif
        constexpr i64 min = numeric_limits<T>::min();
        constexpr i64 max = T::max;
        auto ret = static_cast<T>(min<i64>(max<i64>(quantize_down, min), max));
        return ret;
    }

    template <typename T>
    float dequantize_val(float scale, i64 zero_point, T value) {
        //when negated scale is used as addition
    #if defined(CHECK_WITH_FMA)
        float neg_p = -(zero_point * scale);
        float v = static_cast<float>(value);
        float ret = fma(v, scale, neg_p);
    #else
        float ret = (static_cast<float>(value) - zero_point) * scale;
    #endif
        return ret;
    }

    template<typename T>
    T relu(const T & val, const T & zero_point) {
        return max(val, zero_point);
    }

    template<typename T>
    T relu6(T val, T zero_point, T q_six) {
        return min<T>(max<T>(val, zero_point), q_six);
    }

    template<typename T>
    i32 widening_subtract(T val, T b) {
        return static_cast<i32>(val) - static_cast<i32>(b);
    }

    //default testing case
    template<typename T>
    T getDefaultTolerance() {
        return static_cast<T>(0.0);
    }

    template<>
    float getDefaultTolerance() {
        return 5.e-5f;
    }

    template<>
    double getDefaultTolerance() {
        return 1.e-9;
    }

    template<typename T>
    TestingCase<T> createDefaultUnaryTestCase(TestSeed seed = TestSeed(), bool bitwise = false, bool checkWithTolerance = false, usize trials = 0) {
        using UVT = UvalueType<T>;
        TestingCase<T> testCase;
        if (!bitwise && is_floating_point<UVT>::value) {
            //for float types lets add manual ranges
            UVT tolerance = getDefaultTolerance<UVT>();
            testCase = TestingCase<T>::getBuilder()
                .set(bitwise, false)
                .addDomain(CheckWithinDomains<UVT>{ { {(UVT)-10, (UVT)10}}, checkWithTolerance, tolerance})
                .addDomain(CheckWithinDomains<UVT>{ { {(UVT)10, (UVT)100 }}, checkWithTolerance, tolerance})
                .addDomain(CheckWithinDomains<UVT>{ { {(UVT)100, (UVT)1000 }}, checkWithTolerance, tolerance})
                .addDomain(CheckWithinDomains<UVT>{ { {(UVT)-100, (UVT)-10 }}, checkWithTolerance, tolerance})
                .addDomain(CheckWithinDomains<UVT>{ { {(UVT)-1000, (UVT)-100 }}, checkWithTolerance, tolerance})
                .addDomain(CheckWithinDomains<UVT>{ {}, checkWithTolerance, tolerance})
                .setTrialCount(trials)
                .setTestSeed(seed);
        }
        else {
            testCase = TestingCase<T>::getBuilder()
                .set(bitwise, false)
                .addDomain(CheckWithinDomains<UVT>{})
                .setTrialCount(trials)
                .setTestSeed(seed);
        }
        return testCase;
    }

    template<typename T>
    TestingCase<T> createDefaultBinaryTestCase(TestSeed seed = TestSeed(), bool bitwise = false, bool checkWithTolerance = false, usize trials = 0) {
        using UVT = UvalueType<T>;
        TestingCase<T> testCase;
        if (!bitwise && is_floating_point<UVT>::value) {
            //for float types lets add manual ranges
            UVT tolerance = getDefaultTolerance<UVT>();
            testCase = TestingCase<T>::getBuilder()
                .set(bitwise, false)
                .addDomain(CheckWithinDomains<UVT>{ { {(UVT)-10, (UVT)10}, { (UVT)-10, (UVT)10 }}, checkWithTolerance, tolerance})
                .addDomain(CheckWithinDomains<UVT>{ { {(UVT)10, (UVT)100 }, { (UVT)-10, (UVT)100 }}, checkWithTolerance, tolerance})
                .addDomain(CheckWithinDomains<UVT>{ { {(UVT)100, (UVT)1000 }, { (UVT)-100, (UVT)1000 }}, checkWithTolerance, tolerance})
                .addDomain(CheckWithinDomains<UVT>{ { {(UVT)-100, (UVT)-10 }, { (UVT)-100, (UVT)10 }}, checkWithTolerance, tolerance})
                .addDomain(CheckWithinDomains<UVT>{ { {(UVT)-1000, (UVT)-100 }, { (UVT)-1000, (UVT)100 }}, checkWithTolerance, tolerance})
                .addDomain(CheckWithinDomains<UVT>{ {}, checkWithTolerance, tolerance})
                .setTrialCount(trials)
                .setTestSeed(seed);
        }
        else {
            testCase = TestingCase<T>::getBuilder()
                .set(bitwise, false)
                .addDomain(CheckWithinDomains<UVT>{})
                .setTrialCount(trials)
                .setTestSeed(seed);
        }
        return testCase;
    }

    template<typename T>
    TestingCase<T> createDefaultTernaryTestCase(TestSeed seed = TestSeed(), bool bitwise = false, bool checkWithTolerance = false, usize trials = 0) {
        TestingCase<T> testCase = TestingCase<T>::getBuilder()
            .set(bitwise, false)
            .addDomain(CheckWithinDomains<UvalueType<T>>{})
            .setTrialCount(trials)
            .setTestSeed(seed);
        return testCase;
    }
    //-------------------------------------------[.cpp/pytorch/aten/src/ATen/test/vec_test_all_types.cpp]

    namespace {
    #if GTEST_HAS_TYPED_TEST
        template <typename T>
        class Memory : public ::testing::Test {};
        template <typename T>
        class Arithmetics : public ::testing::Test {};
        template <typename T>
        class Comparison : public ::testing::Test {};
        template <typename T>
        class Bitwise : public ::testing::Test {};
        template <typename T>
        class MinMax : public ::testing::Test {};
        template <typename T>
        class Nan : public ::testing::Test {};
        template <typename T>
        class Interleave : public ::testing::Test {};
        template <typename T>
        class SignManipulation : public ::testing::Test {};
        template <typename T>
        class Rounding : public ::testing::Test {};
        template <typename T>
        class SqrtAndReciprocal : public ::testing::Test {};
        template <typename T>
        class SqrtAndReciprocalReal : public ::testing::Test {};
        template <typename T>
        class FractionAndRemainderReal : public ::testing::Test {};
        template <typename T>
        class Trigonometric : public ::testing::Test {};
        template <typename T>
        class ErrorFunctions : public ::testing::Test {};
        template <typename T>
        class Exponents : public ::testing::Test {};
        template <typename T>
        class Hyperbolic : public ::testing::Test {};
        template <typename T>
        class InverseTrigonometric : public ::testing::Test {};
        template <typename T>
        class InverseTrigonometricReal : public ::testing::Test {};
        template <typename T>
        class LGamma : public ::testing::Test {};
        template <typename T>
        class Logarithm : public ::testing::Test {};
        template <typename T>
        class LogarithmReals : public ::testing::Test {};
        template <typename T>
        class Pow : public ::testing::Test {};
        template <typename T>
        class RangeFactories : public ::testing::Test {};
        template <typename T>
        class BitwiseFloatsAdditional : public ::testing::Test {};
        template <typename T>
        class BitwiseFloatsAdditional2 : public ::testing::Test {};
        template <typename T>
        class RealTests : public ::testing::Test {};
        template <typename T>
        class ComplexTests : public ::testing::Test {};
        template <typename T>
        class QuantizationTests : public ::testing::Test {};
        template <typename T>
        class FunctionalTests : public ::testing::Test {};
        using RealFloatTestedTypes = ::testing::Types<vfloat, vdouble>;
        using FloatTestedTypes = ::testing::Types<vfloat, vdouble, vcomplex, vcomplexDbl>;
        using ALLTestedTypes = ::testing::Types<vfloat, vdouble, vcomplex, vlong, vint, vshort, vqint8, vquint8, vqint>;
        using QuantTestedTypes = ::testing::Types<vqint8, vquint8, vqint>;
        using RealFloatIntTestedTypes = ::testing::Types<vfloat, vdouble, vlong, vint, vshort>;
        using FloatIntTestedTypes = ::testing::Types<vfloat, vdouble, vcomplex, vcomplexDbl, vlong, vint, vshort>;
        using ComplexTypes = ::testing::Types<vcomplex, vcomplexDbl>;
        TYPED_TEST_CASE(Memory, ALLTestedTypes);
        TYPED_TEST_CASE(Arithmetics, FloatIntTestedTypes);
        TYPED_TEST_CASE(Comparison, RealFloatIntTestedTypes);
        TYPED_TEST_CASE(Bitwise, FloatIntTestedTypes);
        TYPED_TEST_CASE(MinMax, RealFloatIntTestedTypes);
        TYPED_TEST_CASE(Nan, RealFloatTestedTypes);
        TYPED_TEST_CASE(Interleave, RealFloatIntTestedTypes);
        TYPED_TEST_CASE(SignManipulation, FloatIntTestedTypes);
        TYPED_TEST_CASE(Rounding, RealFloatTestedTypes);
        TYPED_TEST_CASE(SqrtAndReciprocal, FloatTestedTypes);
        TYPED_TEST_CASE(SqrtAndReciprocalReal, RealFloatTestedTypes);
        TYPED_TEST_CASE(FractionAndRemainderReal, RealFloatTestedTypes);
        TYPED_TEST_CASE(Trigonometric, RealFloatTestedTypes);
        TYPED_TEST_CASE(ErrorFunctions, RealFloatTestedTypes);
        TYPED_TEST_CASE(Exponents, RealFloatTestedTypes);
        TYPED_TEST_CASE(Hyperbolic, RealFloatTestedTypes);
        TYPED_TEST_CASE(InverseTrigonometricReal, RealFloatTestedTypes);
        TYPED_TEST_CASE(InverseTrigonometric, FloatTestedTypes);
        TYPED_TEST_CASE(LGamma, RealFloatTestedTypes);
        TYPED_TEST_CASE(Logarithm, FloatTestedTypes);
        TYPED_TEST_CASE(LogarithmReals, RealFloatTestedTypes);
        TYPED_TEST_CASE(Pow, RealFloatTestedTypes);
        TYPED_TEST_CASE(RealTests, RealFloatTestedTypes);
        TYPED_TEST_CASE(RangeFactories, FloatIntTestedTypes);
        TYPED_TEST_CASE(BitwiseFloatsAdditional, RealFloatTestedTypes);
        TYPED_TEST_CASE(BitwiseFloatsAdditional2, FloatTestedTypes);
        TYPED_TEST_CASE(QuantizationTests, QuantTestedTypes);
        TYPED_TEST_CASE(FunctionalTests, RealFloatIntTestedTypes);
        TYPED_TEST(Memory, UnAlignedLoadStore) {
            using vec = TypeParam;
            using VT = ValueType<TypeParam>;
            constexpr usize b_size = vec::size() * sizeof(VT);
            CACHE_ALIGN unsigned char ref_storage[128 * b_size];
            CACHE_ALIGN unsigned char storage[128 * b_size];
            auto seed = TestSeed();
            ValueGen<unsigned char> generator(seed);
            for (auto& x : ref_storage) {
                x = generator.get();
            }
            // test counted load stores
    #if defined(CPU_CAPABILITY_VSX)
            for (int i = 1; i < 2 * vec::size(); i++) {
                vec v = vec::loadu(ref_storage, i);
                v.store(storage);
                usize count = min(i * sizeof(VT), b_size);
                bool cmp = (memcmp(ref_storage, storage, count) == 0);
                ASSERT_TRUE(cmp) << "Failure Details:\nTest Seed to reproduce: " << seed
                    << "\nCount: " << i;
                if (::testing::Test::HasFailure()) {
                    break;
                }
                // clear storage
                memset(storage, 0, b_size);
            }
    #endif
            // testing unaligned load store
            for (usize offset = 0; offset < b_size; offset += 1) {
                unsigned char* p1 = ref_storage + offset;
                unsigned char* p2 = storage + offset;
                for (; p1 + b_size <= end(ref_storage); p1 += b_size, p2 += b_size) {
                    vec v = vec::loadu(p1);
                    v.store(p2);
                }
                usize written = p1 - ref_storage - offset;
                bool cmp = (memcmp(ref_storage + offset, storage + offset, written) == 0);
                ASSERT_TRUE(cmp) << "Failure Details:\nTest Seed to reproduce: " << seed
                    << "\nMismatch at unaligned offset: " << offset;
                if (::testing::Test::HasFailure()) {
                    break;
                }
                // clear storage
                memset(storage, 0, sizeof storage);
            }
        }
        TYPED_TEST(SignManipulation, Absolute) {
            using vec = TypeParam;
            bool checkRelativeErr = is_complex<ValueType<TypeParam>>();
            test_unary<vec>(
                NAME_INFO(absolute), RESOLVE_OVERLOAD(local_abs),
                [](vec v) { return v.abs(); },
                createDefaultUnaryTestCase<vec>(TestSeed(), false, checkRelativeErr),
                RESOLVE_OVERLOAD(filter_int_minimum));
        }
        TYPED_TEST(SignManipulation, Negate) {
            using vec = TypeParam;
            // negate overflows for minimum on int and long
            test_unary<vec>(
                NAME_INFO(negate), negate<ValueType<vec>>(),
                [](vec v) { return v.neg(); },
                createDefaultUnaryTestCase<vec>(TestSeed()),
                RESOLVE_OVERLOAD(filter_int_minimum));
        }
        TYPED_TEST(Rounding, Round) {
            using vec = TypeParam;
            using UVT = UvalueType<TypeParam>;
            UVT case1 = -658.5f;
            UVT exp1 = -658.f;
            UVT case2 = -657.5f;
            UVT exp2 = -658.f;
            auto test_case = TestingCase<vec>::getBuilder()
                .addDomain(CheckWithinDomains<UVT>{ { {-1000, 1000}} })
                .addCustom({ {case1},exp1 })
                .addCustom({ {case2},exp2 })
                .setTrialCount(64000)
                .setTestSeed(TestSeed());
            test_unary<vec>(
                NAME_INFO(round),
                RESOLVE_OVERLOAD(native::round_impl),
                [](vec v) { return v.round(); },
                test_case);
        }
        TYPED_TEST(Rounding, Ceil) {
            using vec = TypeParam;
            test_unary<vec>(
                NAME_INFO(ceil),
                RESOLVE_OVERLOAD(ceil),
                [](vec v) { return v.ceil(); },
                createDefaultUnaryTestCase<vec>(TestSeed()));
        }
        TYPED_TEST(Rounding, Floor) {
            using vec = TypeParam;
            test_unary<vec>(
                NAME_INFO(floor),
                RESOLVE_OVERLOAD(floor),
                [](vec v) { return v.floor(); },
                createDefaultUnaryTestCase<vec>(TestSeed()));
        }
        TYPED_TEST(Rounding, Trunc) {
            using vec = TypeParam;
            test_unary<vec>(
                NAME_INFO(trunc),
                RESOLVE_OVERLOAD(trunc),
                [](vec v) { return v.trunc(); },
                createDefaultUnaryTestCase<vec>(TestSeed()));
        }
        TYPED_TEST(SqrtAndReciprocal, Sqrt) {
            using vec = TypeParam;
            test_unary<vec>(
                NAME_INFO(sqrt),
                RESOLVE_OVERLOAD(local_sqrt),
                [](vec v) { return v.sqrt(); },
                createDefaultUnaryTestCase<vec>(TestSeed(), false, true));
        }
        TYPED_TEST(SqrtAndReciprocalReal, RSqrt) {
            using vec = TypeParam;
            test_unary<vec>(
                NAME_INFO(rsqrt),
                rsqrt<ValueType<vec>>,
                [](vec v) { return v.rsqrt(); },
                createDefaultUnaryTestCase<vec>(TestSeed()),
                RESOLVE_OVERLOAD(filter_zero));
        }
        TYPED_TEST(SqrtAndReciprocalReal, Reciprocal) {
            using vec = TypeParam;
            test_unary<vec>(
                NAME_INFO(reciprocal),
                reciprocal<ValueType<vec>>,
                [](vec v) { return v.reciprocal(); },
                createDefaultUnaryTestCase<vec>(TestSeed()),
                RESOLVE_OVERLOAD(filter_zero));
        }
        TYPED_TEST(FractionAndRemainderReal, Frac) {
          using vec = TypeParam;
          test_unary<vec>(
              NAME_INFO(frac),
              RESOLVE_OVERLOAD(frac),
              [](vec v) { return v.frac(); },
              createDefaultUnaryTestCase<vec>(TestSeed(), false, true));
        }
        TYPED_TEST(FractionAndRemainderReal, Fmod) {
          using vec = TypeParam;
          test_binary<vec>(
              NAME_INFO(fmod),
              RESOLVE_OVERLOAD(fmod),
              [](vec v0, vec v1) { return v0.fmod(v1); },
              createDefaultBinaryTestCase<vec>(TestSeed()),
              RESOLVE_OVERLOAD(filter_fmod));
        }
        TYPED_TEST(Trigonometric, Sin) {
            using vec = TypeParam;
            using UVT = UvalueType<TypeParam>;
            auto test_case = TestingCase<vec>::getBuilder()
                .addDomain(CheckWithinDomains<UVT>{ { {-4096, 4096}}, true, 1.2e-7f})
                .addDomain(CheckWithinDomains<UVT>{ { {-8192, 8192}}, true, 3.0e-7f})
                .setTrialCount(8000)
                .setTestSeed(TestSeed());
            test_unary<vec>(
                NAME_INFO(sin),
                RESOLVE_OVERLOAD(sin),
                [](vec v) { return v.sin(); },
                test_case);
        }
        TYPED_TEST(Trigonometric, Cos) {
            using vec = TypeParam;
            using UVT = UvalueType<TypeParam>;
            auto test_case = TestingCase<vec>::getBuilder()
                .addDomain(CheckWithinDomains<UVT>{ { {-4096, 4096}}, true, 1.2e-7f})
                .addDomain(CheckWithinDomains<UVT>{ { {-8192, 8192}}, true, 3.0e-7f})
                .setTrialCount(8000)
                .setTestSeed(TestSeed());
            test_unary<vec>(
                NAME_INFO(cos),
                RESOLVE_OVERLOAD(cos),
                [](vec v) { return v.cos(); },
                test_case);
        }
        TYPED_TEST(Trigonometric, Tan) {
            using vec = TypeParam;
            test_unary<vec>(
                NAME_INFO(tan),
                RESOLVE_OVERLOAD(tan),
                [](vec v) { return v.tan(); },
                createDefaultUnaryTestCase<vec>(TestSeed()));
        }
        TYPED_TEST(Hyperbolic, Tanh) {
            using vec = TypeParam;
            test_unary<vec>(
                NAME_INFO(tanH),
                RESOLVE_OVERLOAD(tanh),
                [](vec v) { return v.tanh(); },
                createDefaultUnaryTestCase<vec>(TestSeed()));
        }
        TYPED_TEST(Hyperbolic, Sinh) {
            using vec = TypeParam;
            using UVT = UvalueType<TypeParam>;
            auto test_case =
                TestingCase<vec>::getBuilder()
                .addDomain(CheckWithinDomains<UVT>{ { {-88, 88}}, true, getDefaultTolerance<UVT>()})
                .setTrialCount(65536)
                .setTestSeed(TestSeed());
            test_unary<vec>(
                NAME_INFO(sinh),
                RESOLVE_OVERLOAD(sinh),
                [](vec v) { return v.sinh(); },
                test_case);
        }
        TYPED_TEST(Hyperbolic, Cosh) {
            using vec = TypeParam;
            using UVT = UvalueType<TypeParam>;
            auto test_case =
                TestingCase<vec>::getBuilder()
                .addDomain(CheckWithinDomains<UVT>{ { {-88, 88}}, true, getDefaultTolerance<UVT>()})
                .setTrialCount(65536)
                .setTestSeed(TestSeed());
            test_unary<vec>(
                NAME_INFO(cosh),
                RESOLVE_OVERLOAD(cosh),
                [](vec v) { return v.cosh(); },
                test_case);
        }
        TYPED_TEST(InverseTrigonometric, Asin) {
            using vec = TypeParam;
            using UVT = UvalueType<TypeParam>;
            bool checkRelativeErr = is_complex<ValueType<TypeParam>>();
            auto test_case =
                TestingCase<vec>::getBuilder()
                .addDomain(CheckWithinDomains<UVT>{ { {-10, 10}}, checkRelativeErr, getDefaultTolerance<UVT>() })
                .setTrialCount(125536)
                .setTestSeed(TestSeed());
            test_unary<vec>(
                NAME_INFO(asin),
                RESOLVE_OVERLOAD(local_asin),
                [](vec v) { return v.asin(); },
                test_case);
        }
        TYPED_TEST(InverseTrigonometric, ACos) {
            using vec = TypeParam;
            using UVT = UvalueType<TypeParam>;
            bool checkRelativeErr = is_complex<ValueType<TypeParam>>();
            auto test_case =
                TestingCase<vec>::getBuilder()
                .addDomain(CheckWithinDomains<UVT>{ { {-10, 10}}, checkRelativeErr, getDefaultTolerance<UVT>() })
                .setTrialCount(125536)
                .setTestSeed(TestSeed());
            test_unary<vec>(
                NAME_INFO(acos),
                RESOLVE_OVERLOAD(local_acos),
                [](vec v) { return v.acos(); },
                test_case);
        }
        TYPED_TEST(InverseTrigonometric, ATan) {
            bool checkRelativeErr = is_complex<ValueType<TypeParam>>();
            using vec = TypeParam;
            using UVT = UvalueType<TypeParam>;
            auto test_case =
                TestingCase<vec>::getBuilder()
                .addDomain(CheckWithinDomains<UVT>{ { {-100, 100}}, checkRelativeErr, getDefaultTolerance<UVT>()})
                .setTrialCount(65536)
                .setTestSeed(TestSeed());
            test_unary<vec>(
                NAME_INFO(atan),
                RESOLVE_OVERLOAD(atan),
                [](vec v) { return v.atan(); },
                test_case,
                RESOLVE_OVERLOAD(filter_zero));
        }
        TYPED_TEST(Logarithm, Log) {
            using vec = TypeParam;
            test_unary<vec>(
                NAME_INFO(log),
                RESOLVE_OVERLOAD(log),
                [](const vec& v) { return v.log(); },
                createDefaultUnaryTestCase<vec>(TestSeed()));
        }
        TYPED_TEST(LogarithmReals, Log2) {
            using vec = TypeParam;
            test_unary<vec>(
                NAME_INFO(log2),
                RESOLVE_OVERLOAD(local_log2),
                [](const vec& v) { return v.log2(); },
                createDefaultUnaryTestCase<vec>(TestSeed()));
        }
        TYPED_TEST(Logarithm, Log10) {
            using vec = TypeParam;
            test_unary<vec>(
                NAME_INFO(log10),
                RESOLVE_OVERLOAD(log10),
                [](const vec& v) { return v.log10(); },
                createDefaultUnaryTestCase<vec>(TestSeed()));
        }
        TYPED_TEST(LogarithmReals, Log1p) {
            using vec = TypeParam;
            using UVT = UvalueType<TypeParam>;
            auto test_case =
                TestingCase<vec>::getBuilder()
                .addDomain(CheckWithinDomains<UVT>{ { {-1, 1000}}, true, getDefaultTolerance<UVT>()})
                .addDomain(CheckWithinDomains<UVT>{ { {1000, 1.e+30}}, true, getDefaultTolerance<UVT>()})
                .setTrialCount(65536)
                .setTestSeed(TestSeed());
            test_unary<vec>(
                NAME_INFO(log1p),
                RESOLVE_OVERLOAD(log1p),
                [](const vec& v) { return v.log1p(); },
                test_case);
        }
        TYPED_TEST(Exponents, Exp) {
            using vec = TypeParam;
            test_unary<vec>(
                NAME_INFO(exp),
                RESOLVE_OVERLOAD(exp),
                [](const vec& v) { return v.exp(); },
                createDefaultUnaryTestCase<vec>(TestSeed()));
        }
        TYPED_TEST(Exponents, Expm1) {
            using vec = TypeParam;
            test_unary<vec>(
                NAME_INFO(expm1),
                RESOLVE_OVERLOAD(expm1),
                [](const vec& v) { return v.expm1(); },
                createDefaultUnaryTestCase<vec>(TestSeed(), false, true));
        }
        TYPED_TEST(ErrorFunctions, Erf) {
            using vec = TypeParam;
            test_unary<vec>(
                NAME_INFO(erf),
                RESOLVE_OVERLOAD(erf),
                [](const vec& v) { return v.erf(); },
                createDefaultUnaryTestCase<vec>(TestSeed(), false, true));
        }
        TYPED_TEST(ErrorFunctions, Erfc) {
            using vec = TypeParam;
            test_unary<vec>(
                NAME_INFO(erfc),
                RESOLVE_OVERLOAD(erfc),
                [](const vec& v) { return v.erfc(); },
                createDefaultUnaryTestCase<vec>(TestSeed(), false, true));
        }
        TYPED_TEST(ErrorFunctions, Erfinv) {
            using vec = TypeParam;
            test_unary<vec>(
                NAME_INFO(erfinv),
                RESOLVE_OVERLOAD(calc_erfinv),
                [](const vec& v) { return v.erfinv(); },
                createDefaultUnaryTestCase<vec>(TestSeed(), false, true));
        }
        TYPED_TEST(Nan, IsNan) {
            using vec = TypeParam;
            using VT = ValueType<TypeParam>;
            CACHE_ALIGN VT test_vals[vec::size()];
            CACHE_ALIGN VT expected_vals[vec::size()];
            auto vals = 1 << (vec::size());
            for (int val = 0; val < vals; ++val) {
              for (int i = 0; i < vec::size(); ++i) {
                if (val & (1 << i)) {
                  test_vals[i] = numeric_limits<VT>::quiet_NaN();
                  // All bits are set to 1 if true, otherwise 0.
                  // same rule as Vectorized<T>::binary_pred.
                  memset(static_cast<void*>(&expected_vals[i]), 0xFF, sizeof(VT));
                } else {
                  test_vals[i] = (VT)0.123;
                  memset(static_cast<void*>(&expected_vals[i]), 0, sizeof(VT));
                }
              }
              vec actual = vec::loadu(test_vals).isnan();
              vec expected = vec::loadu(expected_vals);
              AssertVectorized<vec>(NAME_INFO(isnan), expected, actual).check();
            }
        }
        TYPED_TEST(LGamma, LGamma) {
            using vec = TypeParam;
            using UVT = UvalueType<vec>;
            UVT tolerance = getDefaultTolerance<UVT>();
            // double: 2e+305  float: 4e+36 (https://sleef.org/purec.xhtml#eg)
            UVT maxCorrect = is_same<UVT, float>::value ? (UVT)4e+36 : (UVT)2e+305;
            TestingCase<vec> testCase = TestingCase<vec>::getBuilder()
                .addDomain(CheckWithinDomains<UVT>{ { {(UVT)-100, (UVT)0}}, true, tolerance})
                .addDomain(CheckWithinDomains<UVT>{ { {(UVT)0, (UVT)1000 }}, true, tolerance})
                .addDomain(CheckWithinDomains<UVT>{ { {(UVT)1000, maxCorrect }}, true, tolerance})
                .setTestSeed(TestSeed());
            test_unary<vec>(
                NAME_INFO(lgamma),
                RESOLVE_OVERLOAD(lgamma),
                [](vec v) { return v.lgamma(); },
                testCase);
        }
        TYPED_TEST(InverseTrigonometricReal, ATan2) {
            using vec = TypeParam;
            test_binary<vec>(
                NAME_INFO(atan2),
                RESOLVE_OVERLOAD(atan2),
                [](vec v0, vec v1) {
                    return v0.atan2(v1);
                },
                createDefaultBinaryTestCase<vec>(TestSeed()));
        }
        TYPED_TEST(Pow, Pow) {
            using vec = TypeParam;
            test_binary<vec>(
                NAME_INFO(pow),
                RESOLVE_OVERLOAD(pow),
                [](vec v0, vec v1) { return v0.pow(v1); },
                createDefaultBinaryTestCase<vec>(TestSeed(), false, true));
        }
        TYPED_TEST(RealTests, Hypot) {
            using vec = TypeParam;
            test_binary<vec>(
                NAME_INFO(hypot),
                RESOLVE_OVERLOAD(hypot),
                [](vec v0, vec v1) { return v0.hypot(v1); },
                createDefaultBinaryTestCase<vec>(TestSeed(), false, true));
        }
        TYPED_TEST(RealTests, NextAfter) {
            using vec = TypeParam;
            test_binary<vec>(
                NAME_INFO(nextafter),
                RESOLVE_OVERLOAD(nextafter),
                [](vec v0, vec v1) { return v0.nextafter(v1); },
                createDefaultBinaryTestCase<vec>(TestSeed(), false, true));
        }
        TYPED_TEST(Interleave, Interleave) {
            using vec = TypeParam;
            using VT = ValueType<TypeParam>;
            constexpr auto N = vec::size() * 2LL;
            CACHE_ALIGN VT vals[N];
            CACHE_ALIGN VT interleaved[N];
            auto seed = TestSeed();
            ValueGen<VT> generator(seed);
            for (VT& v : vals) {
                v = generator.get();
            }
            copy_interleave(vals, interleaved);
            auto a = vec::loadu(vals);
            auto b = vec::loadu(vals + vec::size());
            auto cc = interleave2(a, b);
            AssertVectorized<vec>(NAME_INFO(Interleave FirstHalf), get<0>(cc), vec::loadu(interleaved)).check(true);
            AssertVectorized<vec>(NAME_INFO(Interleave SecondHalf), get<1>(cc), vec::loadu(interleaved + vec::size())).check(true);
        }
        TYPED_TEST(Interleave, DeInterleave) {
            using vec = TypeParam;
            using VT = ValueType<TypeParam>;
            constexpr auto N = vec::size() * 2LL;
            CACHE_ALIGN VT vals[N];
            CACHE_ALIGN VT interleaved[N];
            auto seed = TestSeed();
            ValueGen<VT> generator(seed);
            for (VT& v : vals) {
                v = generator.get();
            }
            copy_interleave(vals, interleaved);
            // test interleaved with vals this time
            auto a = vec::loadu(interleaved);
            auto b = vec::loadu(interleaved + vec::size());
            auto cc = deinterleave2(a, b);
            AssertVectorized<vec>(NAME_INFO(DeInterleave FirstHalf), get<0>(cc), vec::loadu(vals)).check(true);
            AssertVectorized<vec>(NAME_INFO(DeInterleave SecondHalf), get<1>(cc), vec::loadu(vals + vec::size())).check(true);
        }
        TYPED_TEST(Arithmetics, Plus) {
            using vec = TypeParam;
            using VT = ValueType<TypeParam>;
            test_binary<vec>(
                NAME_INFO(plus),
                plus<VT>(),
                [](const vec& v0, const vec& v1) -> vec {
                    return v0 + v1;
                },
                createDefaultBinaryTestCase<vec>(TestSeed()),
                    RESOLVE_OVERLOAD(filter_add_overflow));
        }
        TYPED_TEST(Arithmetics, Minus) {
            using vec = TypeParam;
            using VT = ValueType<TypeParam>;
            test_binary<vec>(
                NAME_INFO(minus),
                minus<VT>(),
                [](const vec& v0, const vec& v1) -> vec {
                    return v0 - v1;
                },
                createDefaultBinaryTestCase<vec>(TestSeed()),
                    RESOLVE_OVERLOAD(filter_sub_overflow));
        }
        TYPED_TEST(Arithmetics, Multiplication) {
            using vec = TypeParam;
            test_binary<vec>(
                NAME_INFO(mult),
                RESOLVE_OVERLOAD(local_multiply),
                [](const vec& v0, const vec& v1) { return v0 * v1; },
                createDefaultBinaryTestCase<vec>(TestSeed(), false, true),
                RESOLVE_OVERLOAD(filter_mult_overflow));
        }
        TYPED_TEST(Arithmetics, Division) {
            using vec = TypeParam;
            TestSeed seed;
            test_binary<vec>(
                NAME_INFO(division),
                RESOLVE_OVERLOAD(local_division),
                [](const vec& v0, const vec& v1) { return v0 / v1; },
                createDefaultBinaryTestCase<vec>(seed),
                RESOLVE_OVERLOAD(filter_div_ub));
        }
        TYPED_TEST(Bitwise, BitAnd) {
            using vec = TypeParam;
            test_binary<vec>(
                NAME_INFO(bit_and),
                RESOLVE_OVERLOAD(local_and),
                [](const vec& v0, const vec& v1) { return v0 & v1; },
                createDefaultBinaryTestCase<vec>(TestSeed(), true));
        }
        TYPED_TEST(Bitwise, BitOr) {
            using vec = TypeParam;
            test_binary<vec>(
                NAME_INFO(bit_or),
                RESOLVE_OVERLOAD(local_or),
                [](const vec& v0, const vec& v1) { return v0 | v1; },
                createDefaultBinaryTestCase<vec>(TestSeed(), true));
        }
        TYPED_TEST(Bitwise, BitXor) {
            using vec = TypeParam;
            test_binary<vec>(
                NAME_INFO(bit_xor),
                RESOLVE_OVERLOAD(local_xor),
                [](const vec& v0, const vec& v1) { return v0 ^ v1; },
                createDefaultBinaryTestCase<vec>(TestSeed(), true));
        }
        TYPED_TEST(Comparison, Equal) {
            using vec = TypeParam;
            using VT = ValueType<TypeParam>;
            test_binary<vec>(
                NAME_INFO(== ),
                [](const VT& v1, const VT& v2) {return func_cmp(equal_to<VT>(), v1, v2); },
                [](const vec& v0, const vec& v1) { return v0 == v1; },
                createDefaultBinaryTestCase<vec>(TestSeed(), true));
        }
        TYPED_TEST(Comparison, NotEqual) {
            using vec = TypeParam;
            using VT = ValueType<TypeParam>;
            test_binary<vec>(
                NAME_INFO(!= ),
                [](const VT& v1, const VT& v2) {return func_cmp(not_equal_to<VT>(), v1, v2); },
                [](const vec& v0, const vec& v1) { return v0 != v1; },
                createDefaultBinaryTestCase<vec>(TestSeed(), true));
        }
        TYPED_TEST(Comparison, Greater) {
            using vec = TypeParam;
            using VT = ValueType<TypeParam>;
            test_binary<vec>(
                NAME_INFO(> ),
                [](const VT& v1, const VT& v2) {return func_cmp(greater<VT>(), v1, v2); },
                [](const vec& v0, const vec& v1) { return v0 > v1; },
                createDefaultBinaryTestCase<vec>(TestSeed(), true));
        }
        TYPED_TEST(Comparison, Less) {
            using vec = TypeParam;
            using VT = ValueType<TypeParam>;
            test_binary<vec>(
                NAME_INFO(< ),
                [](const VT& v1, const VT& v2) {return func_cmp(less<VT>(), v1, v2); },
                [](const vec& v0, const vec& v1) { return v0 < v1; },
                createDefaultBinaryTestCase<vec>(TestSeed(), true));
        }
        TYPED_TEST(Comparison, GreaterEqual) {
            using vec = TypeParam;
            using VT = ValueType<TypeParam>;
            test_binary<vec>(
                NAME_INFO(>= ),
                [](const VT& v1, const VT& v2) {return func_cmp(greater_equal<VT>(), v1, v2); },
                [](const vec& v0, const vec& v1) { return v0 >= v1; },
                createDefaultBinaryTestCase<vec>(TestSeed(), true));
        }
        TYPED_TEST(Comparison, LessEqual) {
            using vec = TypeParam;
            using VT = ValueType<TypeParam>;
            test_binary<vec>(
                NAME_INFO(<= ),
                [](const VT& v1, const VT& v2) {return func_cmp(less_equal<VT>(), v1, v2); },
                [](const vec& v0, const vec& v1) { return v0 <= v1; },
                createDefaultBinaryTestCase<vec>(TestSeed(), true));
        }
        TYPED_TEST(MinMax, Minimum) {
            using vec = TypeParam;
            using VT = ValueType<TypeParam>;
            test_binary<vec>(
                NAME_INFO(minimum),
                minimum<VT>,
                [](const vec& v0, const vec& v1) {
                    return minimum(v0, v1);
                },
                createDefaultBinaryTestCase<vec>(TestSeed()));
        }
        TYPED_TEST(MinMax, Maximum) {
            using vec = TypeParam;
            using VT = ValueType<TypeParam>;
            test_binary<vec>(
                NAME_INFO(maximum),
                maximum<VT>,
                [](const vec& v0, const vec& v1) {
                    return maximum(v0, v1);
                },
                createDefaultBinaryTestCase<vec>(TestSeed()));
        }
        TYPED_TEST(MinMax, ClampMin) {
            using vec = TypeParam;
            using VT = ValueType<TypeParam>;
            test_binary<vec>(
                NAME_INFO(clamp min),
                clamp_min<VT>,
                [](const vec& v0, const vec& v1) {
                    return clamp_min(v0, v1);
                },
                createDefaultBinaryTestCase<vec>(TestSeed()));
        }
        TYPED_TEST(MinMax, ClampMax) {
            using vec = TypeParam;
            using VT = ValueType<TypeParam>;
            test_binary<vec>(
                NAME_INFO(clamp max),
                clamp_max<VT>,
                [](const vec& v0, const vec& v1) {
                    return clamp_max(v0, v1);
                },
                createDefaultBinaryTestCase<vec>(TestSeed()));
        }
        TYPED_TEST(MinMax, Clamp) {
            using vec = TypeParam;
            using VT = ValueType<TypeParam>;
            test_ternary<vec>(
                NAME_INFO(clamp), clamp<VT>,
                [](const vec& v0, const vec& v1, const vec& v2) {
                    return clamp(v0, v1, v2);
                },
                createDefaultTernaryTestCase<vec>(TestSeed()),
                    RESOLVE_OVERLOAD(filter_clamp));
        }
        TYPED_TEST(BitwiseFloatsAdditional, ZeroMask) {
            using vec = TypeParam;
            using VT = ValueType<TypeParam>;
            CACHE_ALIGN VT test_vals[vec::size()];
            //all sets will be within 0  2^(n-1)
            auto power_sets = 1 << (vec::size());
            for (int expected = 0; expected < power_sets; expected++) {
                // generate test_val based on expected
                for (int i = 0; i < vec::size(); ++i)
                {
                    if (expected & (1 << i)) {
                        test_vals[i] = (VT)0;
                    }
                    else {
                        test_vals[i] = (VT)0.897;
                    }
                }
                int actual = vec::loadu(test_vals).zero_mask();
                ASSERT_EQ(expected, actual) << "Failure Details:\n"
                    << hex << "Expected:\n#\t" << expected
                    << "\nActual:\n#\t" << actual;
            }
        }
        TYPED_TEST(BitwiseFloatsAdditional, Convert) {
            using vec = TypeParam;
            using VT = ValueType<TypeParam>;
            using IntVT = vec::int_same_Size<VT>;

            // verify float to int
            CACHE_ALIGN VT input1[vec::size()];
            CACHE_ALIGN IntVT expected_vals1[vec::size()];
            CACHE_ALIGN IntVT actual_vals1[vec::size()];
            for (i64 i = 0; i < vec::size(); i++) {
                input1[i] = (VT)i * (VT)2.1 + (VT)0.5;
                expected_vals1[i] = static_cast<IntVT>(input1[i]);
            }
            vec::convert(input1, actual_vals1, vec::size());
            auto expected1 = VecType<IntVT>::loadu(expected_vals1);
            auto actual1 = VecType<IntVT>::loadu(actual_vals1);
            if (AssertVectorized<VecType<IntVT>>(NAME_INFO(test_convert_to_int), expected1, actual1).check()) {
              return;
            }

            // verify int to float
            CACHE_ALIGN IntVT input2[vec::size()];
            CACHE_ALIGN VT expected_vals2[vec::size()];
            CACHE_ALIGN VT actual_vals2[vec::size()];
            for (i64 i = 0; i < vec::size(); i++) {
                input2[i] = (IntVT)i * (IntVT)2 + (IntVT)1;
                expected_vals2[i] = (VT)input2[i];
            }
            vec::convert(input2, actual_vals2, vec::size());
            auto expected2 = vec::loadu(expected_vals2);
            auto actual2 = vec::loadu(actual_vals2);
            AssertVectorized<vec>(NAME_INFO(test_convert_to_float), expected2, actual2).check();
        }
        TYPED_TEST(BitwiseFloatsAdditional, Fmadd) {
            using vec = TypeParam;
            using VT = ValueType<TypeParam>;

            auto test_case = TestingCase<vec>::getBuilder()
              .addDomain(CheckWithinDomains<VT>{
                  {{(VT)-1000, (VT)1000}, {(VT)-1000, (VT)1000}, {(VT)-1000, (VT)1000}},
                  true, getDefaultTolerance<VT>()})
              .setTestSeed(TestSeed());

            test_ternary<vec>(
                NAME_INFO(clamp), RESOLVE_OVERLOAD(local_fmadd),
                [](const vec& v0, const vec& v1, const vec& v2) {
                    return vec::fmadd(v0, v1, v2);
                },
                test_case,
                RESOLVE_OVERLOAD(filter_fmadd));
        }
        template<typename vec, typename VT, i64 mask>
        typename enable_if_t<(mask < 0 || mask> 255), void>
        test_blend(VT expected_val[vec::size()], VT a[vec::size()], VT b[vec::size()])
        {
        }
        template<typename vec, typename VT, i64 mask>
        typename enable_if_t<(mask >= 0 && mask <= 255), void>
        test_blend(VT expected_val[vec::size()], VT a[vec::size()], VT b[vec::size()]) {
            // generate expected_val
            i64 m = mask;
            for (i64 i = 0; i < vec::size(); i++) {
                expected_val[i] = (m & 0x01) ? b[i] : a[i];
                m = m >> 1;
            }
            // test with blend
            auto vec_a = vec::loadu(a);
            auto vec_b = vec::loadu(b);
            auto expected = vec::loadu(expected_val);
            auto actual = vec::template blend<mask>(vec_a, vec_b);
            auto mask_str = string("\nblend mask: ") + to_string(mask);
            if (AssertVectorized<vec>(string(NAME_INFO(test_blend)) + mask_str, expected, actual).check()) return;
            test_blend<vec, VT, mask - 1>(expected_val, a, b);
        }
        template<typename vec, typename VT, i64 idx, i64 N>
        enable_if_t<(!is_complex<VT>::value && idx == N), bool>
        test_blendv(VT expected_val[vec::size()], VT a[vec::size()], VT b[vec::size()], VT mask[vec::size()]) {
            // generate expected_val
            for (i64 i = 0; i < vec::size(); i++) {
                i64 hex_mask = 0;
                memcpy(&hex_mask, &mask[i], sizeof(VT));
                expected_val[i] = (hex_mask & 0x01) ? b[i] : a[i];
            }
            // test with blendv
            auto vec_a = vec::loadu(a);
            auto vec_b = vec::loadu(b);
            auto vec_m = vec::loadu(mask);
            auto expected = vec::loadu(expected_val);
            auto actual = vec::blendv(vec_a, vec_b, vec_m);
            auto mask_str = string("\nblendv mask: ");
            for (i64 i = 0; i < vec::size(); i++) {
                mask_str += to_string(mask[i]) + " ";
            }
            if (AssertVectorized<vec>(string(NAME_INFO(test_blendv)) + mask_str, expected, actual).check()) {
                return false;
            }
            return true;
        }
        template<typename vec, typename VT, i64 idx, i64 N>
        enable_if_t<(!is_complex<VT>::value && idx != N), bool>
        test_blendv(VT expected_val[vec::size()], VT a[vec::size()], VT b[vec::size()], VT mask[vec::size()]) {
            // shuffle mask and do blendv test
            VT m = mask[idx];
            if (!test_blendv<vec, VT, idx+1, N>(expected_val, a, b, mask)) return false;
            if (m != (VT)0) {
              mask[idx] = (VT)0;
            }
            else {
              i64 hex_mask = 0xFFFFFFFFFFFFFFFF;
              memcpy(&mask[idx], &hex_mask, sizeof(VT));
            }
            if (!test_blendv<vec, VT, idx+1, N>(expected_val, a, b, mask)) return false;
            mask[idx] = m;
            return true;
        }
        template<typename T, int N>
        void blend_init(T(&a)[N], T(&b)[N]) {
            a[0] = (T)1.0;
            b[0] = a[0] + (T)N;
            for (int i = 1; i < N; i++) {
                a[i] = a[i - 1] + (T)(1.0);
                b[i] = b[i - 1] + (T)(1.0);
            }
        }
        template<>
        void blend_init<Complex<float>, 4>(Complex<float>(&a)[4], Complex<float>(&b)[4]) {
            auto add = Complex<float>(1., 100.);
            a[0] = Complex<float>(1., 100.);
            b[0] = Complex<float>(5., 1000.);
            for (int i = 1; i < 4; i++) {
                a[i] = a[i - 1] + add;
                b[i] = b[i - 1] + add;
            }
        }
        template<>
        void blend_init<Complex<double>, 2>(Complex<double>(&a)[2], Complex<double>(&b)[2]) {
            auto add = Complex<double>(1.0, 100.0);
            a[0] = Complex<double>(1.0, 100.0);
            b[0] = Complex<double>(3.0, 1000.0);
            a[1] = a[0] + add;
            b[1] = b[0] + add;
        }
        TYPED_TEST(BitwiseFloatsAdditional, Blendv) {
            using vec = TypeParam;
            using VT = ValueType<TypeParam>;
            CACHE_ALIGN VT a[vec::size()];
            CACHE_ALIGN VT b[vec::size()];
            CACHE_ALIGN VT mask[vec::size()] = {0};
            CACHE_ALIGN VT expected_val[vec::size()];
            blend_init(a, b);
            test_blendv<vec, VT, 0, vec::size()>(expected_val, a, b, mask);
        }
        TYPED_TEST(BitwiseFloatsAdditional2, Blend) {
            using vec = TypeParam;
            using VT = ValueType<TypeParam>;
            CACHE_ALIGN VT a[vec::size()];
            CACHE_ALIGN VT b[vec::size()];
            CACHE_ALIGN VT expected_val[vec::size()];
            blend_init(a, b);
            constexpr i64 power_sets = 1LL << (vec::size());
            test_blend<vec, VT, power_sets - 1>(expected_val, a, b);
        }
        template<typename vec, typename VT>
        void test_set(VT expected_val[vec::size()], VT a[vec::size()], VT b[vec::size()], i64 count){
            if (count < 0) return;
            //generate expected_val
            for (i64 i = 0; i < vec::size(); i++) {
                expected_val[i] = (i < count) ? b[i] : a[i];
            }
            // test with set
            auto vec_a = vec::loadu(a);
            auto vec_b = vec::loadu(b);
            auto expected = vec::loadu(expected_val);
            auto actual = vec::set(vec_a, vec_b, count);

            auto count_str = string("\ncount: ") + to_string(count);
            if (AssertVectorized<vec>(string(NAME_INFO(test_set)) + count_str, expected, actual).check()) {
              return;
            }
            test_set<vec, VT>(expected_val, a, b, (count == 0 ? -1 : count / 2));
        }
        TYPED_TEST(BitwiseFloatsAdditional2, Set) {
            using vec = TypeParam;
            using VT = ValueType<TypeParam>;
            CACHE_ALIGN VT a[vec::size()];
            CACHE_ALIGN VT b[vec::size()];
            CACHE_ALIGN VT expected_val[vec::size()];
            blend_init(a, b);
            test_set<vec, VT>(expected_val, a, b, vec::size());
        }
        template<typename T>
        enable_if_t<!is_complex<T>::value, void>
        arange_init(T& base, T& step) {
            base = (T)5.0;
            step = (T)2.0;
        }
        template<typename T>
        enable_if_t<is_complex<T>::value, void>
        arange_init(T& base, T& step) {
           base = T(5.0, 5.0);
           step = T(2.0, 3.0);
        }
        TYPED_TEST(RangeFactories, Arange) {
            using vec = TypeParam;
            using VT = ValueType<TypeParam>;
            using UVT = UvalueType<TypeParam>;
            CACHE_ALIGN VT expected_val[vec::size()];
            VT base, step;
            arange_init(base, step);
            for (i64 i = 0; i < vec::size(); i++) {
                expected_val[i] = base + VT((UVT)i) * step;
            }
            auto expected = vec::loadu(expected_val);
            auto actual = vec::arange(base, step);
            AssertVectorized<vec>(NAME_INFO(test_arange), expected, actual).check();
        }
        TEST(ComplexTests, TestComplexFloatImagRealConj) {
            float aa[] = { 1.5488e-28,2.5488e-28,3.5488e-28,4.5488e-28,5.5488e-28,6.5488e-28,7.5488e-28,8.5488e-28 };
            float exp[] = { aa[0],0,aa[2],0,aa[4],0,aa[6],0 };
            float exp3[] = { aa[1],0,aa[3],0,aa[5],0,aa[7],0 };
            float exp4[] = { 1.5488e-28, -2.5488e-28,3.5488e-28,-4.5488e-28,5.5488e-28,-6.5488e-28,7.5488e-28,-8.5488e-28 };
            auto a = vcomplex::loadu(aa);
            auto actual1 = a.real();
            auto actual3 = a.imag();
            auto actual4 = a.conj();
            auto expected1 = vcomplex::loadu(exp);
            auto expected3 = vcomplex::loadu(exp3);
            auto expected4 = vcomplex::loadu(exp4);
            AssertVectorized<vcomplex>(NAME_INFO(complex real), expected1, actual1).check();
            AssertVectorized<vcomplex>(NAME_INFO(complex imag), expected3, actual3).check();
            AssertVectorized<vcomplex>(NAME_INFO(complex conj), expected4, actual4).check();
        }
        TYPED_TEST(QuantizationTests, Quantize) {
            using vec = TypeParam;
            using underlying = ValueType<vec>;
            constexpr int trials = 4000;
            constexpr int min_val = numeric_limits<underlying>::min();
            constexpr int max_val = underlying::max;
            constexpr int el_count = vfloat::size();
            CACHE_ALIGN float unit_float_vec[el_count];
            CACHE_ALIGN underlying expected_qint_vals[vec::size()];
            typename vec::float_vec_return_type  float_ret;
            auto seed = TestSeed();
            //zero point
            ValueGen<int> generator_zp(min_val, max_val, seed);
            //scale
            ValueGen<float> generator_sc(1.f, 15.f, seed.add(1));
            //value
            float minv = static_cast<float>(static_cast<double>(min_val) * 2.0);
            float maxv = static_cast<float>(static_cast<double>(max_val) * 2.0);
            ValueGen<float> gen(minv, maxv, seed.add(2));
            for (int i = 0; i < trials; i++) {
                float scale = generator_sc.get();
                float inv_scale = 1.0f / static_cast<float>(scale);
                auto zero_point_val = generator_zp.get();
                int index = 0;
                for (int j = 0; j < vec::float_num_vecs(); j++) {
                    //generate vals
                    for (auto& v : unit_float_vec) {
                        v = gen.get();
                        expected_qint_vals[index] = quantize_val<underlying>(scale, zero_point_val, v);
                        index++;
                    }
                    float_ret[j] = vfloat::loadu(unit_float_vec);
                }
                auto expected = vec::loadu(expected_qint_vals);
                auto actual = vec::quantize(float_ret, scale, zero_point_val, inv_scale);
                if (AssertVectorized<vec>(NAME_INFO(Quantize), expected, actual).check()) return;
            } //trials;
        }
        TYPED_TEST(QuantizationTests, DeQuantize) {
            using vec = TypeParam;
            using underlying = ValueType<vec>;
            constexpr bool is_large = sizeof(underlying) > 1;
            constexpr int trials = is_large ? 4000 : underlying::max / 2;
            constexpr int min_val = is_large ? -2190 : numeric_limits<underlying>::min();
            constexpr int max_val = is_large ? 2199 : underlying::max;
            CACHE_ALIGN float unit_exp_vals[vfloat::size()];
            CACHE_ALIGN underlying qint_vals[vec::size()];
    #if  defined(CHECK_DEQUANT_WITH_LOW_PRECISION)
            cout << "Dequant will be tested with relative error " << 1.e-3f << endl;
    #endif
            auto seed = TestSeed();
            ValueGen<int> generator(min_val, max_val, seed.add(1));
            //scale
            ValueGen<float> generator_sc(1.f, 15.f, seed.add(2));
            for (int i = 0; i < trials; i++) {
                float scale = generator_sc.get();
                i32 zero_point_val = generator.get();
                float scale_zp_premul = -(scale * zero_point_val);
                vfloat vf_scale = vfloat{ scale };
                vfloat vf_zp = vfloat{ static_cast<float>(zero_point_val) };
                vfloat vf_scale_zp = vfloat{ scale_zp_premul };
                //generate vals
                for (auto& x : qint_vals) {
                    x = generator.get();
                }
                //get expected
                int index = 0;
                auto qint_vec = vec::loadu(qint_vals);
                auto actual_float_ret = qint_vec.dequantize(vf_scale, vf_zp, vf_scale_zp);
                for (int j = 0; j < vec::float_num_vecs(); j++) {
                    for (auto& v : unit_exp_vals) {
                        v = dequantize_val(scale, zero_point_val, qint_vals[index]);
                        index++;
                    }
                    vfloat expected = vfloat::loadu(unit_exp_vals);
                    const auto& actual = actual_float_ret[j];
    #if  defined(CHECK_DEQUANT_WITH_LOW_PRECISION)
                    if (AssertVectorized<vfloat>(NAME_INFO(DeQuantize), seed, expected, actual).check(false, true, 1.e-3f)) return;
    #else
                    if (AssertVectorized<vfloat>(NAME_INFO(DeQuantize), seed, expected, actual).check()) return;
    #endif
                }
            } //trials;
        }
        TYPED_TEST(QuantizationTests, ReQuantizeFromInt) {
            using vec = TypeParam;
            using underlying = ValueType<vec>;
            constexpr int trials = 4000;
            constexpr int min_val = -65535;
            constexpr int max_val = 65535;
            constexpr int el_count = vint::size();
            CACHE_ALIGN qint32 unit_int_vec[el_count];
            CACHE_ALIGN underlying expected_qint_vals[vec::size()];
            typename vec::int_vec_return_type  int_ret;
            auto seed = TestSeed();
            //zero point and value
            ValueGen<i32> generator(min_val, max_val, seed);
            //scale
            ValueGen<float> generator_sc(1.f, 15.f, seed.add(1));
            for (int i = 0; i < trials; i++) {
                float multiplier = 1.f / (generator_sc.get());
                auto zero_point_val = generator.get();
                int index = 0;
                for (int j = 0; j < vec::float_num_vecs(); j++) {
                    //generate vals
                    for (auto& v : unit_int_vec) {
                        v = qint32(generator.get());
                        expected_qint_vals[index] = requantize_from_int<underlying>(multiplier, zero_point_val, v.val_);
                        index++;
                    }
                    int_ret[j] = vqint::loadu(unit_int_vec);
                }
                auto expected = vec::loadu(expected_qint_vals);
                auto actual = vec::requantize_from_int(int_ret, multiplier, zero_point_val);
                if (AssertVectorized<vec>(NAME_INFO(ReQuantizeFromInt), seed, expected, actual).check()) {
                    return;
                }
            } //trials;
        }
        TYPED_TEST(QuantizationTests, WideningSubtract) {
            using vec = TypeParam;
            using underlying = ValueType<vec>;
            constexpr bool is_large = sizeof(underlying) > 1;
            constexpr int trials = is_large ? 4000 : underlying::max / 2;
            constexpr int min_val = numeric_limits<underlying>::min();
            constexpr int max_val = underlying::max;
            CACHE_ALIGN i32 unit_exp_vals[vfloat::size()];
            CACHE_ALIGN underlying qint_vals[vec::size()];
            CACHE_ALIGN underlying qint_b[vec::size()];
            typename vec::int_vec_return_type  expected_int_ret;
            auto seed = TestSeed();
            ValueGen<underlying> generator(min_val, max_val, seed);
            for (int i = 0; i < trials; i++) {
                //generate vals
                for (int j = 0; j < vec::size(); j++) {
                    qint_vals[j] = generator.get();
                    qint_b[j] = generator.get();
                    if (is_same<underlying, int>::value) {
                        //filter overflow cases
                        filter_sub_overflow(qint_vals[j], qint_b[j]);
                    }
                }
                int index = 0;
                auto qint_vec = vec::loadu(qint_vals);
                auto qint_vec_b = vec::loadu(qint_b);
                auto actual_int_ret = qint_vec.widening_subtract(qint_vec_b);
                for (int j = 0; j < vec::float_num_vecs(); j++) {
                    for (auto& v : unit_exp_vals) {
                        v = widening_subtract(qint_vals[index], qint_b[index]);
                        index++;
                    }
                    auto expected = vqint::loadu(unit_exp_vals);
                    const auto& actual = actual_int_ret[j];
                    if (AssertVectorized<vqint>(NAME_INFO(WideningSubtract), seed, expected, actual).check()) return;
                }
            } //trials;
        }
        TYPED_TEST(QuantizationTests, Relu) {
            using vec = TypeParam;
            using VT = ValueType<TypeParam>;
            constexpr VT min_val = numeric_limits<VT>::min();
            constexpr VT max_val = VT::max;
            constexpr VT fake_zp = sizeof(VT) > 1 ? static_cast<VT>(65535) : static_cast<VT>(47);
            auto test_case = TestingCase<vec>::getBuilder()
                .addDomain(CheckWithinDomains<VT>{ { DomainRange<VT>{min_val, max_val}, DomainRange<VT>{(VT)0, (VT)fake_zp}} })
                .setTestSeed(TestSeed());
            test_binary<vec>(
                NAME_INFO(relu),
                RESOLVE_OVERLOAD(relu),
                [](const vec& v0, const vec& v1) {
                    return v0.relu(v1);
                },
                test_case);
        }
        TYPED_TEST(QuantizationTests, Relu6) {
            using vec = TypeParam;
            using VT = ValueType<TypeParam>;
            constexpr VT min_val = numeric_limits<VT>::min();
            constexpr VT max_val = VT::max;
            constexpr VT fake_zp = sizeof(VT) > 1 ? static_cast<VT>(65535) : static_cast<VT>(47);
            constexpr VT temp = sizeof(VT) > 1 ? static_cast<VT>(12345) : static_cast<VT>(32);
            constexpr VT fake_qsix = fake_zp + temp;
            auto test_case = TestingCase<vec>::getBuilder()
                .addDomain(CheckWithinDomains<VT>{
                    {
                        DomainRange<VT>{min_val, max_val},
                            DomainRange<VT>{(VT)0, (VT)fake_zp},
                            DomainRange<VT>{(VT)fake_zp, (VT)fake_qsix}
                    }})
                .setTestSeed(TestSeed());
            test_ternary<vec>(
                NAME_INFO(relu6),
                RESOLVE_OVERLOAD(relu6),
                [](/*const*/ vec& v0, const vec& v1, const vec& v2) {
                    return  v0.relu6(v1, v2);
                },
                test_case);
        }
        TYPED_TEST(FunctionalTests, Map) {
            using vec = TypeParam;
            using VT = ValueType<TypeParam>;
            constexpr auto R = 2LL; // residual
            constexpr auto N = vec::size() + R;
            CACHE_ALIGN VT x1[N];
            CACHE_ALIGN VT x2[N];
            CACHE_ALIGN VT x3[N];
            CACHE_ALIGN VT x4[N];
            CACHE_ALIGN VT y[N];
            CACHE_ALIGN VT ref_y[N];
            auto seed = TestSeed();
            ValueGen<VT> generator(VT(-100), VT(100), seed);
            for (i64 i = 0; i < N; i++) {
              x1[i] = generator.get();
              x2[i] = generator.get();
              x3[i] = generator.get();
              x4[i] = generator.get();
            }
            auto cmp = [&](VT* y, VT* ref_y) {
              AssertVectorized<vec>(NAME_INFO(Map), vec::loadu(y), vec::loadu(ref_y)).check(true);
              AssertVectorized<vec>(NAME_INFO(Map), vec::loadu(y + vec::size(), R), vec::loadu(ref_y + vec::size(), R)).check(true);
            };
            // test map: y = x1
            vec::map<VT>([](vec x) { return x; }, y, x1, N);
            for (i64 i = 0; i < N; i++) { ref_y[i] = x1[i]; }
            cmp(y, ref_y);
            // test map2: y = x1 + x2
            vec::map2<VT>([](vec x1, vec x2) { return x1 + x2; }, y, x1, x2, N);
            for (i64 i = 0; i < N; i++) { ref_y[i] = x1[i] + x2[i]; }
            cmp(y, ref_y);
            // test map3: y = x1 + x2 + x3
            vec::map3<VT>([](vec x1, vec x2, vec x3) { return x1 + x2 + x3; }, y, x1, x2, x3, N);
            for (i64 i = 0; i < N; i++) { ref_y[i] = x1[i] + x2[i] + x3[i]; }
            cmp(y, ref_y);
            // test map3: y = x1 + x2 + x3 + x4
            vec::map4<VT>([](vec x1, vec x2, vec x3, vec x4) { return x1 + x2 + x3 + x4; }, y, x1, x2, x3, x4, N);
            for (i64 i = 0; i < N; i++) { ref_y[i] = x1[i] + x2[i] + x3[i] + x4[i]; }
            cmp(y, ref_y);
        }

    #else
    #error GTEST does not have TYPED_TEST
    #endif
    }  // namespace
    */
}

