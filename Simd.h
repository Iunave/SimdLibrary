/*
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#pragma once

#include <concepts>
#include "Memory.h"

#define ATTRAVX inline __attribute__((always_inline, nodebug, flatten))

#ifdef __AVX__
#define AVX128 __AVX__
#endif
#ifdef __AVX2__
#define AVX256 __AVX2__
#endif
#ifdef __AVX512__
#define AVX512 __AVX512__ //not yet implemented
#endif

namespace Simd
{

    namespace Internal
    {
        template<typename VectorType, typename ElementType>
        class TVectorRegister;

#ifdef AVX128

        template<typename T>
        using Vector16 = T __attribute__((__vector_size__(16), __aligned__(16)));

        using char8_16 = Vector16<char8>;
        using int8_16 = Vector16<int8>;
        using uint8_16 = Vector16<uint8>;

        using int16_8 = Vector16<int16>;
        using uint16_8 = Vector16<uint16>;

        using int32_4 = Vector16<int32>;
        using uint32_4 = Vector16<uint32>;

        using int64_2 = Vector16<int64>;
        using uint64_2 = Vector16<uint64>;

        using float32_4 = Vector16<float32>;
        using float64_2 = Vector16<float64>;

        static_assert(alignof(char8_16) == 16);
        static_assert(alignof(int8_16) == 16);
        static_assert(alignof(uint8_16) == 16);

        static_assert(alignof(int16_8) == 16);
        static_assert(alignof(uint16_8) == 16);

        static_assert(alignof(int32_4) == 16);
        static_assert(alignof(uint32_4) == 16);

        static_assert(alignof(int64_2) == 16);
        static_assert(alignof(uint64_2) == 16);

        static_assert(alignof(float32_4) == 16);
        static_assert(alignof(float64_2) == 16);
    }

    using char8_16 = Internal::TVectorRegister<Internal::char8_16, char8>;
    using int8_16 = Internal::TVectorRegister<Internal::int8_16, int8>;
    using uint8_16 = Internal::TVectorRegister<Internal::uint8_16, uint8>;

    using int16_8 = Internal::TVectorRegister<Internal::int16_8, int16>;
    using uint16_8 = Internal::TVectorRegister<Internal::uint16_8, uint16>;

    using int32_4 = Internal::TVectorRegister<Internal::int32_4, int32>;
    using uint32_4 = Internal::TVectorRegister<Internal::uint32_4, uint32>;

    using int64_2 = Internal::TVectorRegister<Internal::int64_2, int64>;
    using uint64_2 = Internal::TVectorRegister<Internal::uint64_2, uint64>;

    using float32_4 = Internal::TVectorRegister<Internal::float32_4, float32>;
    using float64_2 = Internal::TVectorRegister<Internal::float64_2, float64>;


#endif //AVX128

    namespace Internal
    {

#ifdef AVX256

        template<typename T>
        using Vector32 = T __attribute__((__vector_size__(32), __aligned__(32)));

        using char8_32 = Vector32<char8>;
        using int8_32 = Vector32<int8>;
        using uint8_32 = Vector32<uint8>;

        using int16_16 = Vector32<int16>;
        using uint16_16 = Vector32<uint16>;

        using int32_8 = Vector32<int32>;
        using uint32_8 = Vector32<uint32>;

        using int64_4 = Vector32<int64>;
        using uint64_4 = Vector32<uint64>;

        using float32_8 = Vector32<float32>;
        using float64_4 = Vector32<float64>;

        static_assert(alignof(char8_32) == 32);
        static_assert(alignof(int8_32) == 32);
        static_assert(alignof(uint8_32) == 32);

        static_assert(alignof(int16_16) == 32);
        static_assert(alignof(uint16_16) == 32);

        static_assert(alignof(int32_8) == 32);
        static_assert(alignof(uint32_8) == 32);

        static_assert(alignof(int64_4) == 32);
        static_assert(alignof(uint64_4) == 32);

        static_assert(alignof(float32_8) == 32);
        static_assert(alignof(float64_4) == 32);
    }

    using char8_32 = Internal::TVectorRegister<Internal::char8_32, char8>;
    using int8_32 = Internal::TVectorRegister<Internal::int8_32, int8>;
    using uint8_32 = Internal::TVectorRegister<Internal::uint8_32, uint8>;

    using int16_16 = Internal::TVectorRegister<Internal::int16_16, int16>;
    using uint16_16 = Internal::TVectorRegister<Internal::uint16_16, uint16>;

    using int32_8 = Internal::TVectorRegister<Internal::int32_8, int32>;
    using uint32_8 = Internal::TVectorRegister<Internal::uint32_8, uint32>;

    using int64_4 = Internal::TVectorRegister<Internal::int64_4, int64>;
    using uint64_4 = Internal::TVectorRegister<Internal::uint64_4, uint64>;

    using float32_8 = Internal::TVectorRegister<Internal::float32_8, float32>;
    using float64_4 = Internal::TVectorRegister<Internal::float64_4, float64>;

#endif //AVX256

    template<typename TVector>
    ATTRAVX consteval uint64 ElementSize()
    {
        return sizeof(typename TVector::ElementType);
    }

    ATTRAVX void ZeroUpper()
    {
#ifdef AVX256
        __builtin_ia32_vzeroupper();
#endif
    }

    ATTRAVX void ZeroAll()
    {
#ifdef AVX256
        __builtin_ia32_vzeroall();
#endif
    }

    template<typename TSource, int32... Control>
    ATTRAVX auto ShuffleVector(const TSource& Source)
    {
        return __builtin_shufflevector(Source.Vector, Source.Vector, Control...);
    }

    template<typename TSource, int32... Control>
    ATTRAVX auto ShuffleVector(const TSource& LHS, const TSource& RHS)
    {
        return __builtin_shufflevector(LHS.Vector, RHS.Vector, Control...);
    }

    template<typename TVector>
    ATTRAVX constexpr TVector SetAll(typename TVector::ElementType Value)
    {
        if constexpr(alignof(TVector) == 32)
        {
            if constexpr(ElementSize<TVector>() == 1)
            {
                return TVector{Value, Value, Value, Value, Value, Value, Value, Value,
                               Value, Value, Value, Value, Value, Value, Value, Value,
                               Value, Value, Value, Value, Value, Value, Value, Value,
                               Value, Value, Value, Value, Value, Value, Value, Value};
            }
            else if constexpr(ElementSize<TVector>() == 2)
            {
                return TVector{Value, Value, Value, Value, Value, Value, Value, Value,
                               Value, Value, Value, Value, Value, Value, Value, Value};
            }
            else if constexpr(ElementSize<TVector>() == 4)
            {
                return TVector{Value, Value, Value, Value, Value, Value, Value, Value};
            }
            else if constexpr(ElementSize<TVector>() == 8)
            {
                return TVector{Value, Value, Value, Value};
            }
        }
        else if constexpr(alignof(TVector) == 16)
        {
            if constexpr(ElementSize<TVector>() == 1)
            {
                return TVector{Value, Value, Value, Value, Value, Value, Value, Value,
                               Value, Value, Value, Value, Value, Value, Value, Value};
            }
            else if constexpr(ElementSize<TVector>() == 2)
            {
                return TVector{Value, Value, Value, Value, Value, Value, Value, Value};
            }
            else if constexpr(ElementSize<TVector>() == 4)
            {
                return TVector{Value, Value, Value, Value};
            }
            else if constexpr(ElementSize<TVector>() == 8)
            {
                return TVector{Value, Value};
            }
        }

    }

    template<typename TVector>
    ATTRAVX constexpr int32 CompareEqual(const TVector& LHS, const TVector& RHS)
    {
        if constexpr(alignof(TVector) == 32)
        {
            if constexpr(ElementSize<TVector>() <= 2)
            {
                return __builtin_ia32_pmovmskb256(LHS.Vector == RHS.Vector);
            }
            else if constexpr(ElementSize<TVector>() == 4)
            {
                return __builtin_ia32_movmskps256(LHS.Vector == RHS.Vector);
            }
            else if constexpr(ElementSize<TVector>() == 8)
            {
                return __builtin_ia32_movmskpd256(LHS.Vector == RHS.Vector);
            }
        }
        else if constexpr(alignof(TVector) == 16)
        {
            if constexpr(ElementSize<TVector>() <= 2)
            {
                return __builtin_ia32_pmovmskb128(LHS.Vector == RHS.Vector);
            }
            else if constexpr(ElementSize<TVector>() == 4)
            {
                return __builtin_ia32_movmskps(LHS.Vector == RHS.Vector);
            }
            else if constexpr(ElementSize<TVector>() == 8)
            {
                return __builtin_ia32_movmskpd(LHS.Vector == RHS.Vector);
            }
        }
    }

    template<typename TVector>
    ATTRAVX constexpr int32 CompareNotEqual(const TVector& LHS, const TVector& RHS)
    {
        if constexpr(alignof(TVector) == 32)
        {
            if constexpr(ElementSize<TVector>() <= 2)
            {
                return __builtin_ia32_pmovmskb256(LHS.Vector != RHS.Vector);
            }
            else if constexpr(ElementSize<TVector>() == 4)
            {
                return __builtin_ia32_movmskps256(LHS.Vector != RHS.Vector);
            }
            else if constexpr(ElementSize<TVector>() == 8)
            {
                return __builtin_ia32_movmskpd256(LHS.Vector != RHS.Vector);
            }
        }
        else if constexpr(alignof(TVector) == 16)
        {
            if constexpr(ElementSize<TVector>() <= 2)
            {
                return __builtin_ia32_pmovmskb128(LHS.Vector != RHS.Vector);
            }
            else if constexpr(ElementSize<TVector>() == 4)
            {
                return __builtin_ia32_movmskps(LHS.Vector != RHS.Vector);
            }
            else if constexpr(ElementSize<TVector>() == 8)
            {
                return __builtin_ia32_movmskpd(LHS.Vector != RHS.Vector);
            }
        }
    }

    template<typename TVector>
    ATTRAVX constexpr int32 CompareGreater(const TVector& LHS, const TVector& RHS)
    {
        if constexpr(alignof(TVector) == 32)
        {
            if constexpr(ElementSize<TVector>() <= 2)
            {
                return __builtin_ia32_pmovmskb256(LHS.Vector > RHS.Vector);
            }
            else if constexpr(ElementSize<TVector>() == 4)
            {
                return __builtin_ia32_movmskps256(LHS.Vector > RHS.Vector);
            }
            else if constexpr(ElementSize<TVector>() == 8)
            {
                return __builtin_ia32_movmskpd256(LHS.Vector == RHS.Vector);
            }
        }
        else if constexpr(alignof(TVector) == 16)
        {
            if constexpr(ElementSize<TVector>() <= 2)
            {
                return __builtin_ia32_pmovmskb128(LHS.Vector > RHS.Vector);
            }
            else if constexpr(ElementSize<TVector>() == 4)
            {
                return __builtin_ia32_movmskps(LHS.Vector > RHS.Vector);
            }
            else if constexpr(ElementSize<TVector>() == 8)
            {
                return __builtin_ia32_movmskpd(LHS.Vector > RHS.Vector);
            }
        }
    }

    template<typename TVector>
    ATTRAVX constexpr int32 CompareGreaterOrEqual(const TVector& LHS, const TVector& RHS)
    {
        if constexpr(alignof(TVector) == 32)
        {
            if constexpr(ElementSize<TVector>() <= 2)
            {
                return __builtin_ia32_pmovmskb256(LHS.Vector >= RHS.Vector);
            }
            else if constexpr(ElementSize<TVector>() == 4)
            {
                return __builtin_ia32_movmskps256(LHS.Vector >= RHS.Vector);
            }
            else if constexpr(ElementSize<TVector>() == 8)
            {
                return __builtin_ia32_movmskpd256(LHS.Vector == RHS.Vector);
            }
        }
        else if constexpr(alignof(TVector) == 16)
        {
            if constexpr(ElementSize<TVector>() <= 2)
            {
                return __builtin_ia32_pmovmskb128(LHS.Vector >= RHS.Vector);
            }
            else if constexpr(ElementSize<TVector>() == 4)
            {
                return __builtin_ia32_movmskps(LHS.Vector >= RHS.Vector);
            }
            else if constexpr(ElementSize<TVector>() == 8)
            {
                return __builtin_ia32_movmskpd(LHS.Vector >= RHS.Vector);
            }
        }
    }

    template<typename TVector>
    ATTRAVX constexpr int32 CompareLesser(const TVector& LHS, const TVector& RHS)
    {
        if constexpr(alignof(TVector) == 32)
        {
            if constexpr(ElementSize<TVector>() <= 2)
            {
                return __builtin_ia32_pmovmskb256(LHS.Vector < RHS.Vector);
            }
            else if constexpr(ElementSize<TVector>() == 4)
            {
                return __builtin_ia32_movmskps256(LHS.Vector < RHS.Vector);
            }
            else if constexpr(ElementSize<TVector>() == 8)
            {
                return __builtin_ia32_movmskpd256(LHS.Vector < RHS.Vector);
            }
        }
        else if constexpr(alignof(TVector) == 16)
        {
            if constexpr(ElementSize<TVector>() <= 2)
            {
                return __builtin_ia32_pmovmskb128(LHS.Vector < RHS.Vector);
            }
            else if constexpr(ElementSize<TVector>() == 4)
            {
                return __builtin_ia32_movmskps(LHS.Vector < RHS.Vector);
            }
            else if constexpr(ElementSize<TVector>() == 8)
            {
                return __builtin_ia32_movmskpd(LHS.Vector < RHS.Vector);
            }
        }
    }

    template<typename TVector>
    ATTRAVX constexpr int32 CompareLesserOrEqual(const TVector& LHS, const TVector& RHS)
    {
        if constexpr(alignof(TVector) == 32)
        {
            if constexpr(ElementSize<TVector>() <= 2)
            {
                return __builtin_ia32_pmovmskb256(LHS.Vector <= RHS.Vector);
            }
            else if constexpr(ElementSize<TVector>() == 4)
            {
                return __builtin_ia32_movmskps256(LHS.Vector <= RHS.Vector);
            }
            else if constexpr(ElementSize<TVector>() == 8)
            {
                return __builtin_ia32_movmskpd256(LHS.Vector <= RHS.Vector);
            }
        }
        else if constexpr(alignof(TVector) == 16)
        {
            if constexpr(ElementSize<TVector>() <= 2)
            {
                return __builtin_ia32_pmovmskb128(LHS.Vector <= RHS.Vector);
            }
            else if constexpr(ElementSize<TVector>() == 4)
            {
                return __builtin_ia32_movmskps(LHS.Vector <= RHS.Vector);
            }
            else if constexpr(ElementSize<TVector>() == 8)
            {
                return __builtin_ia32_movmskpd(LHS.Vector <= RHS.Vector);
            }
        }
    }

    template<typename TVector>
    ATTRAVX constexpr TVector MakeFromGreater(const TVector& LHS, const TVector& RHS)
    {
        if constexpr(alignof(TVector) == 32)
        {
            if constexpr(ElementSize<TVector>() == 1)
            {
                if constexpr(std::is_signed_v<typename TVector::ElementType>)
                {
                    return TVector{__builtin_ia32_pmaxsb256(LHS.Vector, RHS.Vector)};
                }
                else if constexpr(!std::is_signed_v<typename TVector::ElementType>)
                {
                    return TVector{__builtin_ia32_pmaxub256(LHS.Vector, RHS.Vector)};
                }
            }
            else if constexpr(ElementSize<TVector>() == 2)
            {
                if constexpr(std::is_signed_v<typename TVector::ElementType>)
                {
                    return TVector{__builtin_ia32_pmaxsw256(LHS.Vector, RHS.Vector)};
                }
                else if constexpr(!std::is_signed_v<typename TVector::ElementType>)
                {
                    return TVector{__builtin_ia32_pmaxuw256(LHS.Vector, RHS.Vector)};
                }
            }
            else if constexpr(ElementSize<TVector>() == 4)
            {
                return TVector{__builtin_ia32_maxps256(LHS.Vector, RHS.Vector)};
            }
            else if constexpr(ElementSize<TVector>() == 8)
            {
                return TVector{__builtin_ia32_maxpd256(LHS.Vector, RHS.Vector)};
            }
        }
        else if constexpr(alignof(TVector) == 16)
        {
            if constexpr(ElementSize<TVector>() == 1)
            {
                if constexpr(std::is_signed_v<typename TVector::ElementType>)
                {
                    return TVector{__builtin_ia32_pmaxsb128(LHS.Vector, RHS.Vector)};
                }
                else if constexpr(!std::is_signed_v<typename TVector::ElementType>)
                {
                    return TVector{__builtin_ia32_pmaxub128(LHS.Vector, RHS.Vector)};
                }
            }
            else if constexpr(ElementSize<TVector>() == 2)
            {
                if constexpr(std::is_signed_v<typename TVector::ElementType>)
                {
                    return TVector{__builtin_ia32_pmaxsw128(LHS.Vector, RHS.Vector)};
                }
                else if constexpr(!std::is_signed_v<typename TVector::ElementType>)
                {
                    return TVector{__builtin_ia32_pmaxuw128(LHS.Vector, RHS.Vector)};
                }
            }
            else if constexpr(ElementSize<TVector>() == 4)
            {
                return TVector{__builtin_ia32_maxps(LHS.Vector, RHS.Vector)};
            }
            else if constexpr(ElementSize<TVector>() == 8)
            {
                return TVector{__builtin_ia32_maxpd(LHS.Vector, RHS.Vector)};
            }
        }
    }

    template<typename TVector>
    ATTRAVX constexpr TVector MakeFromLesser(const TVector& LHS, const TVector& RHS)
    {
        if constexpr(alignof(TVector) == 32)
        {
            if constexpr(ElementSize<TVector>() == 1)
            {
                if constexpr(std::is_signed_v<typename TVector::ElementType>)
                {
                    return TVector{__builtin_ia32_pminsb256(LHS.Vector, RHS.Vector)};
                }
                else if constexpr(!std::is_signed_v<typename TVector::ElementType>)
                {
                    return TVector{__builtin_ia32_pminub256(LHS.Vector, RHS.Vector)};
                }
            }
            else if constexpr(ElementSize<TVector>() == 2)
            {
                if constexpr(std::is_signed_v<typename TVector::ElementType>)
                {
                    return TVector{__builtin_ia32_pminsw256(LHS.Vector, RHS.Vector)};
                }
                else if constexpr(!std::is_signed_v<typename TVector::ElementType>)
                {
                    return TVector{__builtin_ia32_pminuw256(LHS.Vector, RHS.Vector)};
                }
            }
            else if constexpr(ElementSize<TVector>() == 4)
            {
                return TVector{__builtin_ia32_minps256(LHS.Vector, RHS.Vector)};
            }
            else if constexpr(ElementSize<TVector>() == 8)
            {
                return TVector{__builtin_ia32_minpd256(LHS.Vector, RHS.Vector)};
            }
        }
        else if constexpr(alignof(TVector) == 16)
        {
            if constexpr(ElementSize<TVector>() == 1)
            {
                if constexpr(std::is_signed_v<typename TVector::ElementType>)
                {
                    return TVector{__builtin_ia32_pminsb128(LHS.Vector, RHS.Vector)};
                }
                else if constexpr(!std::is_signed_v<typename TVector::ElementType>)
                {
                    return TVector{__builtin_ia32_pminub128(LHS.Vector, RHS.Vector)};
                }
            }
            else if constexpr(ElementSize<TVector>() == 2)
            {
                if constexpr(std::is_signed_v<typename TVector::ElementType>)
                {
                    return TVector{__builtin_ia32_pminsw128(LHS.Vector, RHS.Vector)};
                }
                else if constexpr(!std::is_signed_v<typename TVector::ElementType>)
                {
                    return TVector{__builtin_ia32_pminuw128(LHS.Vector, RHS.Vector)};
                }
            }
            else if constexpr(ElementSize<TVector>() == 4)
            {
                return TVector{__builtin_ia32_minps(LHS.Vector, RHS.Vector)};
            }
            else if constexpr(ElementSize<TVector>() == 8)
            {
                return TVector{__builtin_ia32_minpd(LHS.Vector, RHS.Vector)};
            }
        }
    }

    template<typename TVector>
    ATTRAVX constexpr TVector Absolute(const TVector& Target)
    {
        if constexpr(alignof(TVector) == 32)
        {
            if constexpr(ElementSize<TVector>() == 1)
            {
                return TVector{__builtin_ia32_pabsb256(Target.Vector)};
            }
            else if constexpr(ElementSize<TVector>() == 2)
            {
                return TVector{__builtin_ia32_pabsw256(Target.Vector)};
            }
            else if constexpr(ElementSize<TVector>() == 4)
            {
                return TVector{__builtin_ia32_pabsd256(Target.Vector)};
            }
            else if constexpr(ElementSize<TVector>() == 8)
            {
#ifdef AVX512
                return TVector{__builtin_ia32_pabsq256(Target.Vector)};
#endif
            }
        }
        else if constexpr(alignof(TVector) == 16)
        {
            if constexpr(ElementSize<TVector>() == 1)
            {
                return TVector{__builtin_ia32_pabsb128(Target.Vector)};
            }
            else if constexpr(ElementSize<TVector>() == 2)
            {
                return TVector{__builtin_ia32_pabsw128(Target.Vector)};
            }
            else if constexpr(ElementSize<TVector>() == 4)
            {
                return TVector{__builtin_ia32_pabsd128(Target.Vector)};
            }
            else if constexpr(ElementSize<TVector>() == 8)
            {
#ifdef AVX512
                return TVector{__builtin_ia32_pabsq128(Target.Vector)};
#endif
            }
        }
    }

    template<typename TVector, typename DataType = typename TVector::ElementType>
    ATTRAVX constexpr TVector Load(const DataType* Data)
    {
        using InternalVector = typename TVector::VectorType;

        struct FSource
        {
            InternalVector Vector;
        }
        __attribute__((__packed__, __may_alias__));

        return TVector{reinterpret_cast<InternalVector>(reinterpret_cast<const FSource*>(Data)->Vector)};
    }

    template<typename TVector>
    ATTRAVX TVector ShuffleLeft(const TVector& Source, const int32 ShuffleAmount)
    {
        static constinit typename TVector::ElementType Buffer[TVector::NumElements * 3]{};

        Memory::Copy(&Buffer[TVector::NumElements], Source.ToPtr(), TVector::NumElements * ElementSize<TVector>());

        return Load<TVector>(&Buffer[TVector::NumElements - ShuffleAmount]);
    }

    template<typename TVector>
    ATTRAVX constexpr TVector ShuffleRight(const TVector& Source, const int32 ShuffleAmount)
    {
        return ShuffleLeft(Source, ShuffleAmount * -1);
    }

    namespace Internal
    {

        template<typename InVectorType, typename InElementType>
        class alignas(InVectorType) TVectorRegister final
        {
        public:

            using VectorType = InVectorType;
            using ElementType = InElementType;

            inline static const constinit uint64 NumElements = []() consteval -> uint64
            {
                return sizeof(VectorType) / sizeof(ElementType);
            }();

            inline static const constinit int32 ComparisonMask = []() consteval -> int32
            {
                if constexpr(alignof(TVectorRegister) == 32)
                {
                    switch(NumElements)
                    {
                        case 32: return (int32)0b11111111111111111111111111111111;
                        case 16: return (int32)0b11111111111111111111111111111111;
                        case 8:  return (int32)0b00000000000000000000000011111111;
                        case 4:  return (int32)0b00000000000000000000000000001111;
                        case 2:  return (int32)0b00000000000000000000000000000011;
                        default: ASSERT(false); return 0;
                    }
                }
                else if constexpr(alignof(TVectorRegister) == 16)
                {
                    switch(NumElements)
                    {
                        case 16: return (int32)0b00000000000000001111111111111111;
                        case 8:  return (int32)0b00000000000000000000000011111111;
                        case 4:  return (int32)0b00000000000000000000000000001111;
                        case 2:  return (int32)0b00000000000000000000000000000011;
                        default: ASSERT(false); return 0;
                    }
                }
            }();

        public:

            ATTRAVX explicit constexpr TVectorRegister(ElementType Value = static_cast<ElementType>(0))
                : Vector(Simd::SetAll<TVectorRegister>(Value).Vector)
            {
            }

            template<typename... Elements>
            ATTRAVX explicit constexpr TVectorRegister(Elements... ElementValues)
                : Vector{static_cast<ElementType>(ElementValues)...}
            {
                static_assert(NumElements == sizeof...(ElementValues));
            }

            ATTRAVX TVectorRegister(const TVectorRegister& Other)
                : Vector(Other.Vector)
            {
            }

            ATTRAVX TVectorRegister(TVectorRegister&& Other) noexcept
                : Vector(static_cast<VectorType&&>(Other.Vector))
            {
            }

            ATTRAVX explicit constexpr TVectorRegister(const VectorType& Other)
                : Vector(Other)
            {
            }

            ATTRAVX explicit TVectorRegister(VectorType&& Other)
                : Vector(static_cast<VectorType&&>(Other))
            {
            }

            ATTRAVX ElementType* ToPtr()
            {
                return Memory::AssumeAligned<alignof(VectorType)>(reinterpret_cast<ElementType*>(&Vector));
            }

            ATTRAVX const ElementType* ToPtr() const
            {
                return Memory::AssumeAligned<alignof(VectorType)>(reinterpret_cast<const ElementType*>(&Vector));
            }

        public:

            template<typename IndexType>
            ATTRAVX ElementType& operator[](const IndexType Index)
            {
                return ToPtr()[Index];
            }

            template<typename IndexType>
            ATTRAVX ElementType operator[](const IndexType Index) const
            {
                return Vector[Index];
            }

            ATTRAVX TVectorRegister& operator=(const VectorType& Other)
            {
                Vector = Other;
                return *this;
            }

            ATTRAVX TVectorRegister& operator=(const TVectorRegister& Other)
            {
                Vector = Other.Vector;
                return *this;
            }

            ATTRAVX TVectorRegister& operator=(TVectorRegister&& Other)
            {
                Vector = static_cast<VectorType&&>(Other.Vector);
                return *this;
            }

            ATTRAVX TVectorRegister operator+(const TVectorRegister& Other) const
            {
                return TVectorRegister{Vector + Other.Vector};
            }

            ATTRAVX TVectorRegister operator-(const TVectorRegister& Other) const
            {
                return TVectorRegister{Vector - Other.Vector};
            }

            ATTRAVX TVectorRegister operator*(const TVectorRegister& Other) const
            {
                return TVectorRegister{Vector * Other.Vector};
            }

            ATTRAVX TVectorRegister operator/(const TVectorRegister& Other) const
            {
                return TVectorRegister{Vector / Other.Vector};
            }

            ATTRAVX TVectorRegister& operator+=(const TVectorRegister& Other)
            {
                Vector += Other.Vector;
                return *this;
            }

            ATTRAVX TVectorRegister& operator-=(const TVectorRegister& Other)
            {
                Vector -= Other.Vector;
                return *this;
            }

            ATTRAVX TVectorRegister& operator*=(const TVectorRegister& Other)
            {
                Vector *= Other.Vector;
                return *this;
            }

            ATTRAVX TVectorRegister& operator/=(const TVectorRegister& Other) const
            {
                Vector /= Other.Vector;
                return *this;
            }
            //To see if ANY element does match, compare the result of this function against != 0
            //To see if ALL elements do match, compare the result of this function against == ComparisonMask
            ATTRAVX int32 operator==(const TVectorRegister& Other) const
            {
                return CompareEqual(*this, Other);
            }
            //To see if ANY element does not match, compare the result of this function against != 0
            //To see if ALL elements do not match, compare the result of this function against == ComparisonMask
            ATTRAVX int32 operator!=(const TVectorRegister& Other) const
            {
                return CompareNotEqual(*this, Other);
            }
            //To see if ANY element does match, compare the result of this function against != 0
            //To see if ALL elements do match, compare the result of this function against == ComparisonMask
            ATTRAVX int32 operator>(const TVectorRegister& Other) const
            {
                return CompareGreater(*this, Other);
            }
            //To see if ANY element does match, compare the result of this function against != 0
            //To see if ALL elements do match, compare the result of this function against == ComparisonMask
            ATTRAVX int32 operator>=(const TVectorRegister& Other) const
            {
                return CompareGreaterOrEqual(*this, Other);
            }
            //To see if ANY element does match, compare the result of this function against != 0
            //To see if ALL elements do match, compare the result of this function against == ComparisonMask
            ATTRAVX int32 operator<(const TVectorRegister& Other) const
            {
                return CompareLesser(*this, Other);
            }
            //To see if ANY element does match, compare the result of this function against != 0
            //To see if ALL elements do match, compare the result of this function against == ComparisonMask
            ATTRAVX int32 operator<=(const TVectorRegister& Other) const
            {
                return CompareLesserOrEqual(*this, Other);
            }

            ATTRAVX TVectorRegister operator&(const TVectorRegister& Other) const
            {
                return TVectorRegister{Vector & Other.Vector};
            }

            ATTRAVX TVectorRegister& operator&=(const TVectorRegister& Other)
            {
                Vector &= Other.Vector;
                return *this;
            }

            ATTRAVX TVectorRegister operator^(const TVectorRegister& Other) const
            {
                return TVectorRegister{Vector ^ Other.Vector};
            }

            ATTRAVX TVectorRegister& operator^=(const TVectorRegister& Other)
            {
                Vector ^= Other.Vector;
                return *this;
            }

            ATTRAVX TVectorRegister operator|(const TVectorRegister& Other) const
            {
                return TVectorRegister{Vector | Other.Vector};
            }

            ATTRAVX TVectorRegister& operator|=(const TVectorRegister& Other)
            {
                Vector |= Other.Vector;
                return *this;
            }

            ATTRAVX TVectorRegister& operator>>=(const int32 ShuffleAmount)
            {
                Vector = ShuffleRight(*this, ShuffleAmount).Vector;
                return *this;
            }

            ATTRAVX TVectorRegister operator>>(const int32 ShuffleAmount)
            {
                return TVectorRegister{ShuffleRight(*this, ShuffleAmount)};
            }

            ATTRAVX TVectorRegister& operator<<=(const int32 ShuffleAmount)
            {
                Vector = ShuffleLeft(*this, ShuffleAmount).Vector;
                return *this;
            }

            ATTRAVX TVectorRegister operator<<(const int32 ShuffleAmount) const
            {
                return TVectorRegister{ShuffleLeft(*this, ShuffleAmount)};
            }

            ATTRAVX TVectorRegister& operator+()
            {
                Vector = Simd::Absolute(*this).Vector;
                return *this;
            }

            ATTRAVX TVectorRegister& operator-()
            {
                Vector = ~Simd::Absolute(*this).Vector;
                return *this;
            }

            ATTRAVX void operator--()
            {
                Vector -= Simd::SetAll<TVectorRegister>(1);
            }

            ATTRAVX void operator++()
            {
                Vector += Simd::SetAll<TVectorRegister>(1);
            }

        public:

            VectorType Vector;

        };

    }

    #ifdef AVX128

    static_assert(alignof(char8_16) == 16);
    static_assert(alignof(int8_16) == 16);
    static_assert(alignof(uint8_16) == 16);

    static_assert(alignof(int16_8) == 16);
    static_assert(alignof(uint16_8) == 16);

    static_assert(alignof(int32_4) == 16);
    static_assert(alignof(uint32_4) == 16);

    static_assert(alignof(int64_2) == 16);
    static_assert(alignof(uint64_2) == 16);

    static_assert(alignof(float32_4) == 16);
    static_assert(alignof(float64_2) == 16);

    #endif
    #ifdef AVX256

    static_assert(alignof(char8_32) == 32);
    static_assert(alignof(int8_32) == 32);
    static_assert(alignof(uint8_32) == 32);

    static_assert(alignof(int16_16) == 32);
    static_assert(alignof(uint16_16) == 32);

    static_assert(alignof(int32_8) == 32);
    static_assert(alignof(uint32_8) == 32);

    static_assert(alignof(int64_4) == 32);
    static_assert(alignof(uint64_4) == 32);


    static_assert(alignof(float32_8) == 32);
    static_assert(alignof(float64_4) == 32);

    #endif

}

#ifdef AVX128

ATTRAVX consteval Simd::char8_16 operator"" _char8_16(uint64 Value)
{
    return Simd::SetAll<Simd::char8_16>(Value);
}

ATTRAVX consteval Simd::int8_16 operator"" _int8_16(uint64 Value)
{
    return Simd::SetAll<Simd::int8_16>(Value);
}

ATTRAVX consteval Simd::uint8_16 operator"" _uint8_16(uint64 Value)
{
    return Simd::SetAll<Simd::uint8_16>(Value);
}

ATTRAVX consteval Simd::int16_8 operator"" _int16_8(uint64 Value)
{
    return Simd::SetAll<Simd::int16_8>(Value);
}

ATTRAVX consteval Simd::uint16_8 operator"" _uint16_8(uint64 Value)
{
    return Simd::SetAll<Simd::uint16_8>(Value);
}

ATTRAVX consteval Simd::int32_4 operator"" _int32_4(uint64 Value)
{
    return Simd::SetAll<Simd::int32_4>(Value);
}

ATTRAVX consteval Simd::uint32_4 operator"" _uint32_4(uint64 Value)
{
    return Simd::SetAll<Simd::uint32_4>(Value);
}

ATTRAVX consteval Simd::int64_2 operator"" _int64_2(uint64 Value)
{
    return Simd::SetAll<Simd::int64_2>(Value);
}

ATTRAVX consteval Simd::uint64_2 operator"" _uint64_2(uint64 Value)
{
    return Simd::SetAll<Simd::uint64_2>(Value);
}

ATTRAVX consteval Simd::float32_4 operator"" _float32_4(float128 Value)
{
    return Simd::SetAll<Simd::float32_4>(Value);
}

ATTRAVX consteval Simd::float64_2 operator"" _float64_2(float128 Value)
{
    return Simd::SetAll<Simd::float64_2>(Value);
}

#endif //AVX128
#ifdef AVX256

ATTRAVX consteval Simd::char8_32 operator"" _char8_32(uint64 Value)
{
    return Simd::SetAll<Simd::char8_32>(Value);
}

ATTRAVX consteval Simd::char8_32 operator"" _char8_32(char8 Value)
{
    return Simd::SetAll<Simd::char8_32>(Value);
}

ATTRAVX consteval Simd::int8_32 operator"" _int8_32(uint64 Value)
{
    return Simd::SetAll<Simd::int8_32>(Value);
}

ATTRAVX consteval Simd::uint8_32 operator"" _uint8_32(uint64 Value)
{
    return Simd::SetAll<Simd::uint8_32>(Value);
}

ATTRAVX consteval Simd::int16_16 operator"" _int16_16(uint64 Value)
{
    return Simd::SetAll<Simd::int16_16>(Value);
}

ATTRAVX consteval Simd::uint16_16 operator"" _uint16_16(uint64 Value)
{
    return Simd::SetAll<Simd::uint16_16>(Value);
}

ATTRAVX consteval Simd::int32_8 operator"" _int32_8(uint64 Value)
{
    return Simd::SetAll<Simd::int32_8>(Value);
}

ATTRAVX consteval Simd::uint32_8 operator"" _uint32_8(uint64 Value)
{
    return Simd::SetAll<Simd::uint32_8>(Value);
}

ATTRAVX consteval Simd::int64_4 operator"" _int64_4(uint64 Value)
{
    return Simd::SetAll<Simd::int64_4>(Value);
}

ATTRAVX consteval Simd::uint64_4 operator"" _uint64_4(uint64 Value)
{
    return Simd::SetAll<Simd::uint64_4>(Value);
}

ATTRAVX consteval Simd::float32_8 operator"" _float32_8(float128 Value)
{
    return Simd::SetAll<Simd::float32_8>(Value);
}

ATTRAVX consteval Simd::float64_4 operator"" _float64_4(float128 Value)
{
    return Simd::SetAll<Simd::float64_4>(Value);
}

#endif //AVX256

