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

#include "Simd.h"

#ifndef NULL_CHAR
#define NULL_CHAR static_cast<const char8>('\0')
#endif

namespace StringUtility
{

    PURE uint64 Length(const char8* String);

}


//31 character string made for fast comparisons
class FStaticString final
{
public:

    inline static const constinit int32 ComparisonMask{Simd::char8_32::ComparisonMask};
    inline static const constinit uint64 NumCharacters{Simd::char8_32::NumElements};

    //we need a static method to construct from a raw character array since the compiler will otherwise dismiss template constructors
    static FStaticString MakeFromRaw(const char8* RawString);

    constexpr FStaticString();

    explicit FStaticString(Simd::char8_32 OtherString);

    template<uint64 N>
    inline explicit constexpr FStaticString(const char8 (&StringSource)[N]);

    FStaticString& operator=(Simd::char8_32 OtherString);

    template<uint64 N>
    inline FStaticString& operator=(const char8 (&StringSource)[N]);

    bool operator==(const FStaticString& Other) const;

    template<uint64 N>
    inline bool operator==(const char8 (&StringSource)[N]) const;

    bool operator!=(const FStaticString& Other) const;

    template<uint64 N>
    inline bool operator!=(const char8 (&StringSource)[N]) const;

    inline char8 operator[](const int32 Index) const
    {
        return String[Index];
    }

    inline char8& operator[](const int32 Index)
    {
        return RawString()[Index];
    }

    inline const char8* RawString() const
    {
        return Memory::AssumeAligned<32>(reinterpret_cast<const char8*>(&String.Vector));
    }

    inline char8* RawString()
    {
        return Memory::AssumeAligned<32>(reinterpret_cast<char8*>(&String.Vector));
    }

    uint32 Length() const;

    FStaticString& Append(const FStaticString& Other);

    template<uint64 N>
    inline FStaticString& Append(const char8 (&StringSource)[N]);

    FStaticString& PushBack(const FStaticString& Other);

    template<uint64 N>
    inline FStaticString& PushBack(const char8 (&StringSource)[N]);

    bool Contains(FStaticString Other) const;

    template<uint64 N>
    inline bool Contains(const char8 (&StringSource)[N]) const;

    FStaticString& ToUppercase();
    FStaticString& ToLowercase();

    void RemoveFromEnd(const int32 Num);
    void RemoveFromStart(const int32 Num);

private:

    Simd::char8_32 String;

};

template<uint64 N>
constexpr FStaticString::FStaticString(const char8 (&StringSource)[N])
{
    static_assert(N <= NumCharacters);

    Memory::Copy(&String.Vector, StringSource, N);
}

template<uint64 N>
FStaticString& FStaticString::operator=(const char8 (&StringSource)[N])
{
    static_assert(N <= NumCharacters);

    Memory::Copy(&String.Vector, StringSource, N);

    return *this;
}

template<uint64 N>
bool FStaticString::operator==(const char8 (&StringSource)[N]) const
{
    return operator==(FStaticString{StringSource});
}

template<uint64 N>
bool FStaticString::operator!=(const char8 (&StringSource)[N]) const
{
    return operator!=(FStaticString{StringSource});
}

template<uint64 N>
FStaticString& FStaticString::Append(const char8 (&StringSource)[N])
{
    return Append(FStaticString{StringSource});
}

template<uint64 N>
FStaticString& FStaticString::PushBack(const char8 (&StringSource)[N])
{
    return PushBack(FStaticString{StringSource});
}

template<uint64 N>
bool FStaticString::Contains(const char8 (&StringSource)[N]) const
{
    return Contains(FStaticString{StringSource});
}