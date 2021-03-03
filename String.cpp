#include "String.h"
#include "Math.h"

uint64 StringUtility::Length(const char8* String)
{
    uint64 Length{0};

    while(String[Length] != NULL_CHAR)
    {
        ++Length;
    }

    return Length;
}

FStaticString FStaticString::MakeFromRaw(const char8* RawString)
{
    const uint64 StringLength{StringUtility::Length(RawString)};

    FStaticString ResultString{};

    if(StringLength < 32) //todo assert here
    {
        Memory::Copy(&ResultString.String.Vector, RawString, StringLength);
    }

    return ResultString;
}

constexpr FStaticString::FStaticString()
    : String(NULL_CHAR)
{
}

FStaticString::FStaticString(Simd::char8_32 OtherString)
    : String(static_cast<Simd::char8_32&&>(OtherString))
{
}

FStaticString& FStaticString::operator=(Simd::char8_32 OtherString)
{
    String = static_cast<Simd::char8_32&&>(OtherString);
    return *this;
}

bool FStaticString::operator==(const FStaticString& Other) const
{
    return (String == Other.String) == ComparisonMask;
}

bool FStaticString::operator!=(const FStaticString& Other) const
{
    return (String != Other.String) == ComparisonMask;
}

uint32 FStaticString::Length() const
{
    const int32 BitMask{String != 0_char8_32};

    return Math::NumActiveBits(static_cast<uint32>(BitMask));
}

FStaticString& FStaticString::Append(const FStaticString& Other)
{
    String += (Other.String << this->Length());

    return *this;
}

FStaticString& FStaticString::PushBack(const FStaticString& Other)
{
    String <<= Other.Length();
    String += Other.String;

    return *this;
}

bool FStaticString::Contains(FStaticString Other) const
{
    Simd::char8_32 ComparisonResult;

    for(uint32 Index{this->Length()}; Index > 0; --Index)
    {
        ComparisonResult = String.Vector == Other.String.Vector;
        ComparisonResult &= 1_char8_32;
        ComparisonResult *= String;

        if EXPECT((ComparisonResult == Other.String) == ComparisonMask, false)
        {
            return true;
        }

        Other.String <<= 1;
    }

    return false;
}

FStaticString& FStaticString::ToUppercase()
{
    const Simd::char8_32 Mask{String.Vector >= 'a' && String.Vector <= 'z'};

    String -= (32_char8_32 & Mask);

    return *this;
}

FStaticString& FStaticString::ToLowercase()
{
    const Simd::char8_32 Mask{String.Vector >= 'A' && String.Vector <= 'Z'};

    String += (32_char8_32 & Mask);

    return *this;
}

void FStaticString::RemoveFromEnd(const int32 Num)
{
    uint32 LengthToEnd{32 - Length()};
    String <<= LengthToEnd + Num;

    LengthToEnd = 32 - Length();
    String >>= LengthToEnd;
}

void FStaticString::RemoveFromStart(const int32 Num)
{
    String >>= Num;
}

