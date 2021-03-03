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

#include "Definitions.h"

#ifndef INLINE
#define INLINE inline __attribute__((always_inline))
#endif

namespace Math
{

    template<typename T>
    INLINE T SignExtend(T Arg)
    {
        return Arg >> ((8 * sizeof(T)) - 1);
    }

    template<typename T>
    INLINE T Absolute(T Arg)
    {
        const T Mask{SignExtend(Arg)};
        return (Arg ^ Mask) - Mask;
    }

    template<typename T>
    INLINE T NegativeAbsolute(T Arg)
    {
        const T Mask{~SignExtend(Arg)};
        return (Arg ^ Mask) - Mask;
    }

    template<typename T>
    INLINE int32 NumActiveBits(T Arg)
    {
        if constexpr(sizeof(T) <= 4)
        {
            return __builtin_popcount(Arg);
        }
        else if constexpr(sizeof(T) == 8)
        {
            return __builtin_popcountll(Arg);
        }
    }

    template<typename ChoiceType, typename... TChoices>
    ChoiceType ConditionalChoose(const uint64 Condition, TChoices... Choices)
    {
        struct FChoiceWrapper
        {
            INLINE ChoiceType operator[](const uint64 Index) const &&
            {
                return Choices[Index];
            }

            const ChoiceType Choices[sizeof...(TChoices)];
        };

        return FChoiceWrapper{Choices...}[Condition];
    }
}