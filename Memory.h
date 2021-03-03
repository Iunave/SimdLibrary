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

namespace Memory
{

    //Copies count bytes from the object pointed to by Source to the object pointed to by Destination. Both objects are reinterpreted as arrays of unsigned char.
    template<typename TTarget, typename TSource>
    INLINE constexpr decltype(auto) Copy(TTarget* Destination, const TSource* Source, const uint64 Num)
    {
        return __builtin_memcpy(Destination, Source, Num);
    }

    template<typename TTarget>
    INLINE constexpr decltype(auto) Set(TTarget* Destination, const int32 Value, const uint64 Num)
    {
        return __builtin_memset(Destination, Value, Num);
    }

    template<uint64 Alignment, typename TTarget>
    INLINE constexpr TTarget AssumeAligned(const TTarget Destination)
    {
        return static_cast<TTarget>(__builtin_assume_aligned(Destination, Alignment));
    }

}
