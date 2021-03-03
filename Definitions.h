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

using char8 = char;

using uint8 = unsigned char;
using int8 = signed char;
using uint16 = unsigned short;
using int16 = signed short;
using uint32 = unsigned int;
using int32 = signed int;
using uint64 = unsigned long long;
using int64 = signed long long;

using int128 = __int128_t;
using uint128 = __uint128_t;

using float32 = float;
using float64 = double;
using float128 = long double;

#define ATTRINLINE __attribute__((always_inline))
#define INLINE inline ATTRINLINE
#define EXPECT(cond, tf) (__builtin_expect((cond), tf))
#define PURE __attribute__((pure))

#define ASSERT(expr) /* todo */
