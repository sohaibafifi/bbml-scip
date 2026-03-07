\ SCIP STATISTICS
\   Problem name     : enlight4
\   Variables        : 32 (16 binary, 16 integer, 0 implicit integer, 0 continuous)
\   Constraints      : 16
Minimize
 Obj: +1 x#4#4 +1 x#1#1 +1 x#1#2 +1 x#1#3 +1 x#1#4 +1 x#2#1 +1 x#2#2 +1 x#2#3 +1 x#2#4 +1 x#3#1 +1 x#3#2
      +1 x#3#3 +1 x#3#4 +1 x#4#1 +1 x#4#2 +1 x#4#3
Subject to
 inner_area_1: +1 x#1#2 +1 x#2#1 +1 x#2#2 +1 x#2#3 +1 x#3#2 -2 y#2#2 = +0
 inner_area_2: +1 x#1#3 +1 x#2#2 +1 x#2#3 +1 x#2#4 +1 x#3#3 -2 y#2#3 = +0
 inner_area_3: +1 x#2#2 +1 x#3#1 +1 x#3#2 +1 x#3#3 +1 x#4#2 -2 y#3#2 = +0
 inner_area_4: +1 x#2#3 +1 x#3#2 +1 x#3#3 +1 x#3#4 +1 x#4#3 -2 y#3#3 = +0
 upper_border_1: +1 x#1#1 +1 x#1#2 +1 x#1#3 +1 x#2#2 -2 y#1#2 = +0
 upper_border_2: +1 x#1#2 +1 x#1#3 +1 x#1#4 +1 x#2#3 -2 y#1#3 = +0
 lower_border_1: +1 x#3#2 +1 x#4#1 +1 x#4#2 +1 x#4#3 -2 y#4#2 = +0
 lower_border_2: +1 x#3#3 +1 x#4#2 +1 x#4#3 +1 x#4#4 -2 y#4#3 = +0
 left_border_1: +1 x#1#1 +1 x#2#1 +1 x#2#2 +1 x#3#1 -2 y#2#1 = +0
 left_border_2: +1 x#2#1 +1 x#3#1 +1 x#3#2 +1 x#4#1 -2 y#3#1 = +0
 right_border_1: +1 x#1#4 +1 x#2#3 +1 x#2#4 +1 x#3#4 -2 y#2#4 = +0
 right_border_2: +1 x#2#4 +1 x#3#3 +1 x#3#4 +1 x#4#4 -2 y#3#4 = +0
 left_upper_cor@c: +1 x#1#1 +1 x#1#2 +1 x#2#1 -2 y#1#1 = -1
 left_lower_cor@d: +1 x#3#1 +1 x#4#1 +1 x#4#2 -2 y#4#1 = +0
 right_upper_co@e: +1 x#1#3 +1 x#1#4 +1 x#2#4 -2 y#1#4 = +0
 right_lower_co@f: +1 x#3#4 +1 x#4#3 +1 x#4#4 -2 y#4#4 = +0
Bounds
 0 <= x#4#4 <= 1
 0 <= x#1#1 <= 1
 0 <= x#1#2 <= 1
 0 <= x#1#3 <= 1
 0 <= x#1#4 <= 1
 0 <= x#2#1 <= 1
 0 <= x#2#2 <= 1
 0 <= x#2#3 <= 1
 0 <= x#2#4 <= 1
 0 <= x#3#1 <= 1
 0 <= x#3#2 <= 1
 0 <= x#3#3 <= 1
 0 <= x#3#4 <= 1
 0 <= x#4#1 <= 1
 0 <= x#4#2 <= 1
 0 <= x#4#3 <= 1
Binaries
 x#4#4 x#1#1 x#1#2 x#1#3 x#1#4 x#2#1 x#2#2 x#2#3 x#2#4 x#3#1 x#3#2 x#3#3 x#3#4 x#4#1 x#4#2 x#4#3
Generals
 y#1#4 y#4#1 y#1#1 y#3#4 y#2#4 y#3#1 y#2#1 y#4#3 y#4#2 y#1#3 y#1#2 y#3#3 y#3#2 y#2#3 y#2#2 y#4#4
End
