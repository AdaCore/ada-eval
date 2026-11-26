--
--  Copyright (C) 2004-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

--  Generic_Sorting.Sort's sift-down loop, isolated.

pragma Ada_2022;

with SPARK.Containers.Types; use SPARK.Containers.Types;
with Vectors; use Vectors;

package Sorting with SPARK_Mode, Always_Terminates is

   subtype T is Long_Long_Integer;

   procedure Sift
     (Container : in out Vector; S : T; Max : T; Temp : Element_Type)
   with
     Global => null,
     Pre  =>
       (Static => S in 1 .. Max and then Max <= T (Length (Container))),
     Post => (Static => Length (Container) = Length (Container)'Old);
   --  Sift the element held in Temp down the heap rooted at S, over the
   --  heap region 1 .. Max. Only permutes Elements (1 .. Max); leaves
   --  the vector's length unchanged.

end Sorting;
