--
--  Copyright (C) 2004-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

package body Sorting with SPARK_Mode is

   subtype T is Long_Long_Integer;

   procedure Sift
     (S : T; Container : in out Vector; Max : T; Temp : Element_Type)
   with
     Import,
     Global => null,
     Pre    =>
       (Static => S in 1 .. Max and then Max <= T (Length (Container))),
     Post   => (Static => Length (Container) = Length (Container)'Old);
   --  Sift the element held in Temp down the heap rooted at S, over the
   --  heap region 1 .. Max. Only permutes Elements (1 .. Max); leaves
   --  the vector's length unchanged.

   ----------
   -- Sort --
   ----------

   procedure Sort (Container : in out Vector) is separate;

end Sorting;
