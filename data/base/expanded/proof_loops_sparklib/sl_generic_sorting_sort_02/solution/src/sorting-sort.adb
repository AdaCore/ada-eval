--
--  Copyright (C) 2004-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

separate (Sorting)
procedure Sort (Container : in out Vector) is
   function To_Index (J : T) return Array_Index
   is (Array_Index (J))
   with Pre => J in 1 .. T (Length (Container));
   --  The array index subtype starts at 1, hence To_Index is the
   --  identity on the heapsort indices 1 .. Max.

   Max  : T := T (Length (Container));
   Temp : Element_Type;

begin
   if Container.Last <= Index_Type'First then
      return;
   end if;

   while Max > 1 loop
      pragma Loop_Invariant (Static => Max in 1 .. T (Length (Container)));
      pragma
        Loop_Invariant
          (Static => Length (Container) = Length (Container)'Loop_Entry);
      pragma Loop_Variant (Static => (Decreases => Max));
      Temp := Container.Elements (To_Index (Max)).V;
      Container.Elements (To_Index (Max)).V := Container.Elements (1).V;

      Max := Max - 1;
      Sift (1, Container, Max, Temp);
   end loop;
end Sort;
