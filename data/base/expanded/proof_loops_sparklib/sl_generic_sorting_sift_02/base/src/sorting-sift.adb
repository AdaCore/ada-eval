--
--  Copyright (C) 2004-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

separate (Sorting)
procedure Sift
  (Container : in out Vector;
   S         : T;
   Max       : T;
   Temp      : Element_Type;
   C0        : T)
is
   function To_Index (J : T) return Array_Index
   is (Array_Index (J))
   with Pre => J in 1 .. T (Length (Container));
   --  The array index subtype starts at 1, hence To_Index is the
   --  identity on the heapsort indices 1 .. Max.

   C : T := C0;

begin
   while C /= S loop
      pragma Loop_Variant (Static => (Decreases => C));
      declare
         Father : constant T := C / 2;
      begin
         if Container.Elements (To_Index (Father)).V < Temp then
            Container.Elements (To_Index (C)).V :=
              Container.Elements (To_Index (Father)).V;
            C := Father;
         else
            exit;
         end if;
      end;
   end loop;

   Container.Elements (To_Index (C)).V := Temp;
end Sift;
