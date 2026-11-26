--
--  Copyright (C) 2004-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

separate (Sorting)
procedure Sift
  (Container : in out Vector; S : T; Max : T; Temp : Element_Type)
is
   function To_Index (J : T) return Array_Index
   is (Array_Index (J))
   with Pre => J in 1 .. T (Length (Container));
   --  The array index subtype starts at 1, hence To_Index is the
   --  identity on the heapsort indices 1 .. Max.

   C   : T := S;
   Son : T;

begin
   loop
      pragma Loop_Variant (Static => (Increases => C));
      Son := 2 * C;

      exit when Son > Max;

      declare
         Son_Index : Array_Index := To_Index (Son);

      begin
         if Son < Max then
            if Container.Elements (Son_Index).V
              < Container.Elements (Array_Index'Succ (Son_Index)).V
            then
               Son := Son + 1;
               Son_Index := Array_Index'Succ (Son_Index);
            end if;
         end if;

         Container.Elements (To_Index (C)).V :=
           Container.Elements (Son_Index).V;
      end;

      C := Son;
   end loop;

   Container.Elements (To_Index (C)).V := Temp;
end Sift;
