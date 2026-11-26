--
--  Copyright (C) 2023-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;
separate (SPARK.Containers.Functional.Vectors.Higher_Order)
function Count_Impl
  (S    : Sequence;
   Last : Extended_Index;
   Test : not null access function (E : Element_Type) return Boolean)
   return Big_Natural
is
begin
   return Res : Big_Natural := 0 do
      for I in Index_Type'First .. Last loop
         if Test (Get (S, I)) then
            Res := Res + 1;
         end if;
         pragma Loop_Invariant (Static => I <= Last);
         pragma Loop_Invariant (Static => Res = Count_Rec (S, I, Test));
      end loop;
   end return;
end Count_Impl;
