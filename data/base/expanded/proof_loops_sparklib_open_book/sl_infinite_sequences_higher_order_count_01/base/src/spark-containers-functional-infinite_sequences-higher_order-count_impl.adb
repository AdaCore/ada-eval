--
--  Copyright (C) 2023-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;
separate (SPARK.Containers.Functional.Infinite_Sequences.Higher_Order)
function Count_Impl
  (S    : Sequence;
   Last : Big_Natural;
   Test : not null access function (E : Element_Type) return Boolean)
   return Big_Natural
is
begin
   return Res : Big_Natural := 0 do
      for I in Interval'(1, Last) loop
         if Test (Get (S, I)) then
            Res := Res + 1;
         end if;
      end loop;
   end return;
end Count_Impl;
