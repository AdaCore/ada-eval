--
--  Copyright (C) 2023-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;
separate (SPARK.Containers.Functional.Sets.Higher_Order)
function Count_Impl
  (S    : Set;
   Test : not null access function (E : Element_Type) return Boolean)
   return Big_Natural
is
begin
   return Res : Big_Natural := 0 do
      for Subset in Iterate (S) loop
         pragma
           Loop_Invariant
             (Static =>
                Count_Rec (S, Test) = Res + Count_Rec (Subset, Test));
         if Test (Choose (Subset)) then
            Res := Res + 1;
         end if;
      end loop;
   end return;
end Count_Impl;
