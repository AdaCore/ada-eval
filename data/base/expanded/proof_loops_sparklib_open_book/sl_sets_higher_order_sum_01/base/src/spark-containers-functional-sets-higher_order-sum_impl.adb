--
--  Copyright (C) 2023-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;
separate (SPARK.Containers.Functional.Sets.Higher_Order)
function Sum_Impl
  (S     : Set;
   Value : not null access function (E : Element_Type) return Big_Integer)
   return Big_Integer
is
begin
   return Res : Big_Integer := 0 do
      for Subset in Iterate (S) loop
         Res := Res + Value (Choose (Subset));
      end loop;
   end return;
end Sum_Impl;
