--
--  Copyright (C) 2023-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;
separate (SPARK.Containers.Functional.Vectors.Higher_Order)
function Filter_Impl
  (S    : Sequence;
   Last : Extended_Index;
   Test : not null access function (E : Element_Type) return Boolean)
   return Sequence
is
begin
   return Res : Sequence do
      for I in Index_Type'First .. Last loop
         if Test (Get (S, I)) then
            Res := Add (Res, Get (S, I));
         end if;
      end loop;
   end return;
end Filter_Impl;
