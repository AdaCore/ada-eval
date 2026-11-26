--
--  Copyright (C) 2023-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;
separate (SPARK.Containers.Functional.Vectors.Higher_Order)
function Create
  (New_Last : Extended_Index;
   New_Item : not null access function (I : Index_Type) return Element_Type)
   return Sequence is
begin
   return Res : Sequence do
      for I in Index_Type'First .. New_Last loop
         Res := Add (Res, New_Item (I));
      end loop;
   end return;
end Create;
