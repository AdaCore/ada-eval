--
--  Copyright (C) 2023-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;
separate (SPARK.Containers.Functional.Maps.Higher_Order)
function Create
  (New_Length : Big_Natural;
   New_Key    : not null access function (I : Big_Positive) return Key_Type;
   New_Item   :
     not null access function (I : Big_Positive) return Element_Type)
   return Map
is
   I : Big_Positive := 1;
begin
   return Res : Map do
      while I <= New_Length loop
         Res := Add (Res, New_Key (I), New_Item (I));
         pragma Loop_Variant (Static => (Decreases => New_Length - I + 1));
         I := I + 1;
      end loop;
   end return;
end Create;
