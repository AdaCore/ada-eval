--
--  Copyright (C) 2023-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;
separate (SPARK.Containers.Functional.Infinite_Sequences.Higher_Order)
function Create
  (New_Length : Big_Natural;
   New_Item   :
     not null access function (I : Big_Positive) return Element_Type)
   return Sequence is
begin
   return Res : Sequence do
      for I in Interval'(1, New_Length) loop
         Res := Add (Res, New_Item (I));
      end loop;
   end return;
end Create;
