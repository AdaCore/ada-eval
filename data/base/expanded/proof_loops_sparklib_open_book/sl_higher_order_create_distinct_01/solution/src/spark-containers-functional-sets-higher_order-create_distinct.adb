--
--  Copyright (C) 2023-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;
separate (SPARK.Containers.Functional.Sets.Higher_Order)
function Create_Distinct
  (New_Length : Big_Natural;
   New_Item   :
     not null access function (I : Big_Positive) return Element_Type)
   return Set
is
   I : Big_Positive := 1;
begin
   return Res : Set do
      while I <= New_Length loop
         Res := Add (Res, New_Item (I));

         pragma Loop_Variant (Decreases => New_Length - I + 1);
         pragma Loop_Invariant (Static => I <= New_Length);
         pragma Loop_Invariant (Static => Length (Res) = I);
         pragma
           Loop_Invariant
             (Static =>
                (for all J in Interval'(1, I) =>
                   Contains (Res, New_Item (J))));
         pragma
           Loop_Invariant
             (Static =>
                (for all E of Res =>
                   (for some J in Interval'(1, I) =>
                      Equivalent_Elements (E, New_Item (J)))));
         I := I + 1;
      end loop;
   end return;
end Create_Distinct;
