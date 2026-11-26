--
--  Copyright (C) 2023-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;

package body SPARK.Containers.Functional.Vectors.Higher_Order is

   ----------------------
   -- Lemma_Count_None --
   ----------------------

   procedure Lemma_Count_None
     (S    : Sequence;
      Last : Extended_Index;
      Test : not null access function (E : Element_Type) return Boolean)
   is separate;

   ---------------
   -- Count_Rec --
   ---------------

   function Count_Rec
     (S    : Sequence;
      Last : Extended_Index;
      Test : not null access function (E : Element_Type) return Boolean)
      return Big_Natural
   is (if Last = Extended_Index'First
       then 0
       else
         Count_Rec (S, Extended_Index'Pred (Last), Test)
         + (if Test (Get (S, Last))
            then Big_Integer'(1)
            else Big_Integer'(0)));

end SPARK.Containers.Functional.Vectors.Higher_Order;
