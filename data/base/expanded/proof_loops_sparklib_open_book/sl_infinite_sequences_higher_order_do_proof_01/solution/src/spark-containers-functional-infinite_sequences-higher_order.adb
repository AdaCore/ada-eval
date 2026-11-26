--
--  Copyright (C) 2023-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;

package body SPARK.Containers.Functional.Infinite_Sequences.Higher_Order is

   ---------------------
   -- Lemma_Count_All --
   ---------------------

   procedure Lemma_Count_All
     (S    : Sequence;
      Last : Big_Natural;
      Test : not null access function (E : Element_Type) return Boolean)
   is separate;

   ---------------
   -- Count_Rec --
   ---------------

   function Count_Rec
     (S    : Sequence;
      Last : Big_Natural;
      Test : not null access function (E : Element_Type) return Boolean)
      return Big_Natural
   is (if Last = 0
       then 0
       else
         Count_Rec (S, Last - 1, Test)
         + (if Test (Get (S, Last)) then 1 else Big_Natural'(0)));

end SPARK.Containers.Functional.Infinite_Sequences.Higher_Order;
