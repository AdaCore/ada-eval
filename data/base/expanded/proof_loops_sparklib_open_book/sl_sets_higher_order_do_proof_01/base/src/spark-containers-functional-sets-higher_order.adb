--
--  Copyright (C) 2023-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;

with SPARK.Big_Integers; use SPARK.Big_Integers;

package body SPARK.Containers.Functional.Sets.Higher_Order is

   ---------------
   -- Count_Rec --
   ---------------

   function Count_Rec
     (S    : Set;
      Test : not null access function (E : Element_Type) return Boolean)
      return Big_Natural
   is (if Is_Empty (S)
       then 0
       else
         Count_Rec (Remove (S, Choose (S)), Test)
         + (if Test (Choose (S)) then Big_Natural'(1) else 0));

   ---------------------
   -- Lemma_Count_All --
   ---------------------

   procedure Lemma_Count_All
     (S    : Set;
      Test : not null access function (E : Element_Type) return Boolean)
   is separate;

end SPARK.Containers.Functional.Sets.Higher_Order;
