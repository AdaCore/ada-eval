--
--  Copyright (C) 2023-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;

with SPARK.Big_Integers; use SPARK.Big_Integers;

package body SPARK.Containers.Functional.Sets.Higher_Order is

   -----------------------------------
   -- Local subprogram declarations --
   -----------------------------------

   function Count_Rec
     (S    : Set;
      Test : not null access function (E : Element_Type) return Boolean)
      return Big_Natural
   with
     Ghost              => Static,
     Subprogram_Variant => (Decreases => Length (S)),
     Post               => Count_Rec'Result <= Length (S);
   --  Recursive version of Count

   ----------------
   -- Count_Impl --
   ----------------

   function Count_Impl
     (S    : Set;
      Test : not null access function (E : Element_Type) return Boolean)
      return Big_Natural
   with
     Pre  => (SPARKlib_Full => Eq_Compatible (S, Test)),
     Post => (Static => Count_Impl'Result = Count_Rec (S, Test));

   function Count_Impl
     (S    : Set;
      Test : not null access function (E : Element_Type) return Boolean)
      return Big_Natural
   is separate;

   -----------
   -- Count --
   -----------

   function Count
     (S    : Set;
      Test : not null access function (E : Element_Type) return Boolean)
      return Big_Natural
   with
     Refined_Post => (Static => Count'Result = Count_Rec (S, Test))
   is
   begin
      return Count_Impl (S, Test);
   end Count;

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

   -------------------
   -- Eq_Compatible --
   -------------------

   function Eq_Compatible
     (S    : Set;
      Test : not null access function (E : Element_Type) return Boolean)
      return Boolean
   is (for all E1 of S =>
         (for all E2 of S =>
            (if Equivalent_Elements (E1, E2) then Test (E1) = Test (E2))));

end SPARK.Containers.Functional.Sets.Higher_Order;
