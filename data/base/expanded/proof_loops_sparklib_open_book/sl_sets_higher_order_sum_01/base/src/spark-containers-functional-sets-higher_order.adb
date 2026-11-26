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

   function Sum_Rec
     (S     : Set;
      Value : not null access function (E : Element_Type) return Big_Integer)
      return Big_Integer
   with Subprogram_Variant => (Decreases => Length (S));
   --  Recursive version of Sum

   -------------------
   -- Eq_Compatible --
   -------------------

   function Eq_Compatible
     (S     : Set;
      Value : not null access function (E : Element_Type) return Big_Integer)
      return Boolean
   is (for all E1 of S =>
         (for all E2 of S =>
            (if Equivalent_Elements (E1, E2) then Value (E1) = Value (E2))));

   --------------
   -- Sum_Impl --
   --------------

   function Sum_Impl
     (S     : Set;
      Value : not null access function (E : Element_Type) return Big_Integer)
      return Big_Integer
   with
     Pre  => (SPARKlib_Full => Eq_Compatible (S, Value)),
     Post => (Static => Sum_Impl'Result = Sum_Rec (S, Value));

   function Sum_Impl
     (S     : Set;
      Value : not null access function (E : Element_Type) return Big_Integer)
      return Big_Integer
   is separate;

   ---------
   -- Sum --
   ---------

   function Sum
     (S     : Set;
      Value : not null access function (E : Element_Type) return Big_Integer)
      return Big_Integer
   with
     Refined_Post => (Static => Sum'Result = Sum_Rec (S, Value))
   is
   begin
      return Sum_Impl (S, Value);
   end Sum;

   -------------
   -- Sum_Rec --
   -------------

   function Sum_Rec
     (S     : Set;
      Value : not null access function (E : Element_Type) return Big_Integer)
      return Big_Integer
   is (if Is_Empty (S)
       then 0
       else Sum_Rec (Remove (S, Choose (S)), Value) + Value (Choose (S)));

end SPARK.Containers.Functional.Sets.Higher_Order;
