--
--  Copyright (C) 2023-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;

package body SPARK.Containers.Functional.Maps.Higher_Order is

   -----------------------------------
   -- Local subprogram declarations --
   -----------------------------------

   function Sum_Rec
     (M     : Map;
      Value :
        not null access function
          (K : Key_Type; E : Element_Type) return Big_Integer)
      return Big_Integer
   with
     Ghost              => SPARKlib_Full,
     Subprogram_Variant => (Decreases => Length (M));
   --  Recursive version of Sum

   -------------------
   -- Eq_Compatible --
   -------------------

   function Eq_Compatible
     (M     : Map;
      Value :
        not null access function
          (K : Key_Type; E : Element_Type) return Big_Integer) return Boolean
   is (for all K1 of M =>
         (for all K2 of M =>
            (if Equivalent_Keys (K1, K2)
             then Value (K1, Get (M, K1)) = Value (K2, Get (M, K2)))));

   --------------
   -- Sum_Impl --
   --------------

   function Sum_Impl
     (M     : Map;
      Value :
        not null access function
          (K : Key_Type; E : Element_Type) return Big_Integer)
      return Big_Integer
   with
     Pre  => (SPARKlib_Full => Eq_Compatible (M, Value)),
     Post => (SPARKlib_Full => Sum_Impl'Result = Sum_Rec (M, Value));

   function Sum_Impl
     (M     : Map;
      Value :
        not null access function
          (K : Key_Type; E : Element_Type) return Big_Integer)
      return Big_Integer
   is separate;

   ---------
   -- Sum --
   ---------

   function Sum
     (M     : Map;
      Value :
        not null access function
          (K : Key_Type; E : Element_Type) return Big_Integer)
      return Big_Integer
   with
     Refined_Post => (SPARKlib_Full => Sum'Result = Sum_Rec (M, Value))
   is
   begin
      return Sum_Impl (M, Value);
   end Sum;

   -------------
   -- Sum_Rec --
   -------------

   function Sum_Rec
     (M     : Map;
      Value :
        not null access function
          (K : Key_Type; E : Element_Type) return Big_Integer)
      return Big_Integer
   is (if Is_Empty (M)
       then 0
       else
         Sum_Rec (Remove (M, Choose (M)), Value)
         + Value (Choose (M), Get (M, Choose (M))));

end SPARK.Containers.Functional.Maps.Higher_Order;
