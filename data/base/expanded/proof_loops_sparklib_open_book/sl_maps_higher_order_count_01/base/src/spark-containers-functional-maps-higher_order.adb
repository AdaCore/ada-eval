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

   function Count_Rec
     (M    : Map;
      Test :
        not null access function
          (K : Key_Type; E : Element_Type) return Boolean) return Big_Natural
   with
     Ghost              => SPARKlib_Full,
     Subprogram_Variant => (Decreases => Length (M)),
     Post               => Count_Rec'Result <= Length (M);
   --  Recursive version of Count

   ----------------
   -- Count_Impl --
   ----------------

   function Count_Impl
     (M    : Map;
      Test :
        not null access function
          (K : Key_Type; E : Element_Type) return Boolean) return Big_Natural
   with
     Pre  => (SPARKlib_Full => Eq_Compatible (M, Test)),
     Post => (SPARKlib_Full => Count_Impl'Result = Count_Rec (M, Test));

   function Count_Impl
     (M    : Map;
      Test :
        not null access function
          (K : Key_Type; E : Element_Type) return Boolean) return Big_Natural
   is separate;

   -----------
   -- Count --
   -----------

   function Count
     (M    : Map;
      Test :
        not null access function
          (K : Key_Type; E : Element_Type) return Boolean) return Big_Natural
   with
     Refined_Post => (SPARKlib_Full => Count'Result = Count_Rec (M, Test))
   is
   begin
      return Count_Impl (M, Test);
   end Count;

   ---------------
   -- Count_Rec --
   ---------------

   function Count_Rec
     (M    : Map;
      Test :
        not null access function
          (K : Key_Type; E : Element_Type) return Boolean) return Big_Natural
   is (if Is_Empty (M)
       then 0
       else
         Count_Rec (Remove (M, Choose (M)), Test)
         + (if Test (Choose (M), Get (M, Choose (M)))
            then Big_Natural'(1)
            else 0));

   -------------------
   -- Eq_Compatible --
   -------------------

   function Eq_Compatible
     (M    : Map;
      Test :
        not null access function
          (K : Key_Type; E : Element_Type) return Boolean) return Boolean
   is (for all K1 of M =>
         (for all K2 of M =>
            (if Equivalent_Keys (K1, K2)
             then Test (K1, Get (M, K1)) = Test (K2, Get (M, K2)))));

end SPARK.Containers.Functional.Maps.Higher_Order;
