--
--  Copyright (C) 2023-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;

package body SPARK.Containers.Functional.Vectors.Higher_Order is

   -----------------
   -- Filter_Impl --
   -----------------

   function Filter_Impl
     (S    : Sequence;
      Last : Extended_Index;
      Test : not null access function (E : Element_Type) return Boolean)
      return Sequence
   with
     Pre  => (SPARKlib_Defensive => Last <= Vectors.Last (S)),
     Post =>
       (SPARKlib_Full =>
          Length (Filter_Impl'Result) = Length (Filter_Rec (S, Last, Test))
          and then
            Equal_Prefix (Filter_Impl'Result, Filter_Rec (S, Last, Test)));

   function Filter_Impl
     (S    : Sequence;
      Last : Extended_Index;
      Test : not null access function (E : Element_Type) return Boolean)
      return Sequence
   is separate;

   ------------
   -- Filter --
   ------------

   function Filter
     (S    : Sequence;
      Last : Extended_Index;
      Test : not null access function (E : Element_Type) return Boolean)
      return Sequence
   with
     Refined_Post =>
       (SPARKlib_Full =>
          Length (Filter'Result) = Length (Filter_Rec (S, Last, Test))
          and then Equal_Prefix (Filter'Result, Filter_Rec (S, Last, Test)))
   is
   begin
      return Filter_Impl (S, Last, Test);
   end Filter;

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

   ----------------
   -- Filter_Rec --
   ----------------

   function Filter_Rec
     (S    : Sequence;
      Last : Extended_Index;
      Test : not null access function (E : Element_Type) return Boolean)
      return Sequence
   is (if Last = Extended_Index'First
       then Empty_Sequence
       elsif Test (Get (S, Last))
       then
         Add (Filter_Rec (S, Extended_Index'Pred (Last), Test), Get (S, Last))
       else Filter_Rec (S, Extended_Index'Pred (Last), Test));

end SPARK.Containers.Functional.Vectors.Higher_Order;
