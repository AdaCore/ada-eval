--
--  Copyright (C) 2023-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;
with SPARK.Big_Intervals; use SPARK.Big_Intervals;

package body SPARK.Containers.Functional.Infinite_Sequences.Higher_Order is

   function Count_Rec
     (S    : Sequence;
      Last : Big_Natural;
      Test : not null access function (E : Element_Type) return Boolean)
      return Big_Natural
   with
     Ghost              => SPARKlib_Full,
     Subprogram_Variant => (Decreases => Last),
     Pre                => Last <= Length (S),
     Post               => Count_Rec'Result <= Last;
   --  Recursive version of Count

   ----------------
   -- Count_Impl --
   ----------------

   function Count_Impl
     (S    : Sequence;
      Last : Big_Natural;
      Test : not null access function (E : Element_Type) return Boolean)
      return Big_Natural
   with
     Pre  => (SPARKlib_Defensive => Last <= Length (S)),
     Post =>
       (SPARKlib_Full => Count_Impl'Result = Count_Rec (S, Last, Test));

   function Count_Impl
     (S    : Sequence;
      Last : Big_Natural;
      Test : not null access function (E : Element_Type) return Boolean)
      return Big_Natural
   is separate;

   -----------
   -- Count --
   -----------

   function Count
     (S    : Sequence;
      Last : Big_Natural;
      Test : not null access function (E : Element_Type) return Boolean)
      return Big_Natural
   with
     Refined_Post =>
       (SPARKlib_Full => Count'Result = Count_Rec (S, Last, Test))
   is
   begin
      return Count_Impl (S, Last, Test);
   end Count;

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
