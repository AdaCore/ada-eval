--
--  Copyright (C) 2023-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;
with SPARK.Big_Integers; use SPARK.Big_Integers;

generic
package SPARK.Containers.Functional.Vectors.Higher_Order with
    SPARK_Mode,
    Always_Terminates
is

   function Count_Rec
     (S    : Sequence;
      Last : Extended_Index;
      Test : not null access function (E : Element_Type) return Boolean)
      return Big_Natural
   with
     Ghost              => SPARKlib_Full,
     Subprogram_Variant => (Decreases => Last),
     Pre                => Last <= Vectors.Last (S),
     Post               =>
       Count_Rec'Result <= Big (Last) - Big (Extended_Index'First);
   --  Recursive version of Count

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

   function Filter_Rec
     (S    : Sequence;
      Last : Extended_Index;
      Test : not null access function (E : Element_Type) return Boolean)
      return Sequence
   with
     Ghost              => SPARKlib_Full,
     Subprogram_Variant => (Decreases => Last),
     Pre                => Last <= Vectors.Last (S),
     Post               =>
       Length (Filter_Rec'Result) = Count_Rec (S, Last, Test)
       and then (for all E of Filter_Rec'Result => Test (E));
   --  Recursive version of Filter

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

   function Filter
     (S    : Sequence;
      Last : Extended_Index;
      Test : not null access function (E : Element_Type) return Boolean)
      return Sequence
   with
     Import,
     Global => null,
     Pre    => (SPARKlib_Defensive => Last <= Vectors.Last (S)),
     Post   =>
       (SPARKlib_Full =>
          Length (Filter'Result) = Length (Filter_Rec (S, Last, Test))
          and then Equal_Prefix (Filter'Result, Filter_Rec (S, Last, Test)));
   --  Return a new sequence with all elements of S on which the input Test
   --  function returns True.

   procedure Lemma_Filter_All
     (S    : Sequence;
      Last : Extended_Index;
      Test : not null access function (E : Element_Type) return Boolean)
     --  Additional lemma:
     --  Filter returns the prefix of S up to Last if Test returns True on all
     --  of its elements.

   with
     Ghost    => SPARKlib_Full,
     Global   => null,
     Pre      =>
       Last <= Vectors.Last (S)
       and then (for all I in Index_Type'First .. Last => Test (Get (S, I))),
     Post     =>
       Vectors.Last (Filter (S, Last, Test)) = Last
       and then Equal_Prefix (Filter (S, Last, Test), S);

end SPARK.Containers.Functional.Vectors.Higher_Order;
