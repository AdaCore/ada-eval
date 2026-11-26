--
--  Copyright (C) 2023-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;
with SPARK.Big_Intervals; use SPARK.Big_Intervals;

generic
package SPARK.Containers.Functional.Infinite_Sequences.Higher_Order with
    SPARK_Mode,
    Always_Terminates
is

   function Filter_Rec
     (S    : Sequence;
      Last : Big_Natural;
      Test : not null access function (E : Element_Type) return Boolean)
      return Sequence
   is (if Last = 0
       then Empty_Sequence
       elsif Test (Get (S, Last))
       then Add (Filter_Rec (S, Last - 1, Test), Get (S, Last))
       else Filter_Rec (S, Last - 1, Test))
   with
     Ghost              => SPARKlib_Full,
     Annotate           => (GNATprove, Higher_Order_Specialization),
     Subprogram_Variant => (Decreases => Last),
     Pre                => Last <= Length (S);
   --  Recursive version of Filter

   function Filter
     (S    : Sequence;
      Last : Big_Natural;
      Test : not null access function (E : Element_Type) return Boolean)
      return Sequence
   with
     Global   => null,
     Import,
     Annotate => (GNATprove, Higher_Order_Specialization),
     Pre      => (SPARKlib_Defensive => Last <= Length (S)),
     Post     =>
       (SPARKlib_Full =>
          Length (Filter'Result) = Length (Filter_Rec (S, Last, Test))
          and then Equal_Prefix (Filter'Result, Filter_Rec (S, Last, Test)));
   --  Return a new sequence with all elements of S on which the input Test
   --  function returns True.

   procedure Lemma_Filter_Eq
     (S1, S2 : Sequence;
      Last   : Big_Natural;
      Test   : not null access function (E : Element_Type) return Boolean)
   with
     Ghost    => SPARKlib_Full,
     Global   => null,
     Annotate => (GNATprove, Higher_Order_Specialization),
     Annotate => (GNATprove, Automatic_Instantiation),
     Pre      =>
       Last <= Length (S1)
       and then Last <= Length (S2)
       and then
         (for all I in Interval'(1, Last) =>
            Element_Logic_Equal (Get (S1, I), Get (S2, I))),
     Post     =>
       Length (Filter (S1, Last, Test)) = Length (Filter (S2, Last, Test))
       and then
         Equal_Prefix (Filter (S1, Last, Test), Filter (S2, Last, Test));
   --  Automatically instantiated lemma:
   --  Filter returns the same value on sequences containing the same elements.

end SPARK.Containers.Functional.Infinite_Sequences.Higher_Order;
