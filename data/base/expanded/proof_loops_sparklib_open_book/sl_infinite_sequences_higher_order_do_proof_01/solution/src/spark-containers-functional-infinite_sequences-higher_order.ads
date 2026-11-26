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

   --  Count_Rec is declared in the package body upstream.  The isolation
   --  promotes it here, unchanged, because Lemma_Count_All's Post is stated in
   --  terms of Count_Rec once the sibling Count is deleted.

   function Count_Rec
     (S    : Sequence;
      Last : Big_Natural;
      Test : not null access function (E : Element_Type) return Boolean)
      return Big_Natural
   with
     Ghost              => SPARKlib_Full,
     Annotate           => (GNATprove, Higher_Order_Specialization),
     --  Not upstream's.  The lemma below carries
     --  Higher_Order_Specialization, and a subprogram that does may pass an
     --  access-to-function parameter only to functions that also do; without
     --  this line the lemma's Post is rejected with "shall only reference its
     --  access-to-function parameters in dereferences and as actual
     --  parameters in calls to functions annotated with
     --  Higher_Order_Specialization".  It is a specialization directive, not a
     --  contract, and grants the proof nothing.
     Subprogram_Variant => (Decreases => Last),
     Pre                => Last <= Length (S),
     Post               => Count_Rec'Result <= Last;
   --  Recursive version of Count

   procedure Lemma_Count_All
     (S    : Sequence;
      Last : Big_Natural;
      Test : not null access function (E : Element_Type) return Boolean)
   with
     Ghost    => SPARKlib_Full,
     Global   => null,
     Annotate => (GNATprove, Higher_Order_Specialization),
     Pre      =>
       Last <= Length (S)
       and then (for all I in Interval'(1, Last) => Test (Get (S, I))),
     Post     => Count_Rec (S, Last, Test) = Last;
   --  Additional lemma:
   --  Count returns Last if Test returns True on all elements of S up to Last.

end SPARK.Containers.Functional.Infinite_Sequences.Higher_Order;
