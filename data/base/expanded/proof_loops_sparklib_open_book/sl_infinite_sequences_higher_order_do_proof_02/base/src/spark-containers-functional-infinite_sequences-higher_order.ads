--
--  Copyright (C) 2023-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;

generic
package SPARK.Containers.Functional.Infinite_Sequences.Higher_Order with
    SPARK_Mode,
    Always_Terminates
is

   --  Count_Rec is declared in the package body upstream.  The isolation
   --  promotes it here, unchanged, because Lemma_Count_Eq's Post is stated in
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

   procedure Lemma_Count_Eq
     (S1, S2 : Sequence;
      Last   : Big_Natural;
      Test   : not null access function (E : Element_Type) return Boolean)
   with
     Ghost    => SPARKlib_Full,
     Global   => null,
     Annotate => (GNATprove, Higher_Order_Specialization),
     Pre      =>
       Last <= Length (S1)
       and then Last <= Length (S2)
       and then Range_Equal (S1, S2, 1, Last),
     Post     => Count_Rec (S1, Last, Test) = Count_Rec (S2, Last, Test);
   --  Count returns the same value on sequences containing the same elements.

end SPARK.Containers.Functional.Infinite_Sequences.Higher_Order;
