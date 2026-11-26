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

   --  Count_Rec is declared in the package body upstream.  The isolation
   --  promotes it here, unchanged but for the Annotate line noted below,
   --  because Lemma_Count_Eq's Post is stated in terms of Count_Rec once the
   --  sibling Count is deleted.

   function Count_Rec
     (S    : Sequence;
      Last : Extended_Index;
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
     Pre                => Last <= Vectors.Last (S),
     Post               =>
       Count_Rec'Result <= Big (Last) - Big (Extended_Index'First);
   --  Recursive version of Count

   procedure Lemma_Count_Eq
     (S1, S2 : Sequence;
      Last   : Extended_Index;
      Test   : not null access function (E : Element_Type) return Boolean)
   with
     Ghost    => SPARKlib_Full,
     Global   => null,
     Annotate => (GNATprove, Higher_Order_Specialization),
     Pre      =>
       Last <= Vectors.Last (S1)
       and then Last <= Vectors.Last (S2)
       and then Range_Equal (S1, S2, Index_Type'First, Last),
     Post     => Count_Rec (S1, Last, Test) = Count_Rec (S2, Last, Test);
   --  Count returns the same value on sequences containing the same elements.

end SPARK.Containers.Functional.Vectors.Higher_Order;
