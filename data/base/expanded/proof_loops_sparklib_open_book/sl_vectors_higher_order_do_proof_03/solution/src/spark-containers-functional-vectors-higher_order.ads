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
   --  because Lemma_Count_None's Post is stated in terms of Count_Rec once the
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

   procedure Lemma_Count_None
     (S    : Sequence;
      Last : Extended_Index;
      Test : not null access function (E : Element_Type) return Boolean)
   with
     Ghost    => SPARKlib_Full,
     Global   => null,
     Annotate => (GNATprove, Higher_Order_Specialization),
     Pre      =>
       Last <= Vectors.Last (S)
       and then
         (for all I in Index_Type'First .. Last => not Test (Get (S, I))),
     Post     => Count_Rec (S, Last, Test) = 0;
   --  Additional lemma:
   --  Count returns 0 if Test returns False on all elements of S up to Last.

end SPARK.Containers.Functional.Vectors.Higher_Order;
