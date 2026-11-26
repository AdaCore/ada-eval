--
--  Copyright (C) 2023-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;

generic
package SPARK.Containers.Functional.Sets.Higher_Order with
    SPARK_Mode,
    Always_Terminates
is
   use SPARK.Big_Integers;

   --  Count_Rec is declared in the package body upstream.  The isolation
   --  promotes it here, unchanged but for the Annotate line noted below,
   --  because Lemma_Count_All's Post is stated in terms of Count_Rec once the
   --  sibling Count is deleted.

   function Count_Rec
     (S    : Set;
      Test : not null access function (E : Element_Type) return Boolean)
      return Big_Natural
   with
     Ghost              => Static,
     Annotate           => (GNATprove, Higher_Order_Specialization),
     --  Not upstream's.  The lemma below carries Higher_Order_Specialization,
     --  and a subprogram that does may pass an access-to-function parameter
     --  only to functions that also do; without this line the lemma's Post is
     --  rejected with "shall only reference its access-to-function parameters
     --  in dereferences and as actual parameters in calls to functions
     --  annotated with Higher_Order_Specialization".  It is a specialization
     --  directive, not a contract, and grants the proof nothing.
     Subprogram_Variant => (Decreases => Length (S)),
     Post               => Count_Rec'Result <= Length (S);
   --  Recursive version of Count

   procedure Lemma_Count_All
     (S    : Set;
      Test : not null access function (E : Element_Type) return Boolean)
     --  Additional lemma:
     --  Count returns Length (S) if Test returns True on all elements of S.

   with
     Ghost    => SPARKlib_Full,
     Global   => null,
     Annotate => (GNATprove, Higher_Order_Specialization),
     Pre      => (for all E of S => Test (E)),
     Post     => Count_Rec (S, Test) = Length (S);

end SPARK.Containers.Functional.Sets.Higher_Order;
