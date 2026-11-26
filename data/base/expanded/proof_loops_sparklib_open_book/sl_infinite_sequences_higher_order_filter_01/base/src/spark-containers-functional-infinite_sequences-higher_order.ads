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

   --  Count_Rec and Filter_Rec are declared in the package body upstream.  The
   --  isolation promotes them here, unchanged but for the Annotate line noted
   --  below, because Filter's visible Post is stated in terms of Count_Rec
   --  once the sibling Count is deleted.

   function Count_Rec
     (S    : Sequence;
      Last : Big_Natural;
      Test : not null access function (E : Element_Type) return Boolean)
      return Big_Natural
   with
     Ghost              => SPARKlib_Full,
     Annotate           => (GNATprove, Higher_Order_Specialization),
     --  Not upstream's.  Filter carries Higher_Order_Specialization, and a
     --  subprogram that does may pass an access-to-function parameter only to
     --  functions that also do; without this line Filter's Post is rejected
     --  with "shall only reference its access-to-function parameters in
     --  dereferences and as actual parameters in calls to functions annotated
     --  with Higher_Order_Specialization".  It is a specialization directive,
     --  not a contract, and grants the proof nothing.
     Subprogram_Variant => (Decreases => Last),
     Pre                => Last <= Length (S),
     Post               => Count_Rec'Result <= Last;
   --  Recursive version of Count

   function Filter_Rec
     (S    : Sequence;
      Last : Big_Natural;
      Test : not null access function (E : Element_Type) return Boolean)
      return Sequence
   with
     Ghost              => SPARKlib_Full,
     Subprogram_Variant => (Decreases => Last),
     Pre                => Last <= Length (S),
     Post               =>
       Length (Filter_Rec'Result) = Count_Rec (S, Last, Test)
       and then (for all E of Filter_Rec'Result => Test (E));
   --  Recursive version of Filter

   function Filter
     (S    : Sequence;
      Last : Big_Natural;
      Test : not null access function (E : Element_Type) return Boolean)
      return Sequence
   with
     Global   => null,
     Annotate => (GNATprove, Higher_Order_Specialization),
     Pre      => (SPARKlib_Defensive => Last <= Length (S)),
     Post     =>
       (SPARKlib_Full =>
          Length (Filter'Result) = Count_Rec (S, Last, Test)
          and then (for all E of Filter'Result => Test (E)));
   --  Return a new sequence with all elements of S on which the input Test
   --  function returns True.

end SPARK.Containers.Functional.Infinite_Sequences.Higher_Order;
