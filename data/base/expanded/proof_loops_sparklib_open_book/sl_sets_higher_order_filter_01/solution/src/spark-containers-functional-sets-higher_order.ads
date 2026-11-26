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

   function Eq_Compatible
     (S    : Set;
      Test : not null access function (E : Element_Type) return Boolean)
      return Boolean
   with
     Ghost    => SPARKlib_Full,
     Global   => null,
     Post     =>
       Eq_Compatible'Result
       = (for all E1 of S =>
            (for all E2 of S =>
               (if Equivalent_Elements (E1, E2) then Test (E1) = Test (E2)))),
     Annotate => (GNATprove, Higher_Order_Specialization),
     Annotate => (GNATprove, Inline_For_Proof);
   --  Test returns the same value on equivalent elements

   function Count
     (S    : Set;
      Test : not null access function (E : Element_Type) return Boolean)
      return Big_Natural
   with
     Import,
     Global   => null,
     Annotate => (GNATprove, Higher_Order_Specialization),
     Pre      => (SPARKlib_Full => Eq_Compatible (S, Test)),
     Post     => (SPARKlib_Full => Count'Result <= Length (S));
   --  Count the number of elements on which the input Test function returns
   --  True. Count can only be used with Test functions which return the same
   --  value on equivalent elements.

   procedure Lemma_Count_Eq
     (S1, S2 : Set;
      Test   : not null access function (E : Element_Type) return Boolean)
     --  Automatically instantiated lemma:
     --  Count returns the same value on sets containing the same elements.

   with
     Import,
     Ghost    => SPARKlib_Full,
     Global   => null,
     Annotate => (GNATprove, Higher_Order_Specialization),
     Annotate => (GNATprove, Automatic_Instantiation),
     Pre      => Eq_Compatible (S1, Test) and then S1 = S2,
     Post     => Count (S1, Test) = Count (S2, Test);

   procedure Lemma_Count_Remove
     (S    : Set;
      E    : Element_Type;
      Test : not null access function (E : Element_Type) return Boolean)
     --  Automatically instantiated lemma:
     --  Recursive definition of Count on for any element in S.

   with
     Import,
     Ghost    => SPARKlib_Full,
     Global   => null,
     Annotate => (GNATprove, Higher_Order_Specialization),
     Annotate => (GNATprove, Automatic_Instantiation),
     Pre      => Contains (S, E) and then Eq_Compatible (S, Test),
     Post     =>
       Count (S, Test)
       = Count (Remove (S, E), Test)
         + (if Test (E) then Big_Natural'(1) else 0);

   function Filter
     (S    : Set;
      Test : not null access function (E : Element_Type) return Boolean)
      return Set
   with
     Global   => null,
     Annotate => (GNATprove, Higher_Order_Specialization),
     Pre      => (SPARKlib_Full => Eq_Compatible (S, Test)),
     Post     =>
       (SPARKlib_Full =>
          Length (Filter'Result) = Count (S, Test)
          and then Filter'Result <= S
          and then (for all E of Filter'Result => Test (E))
          and then
            (for all E of S =>
               (if Test (E) then Contains (Filter'Result, E))));
   --  Return a new set with all elements of S on which the input Test function
   --  returns True. Filter can only be used with Test functions which return
   --  the same value on equivalent elements.

end SPARK.Containers.Functional.Sets.Higher_Order;
