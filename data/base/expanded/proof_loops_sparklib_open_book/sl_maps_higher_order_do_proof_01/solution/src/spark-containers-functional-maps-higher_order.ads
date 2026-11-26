--
--  Copyright (C) 2023-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;

generic
package SPARK.Containers.Functional.Maps.Higher_Order with
    SPARK_Mode,
    Always_Terminates
is
   use SPARK.Big_Integers;

   function Eq_Compatible
     (M    : Map;
      Test :
        not null access function
          (K : Key_Type; E : Element_Type) return Boolean) return Boolean
   with
     Ghost    => SPARKlib_Full,
     Global   => null,
     Post     =>
       Eq_Compatible'Result
       = (for all K1 of M =>
            (for all K2 of M =>
               (if Equivalent_Keys (K1, K2)
                then Test (K1, Get (M, K1)) = Test (K2, Get (M, K2))))),
     Annotate => (GNATprove, Higher_Order_Specialization),
     Annotate => (GNATprove, Inline_For_Proof);
   --  Test returns the same value on equivalent keys

   function Count_Rec
     (M    : Map;
      Test :
        not null access function
          (K : Key_Type; E : Element_Type) return Boolean) return Big_Natural
   with
     Ghost              => SPARKlib_Full,
     Annotate           => (GNATprove, Higher_Order_Specialization),
     Subprogram_Variant => (Decreases => Length (M)),
     Post               => Count_Rec'Result <= Length (M);
   --  Recursive version of Count

   function Count
     (M    : Map;
      Test :
        not null access function
          (K : Key_Type; E : Element_Type) return Boolean) return Big_Natural
   with
     Import,
     Global   => null,
     Annotate => (GNATprove, Higher_Order_Specialization),
     Pre      => (SPARKlib_Full => Eq_Compatible (M, Test)),
     Post     =>
       (SPARKlib_Full =>
          Count'Result <= Length (M)
          and then Count'Result = Count_Rec (M, Test));
   --  Count the number of mappings on which the input Test function returns
   --  True. Count can only be used with Test functions which return the same
   --  value on equivalent keys.

   procedure Lemma_Count_All
     (M    : Map;
      Test :
        not null access function
          (K : Key_Type; E : Element_Type) return Boolean)
     --  Additional lemma:
     --  Count returns Length (M) if Test returns True on all elements of M.

   with
     Ghost    => SPARKlib_Full,
     Global   => null,
     Annotate => (GNATprove, Higher_Order_Specialization),
     Pre      => (for all K of M => Test (K, Get (M, K))),
     Post     => Count (M, Test) = Length (M);

end SPARK.Containers.Functional.Maps.Higher_Order;
