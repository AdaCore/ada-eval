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
     Post     => (SPARKlib_Full => Count'Result <= Length (M));
   --  Count the number of mappings on which the input Test function returns
   --  True. Count can only be used with Test functions which return the same
   --  value on equivalent keys.

   procedure Lemma_Count_Eq
     (M1, M2 : Map;
      Test   :
        not null access function
          (K : Key_Type; E : Element_Type) return Boolean)
   with
     Import,
     Ghost    => SPARKlib_Full,
     Global   => null,
     Annotate => (GNATprove, Higher_Order_Specialization),
     Annotate => (GNATprove, Automatic_Instantiation),
     Pre      =>
       Eq_Compatible (M1, Test)
       and then Keys_Included (M2, M1)
       and then Elements_Equal (M1, M2),
     Post     => Count (M1, Test) = Count (M2, Test);
   --  Automatically instantiated lemma:
   --  Count returns the same value on maps containing the same mappings.

   procedure Lemma_Count_Remove
     (M    : Map;
      K    : Key_Type;
      Test :
        not null access function
          (K : Key_Type; E : Element_Type) return Boolean)
     --  Automatically instantiated lemma:
     --  Recursive definition of Count on for any mapping in M.

   with
     Import,
     Ghost    => SPARKlib_Full,
     Global   => null,
     Annotate => (GNATprove, Higher_Order_Specialization),
     Annotate => (GNATprove, Automatic_Instantiation),
     Pre      => Has_Key (M, K) and then Eq_Compatible (M, Test),
     Post     =>
       Count (M, Test)
       = Count (Remove (M, K), Test)
         + (if Test (K, Get (M, K)) then Big_Natural'(1) else 0);

   function Filter
     (M    : Map;
      Test :
        not null access function
          (K : Key_Type; E : Element_Type) return Boolean) return Map
   with
     Global   => null,
     Annotate => (GNATprove, Higher_Order_Specialization),
     Pre      => (SPARKlib_Full => Eq_Compatible (M, Test)),
     Post     =>
       (SPARKlib_Full =>
          Length (Filter'Result) = Count (M, Test)
          and then Elements_Equal (Filter'Result, M)
          and then (for all K of Filter'Result => Test (K, Get (M, K)))
          and then
            (for all K of M =>
               (if Test (K, Get (M, K)) then Has_Key (Filter'Result, K))));
   --  Return a new map with all mappings of M on which the input Test function
   --  returns True. Filter can only be used with Test functions which return
   --  the same value on equivalent keys.

end SPARK.Containers.Functional.Maps.Higher_Order;
