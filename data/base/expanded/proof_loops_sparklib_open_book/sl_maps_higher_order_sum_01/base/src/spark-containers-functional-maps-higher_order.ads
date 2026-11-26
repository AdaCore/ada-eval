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
     (M     : Map;
      Value :
        not null access function
          (K : Key_Type; E : Element_Type) return Big_Integer) return Boolean
   with
     Ghost    => SPARKlib_Full,
     Global   => null,
     Post     =>
       Eq_Compatible'Result
       = (for all K1 of M =>
            (for all K2 of M =>
               (if Equivalent_Keys (K1, K2)
                then Value (K1, Get (M, K1)) = Value (K2, Get (M, K2))))),
     Annotate => (GNATprove, Higher_Order_Specialization),
     Annotate => (GNATprove, Inline_For_Proof);
   --  Value returns the same value on equivalent keys

   function Sum
     (M     : Map;
      Value :
        not null access function
          (K : Key_Type; E : Element_Type) return Big_Integer)
      return Big_Integer
   with
     Global   => null,
     Annotate => (GNATprove, Higher_Order_Specialization),
     Pre      => (SPARKlib_Full => Eq_Compatible (M, Value));
   --  Sum the result of the Value function on all mappings of M. Sum can only
   --  be used with Value functions which return the same value on equivalent
   --  keys.

end SPARK.Containers.Functional.Maps.Higher_Order;
