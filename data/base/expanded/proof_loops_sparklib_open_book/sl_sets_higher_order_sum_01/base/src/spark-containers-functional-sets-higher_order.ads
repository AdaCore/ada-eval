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
     (S     : Set;
      Value : not null access function (E : Element_Type) return Big_Integer)
      return Boolean
   with
     Ghost    => SPARKlib_Full,
     Global   => null,
     Post     =>
       Eq_Compatible'Result
       = (for all E1 of S =>
            (for all E2 of S =>
               (if Equivalent_Elements (E1, E2)
                then Value (E1) = Value (E2)))),
     Annotate => (GNATprove, Higher_Order_Specialization),
     Annotate => (GNATprove, Inline_For_Proof);
   --  Value returns the same value on equivalent keys

   function Sum
     (S     : Set;
      Value : not null access function (E : Element_Type) return Big_Integer)
      return Big_Integer
   with
     Global   => null,
     Annotate => (GNATprove, Higher_Order_Specialization),
     Pre      => (SPARKlib_Full => Eq_Compatible (S, Value));
   --  Sum the result of the Value function on all elements of S. Sum can only
   --  be used with Value functions which return the same value on equivalent
   --  elements.

end SPARK.Containers.Functional.Sets.Higher_Order;
