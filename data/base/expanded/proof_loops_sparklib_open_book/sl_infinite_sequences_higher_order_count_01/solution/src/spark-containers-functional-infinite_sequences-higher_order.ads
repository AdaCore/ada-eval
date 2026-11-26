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

   function Count
     (S    : Sequence;
      Last : Big_Natural;
      Test : not null access function (E : Element_Type) return Boolean)
      return Big_Natural
   with
     Global   => null,
     Annotate => (GNATprove, Higher_Order_Specialization),
     Pre      => (SPARKlib_Defensive => Last <= Length (S)),
     Post     => (SPARKlib_Full => Count'Result <= Last);
   --  Count the number of elements on which the input Test function returns
   --  True.

end SPARK.Containers.Functional.Infinite_Sequences.Higher_Order;
