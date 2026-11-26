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

   function Sum
     (S     : Sequence;
      Last  : Extended_Index;
      Value : not null access function (E : Element_Type) return Big_Integer)
      return Big_Integer
   with
     Global   => null,
     Annotate => (GNATprove, Higher_Order_Specialization),
     Pre      => (SPARKlib_Defensive => Last <= Vectors.Last (S)),
     Post     =>
       (SPARKlib_Full => (if Last = Extended_Index'First then Sum'Result = 0));
   --  Sum the result of the Value function on all elements of S up to Last

end SPARK.Containers.Functional.Vectors.Higher_Order;
