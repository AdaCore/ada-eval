--
--  Copyright (C) 2023-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;
with SPARK.Big_Intervals; use SPARK.Big_Intervals;

generic
package SPARK.Containers.Functional.Infinite_Sequences.Higher_Order with
    SPARK_Mode,
    Always_Terminates
is

   function Create
     (New_Length : Big_Natural;
      New_Item   :
        not null access function (I : Big_Positive) return Element_Type)
      return Sequence
   with
     Global   => null,
     Annotate => (GNATprove, Higher_Order_Specialization),
     Post     =>
       (SPARKlib_Full =>
          Length (Create'Result) = New_Length
          and then
            (for all I in Interval'(1, New_Length) =>
               Element_Logic_Equal (Get (Create'Result, I), New_Item (I))));
   --  Return a new sequence with New_Length elements. Each element is created
   --  by calling New_Item.

end SPARK.Containers.Functional.Infinite_Sequences.Higher_Order;
