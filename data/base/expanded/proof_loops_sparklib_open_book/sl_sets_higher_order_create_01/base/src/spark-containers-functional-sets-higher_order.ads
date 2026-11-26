--
--  Copyright (C) 2023-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;

with SPARK.Big_Intervals; use SPARK.Big_Intervals;

generic
package SPARK.Containers.Functional.Sets.Higher_Order with
    SPARK_Mode,
    Always_Terminates
is
   use SPARK.Big_Integers;

   function Create
     (New_Length : Big_Natural;
      New_Item   :
        not null access function (I : Big_Positive) return Element_Type)
      return Set
   with
     Global   => null,
     Annotate => (GNATprove, Higher_Order_Specialization),
     Post     =>
       (SPARKlib_Full =>
          Length (Create'Result) <= New_Length
          and then
            (for all I in Interval'(1, New_Length) =>
               Contains (Create'Result, New_Item (I)))
          and then
            (for all E of Create'Result =>
               (for some I in Interval'(1, New_Length) =>
                  Equivalent_Elements (E, New_Item (I)))));
   --  Return a new set created by calling New_Item New_Length times. Create
   --  is less efficient than Create_Distinct as it needs to check for
   --  duplicated values before inclusion.

end SPARK.Containers.Functional.Sets.Higher_Order;
