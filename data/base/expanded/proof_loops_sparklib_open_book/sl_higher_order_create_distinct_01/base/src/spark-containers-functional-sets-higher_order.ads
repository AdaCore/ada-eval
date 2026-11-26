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

   function Create_Distinct
     (New_Length : Big_Natural;
      New_Item   :
        not null access function (I : Big_Positive) return Element_Type)
      return Set
   with
     Global   => null,
     Annotate => (GNATprove, Higher_Order_Specialization),
     Pre      =>
       (SPARKlib_Defensive =>
          (for all I1 in Interval'(1, New_Length) =>
             (for all I2 in Interval'(I1 + 1, New_Length) =>
                not Equivalent_Elements (New_Item (I1), New_Item (I2))))),
     Post     =>
       (SPARKlib_Full =>
          Length (Create_Distinct'Result) = New_Length
          and then
            (for all I in Interval'(1, New_Length) =>
               Contains (Create_Distinct'Result, New_Item (I)))
          and then
            (for all E of Create_Distinct'Result =>
               (for some I in Interval'(1, New_Length) =>
                  Equivalent_Elements (E, New_Item (I)))));
   --  Return a new set created by calling New_Item New_Length times.
   --  Create_Distinct can only be used on New_Item functions which create
   --  distinct elements for all integers up to New_Length. Use Create
   --  otherwise.

end SPARK.Containers.Functional.Sets.Higher_Order;
