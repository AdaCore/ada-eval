--
--  Copyright (C) 2023-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;
with SPARK.Big_Intervals; use SPARK.Big_Intervals;

generic
package SPARK.Containers.Functional.Maps.Higher_Order with
    SPARK_Mode,
    Always_Terminates
is
   use SPARK.Big_Integers;

   function Create
     (New_Length : Big_Natural;
      New_Key    : not null access function (I : Big_Positive) return Key_Type;
      New_Item   :
        not null access function (I : Big_Positive) return Element_Type)
      return Map
   with
     Global   => null,
     Annotate => (GNATprove, Higher_Order_Specialization),
     Pre      =>
       (SPARKlib_Defensive =>
          (for all I1 in Interval'(1, New_Length) =>
             (for all I2 in Interval'(I1 + 1, New_Length) =>
                not Equivalent_Keys (New_Key (I1), New_Key (I2))))),
     Post     =>
       (SPARKlib_Full =>
          Length (Create'Result) = New_Length
          and then
            (for all I in Interval'(1, New_Length) =>
               Has_Key (Create'Result, New_Key (I)))
          and then
            (for all K of Create'Result =>
               (for some I in Interval'(1, New_Length) =>
                  Equivalent_Keys (K, New_Key (I))))
          and then
            (for all I in Interval'(1, New_Length) =>
               Element_Logic_Equal
                 (Get (Create'Result, New_Key (I)), New_Item (I))));
   --  Return a new map with New_Length associations. Each association is
   --  created by calling New_Key and New_Item.

end SPARK.Containers.Functional.Maps.Higher_Order;
