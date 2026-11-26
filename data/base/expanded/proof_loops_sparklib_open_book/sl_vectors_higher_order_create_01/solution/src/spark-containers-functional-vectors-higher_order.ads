--
--  Copyright (C) 2023-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;

generic
package SPARK.Containers.Functional.Vectors.Higher_Order with
    SPARK_Mode,
    Always_Terminates
is

   function Create
     (New_Last : Extended_Index;
      New_Item : not null access function (I : Index_Type) return Element_Type)
      return Sequence
   with
     Global   => null,
     Annotate => (GNATprove, Higher_Order_Specialization),
     Post     =>
       (SPARKlib_Full =>
          Last (Create'Result) = New_Last
          and then
            (for all I in Index_Type'First .. New_Last =>
               Element_Logic_Equal (Get (Create'Result, I), New_Item (I))));
   --  Return a new sequence with New_Length elements. Each element is created
   --  by calling New_Item.

end SPARK.Containers.Functional.Vectors.Higher_Order;
