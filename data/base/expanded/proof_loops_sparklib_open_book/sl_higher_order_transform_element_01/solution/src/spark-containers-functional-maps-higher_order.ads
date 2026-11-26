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

   function Transform_Element
     (M              : Map;
      Transform_Item :
        not null access function (E : Element_Type) return Element_Type)
      return Map
   with
     Global   => null,
     Annotate => (GNATprove, Higher_Order_Specialization),
     Post     =>
       (SPARKlib_Full =>
          Length (Transform_Element'Result) = Length (M)
          and then Same_Keys (M, Transform_Element'Result)
          and then
            (for all K of M =>
               Element_Logic_Equal
                 (Get (Transform_Element'Result, K),
                  Transform_Item (Get (M, K)))));
   --  Return a new map containing a mapping per key in M. The new associations
   --  are obtained using Transform_Item on the associated elements.

end SPARK.Containers.Functional.Maps.Higher_Order;
