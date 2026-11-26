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

   function Transform_Distinct
     (S              : Set;
      Transform_Item :
        not null access function (E : Element_Type) return Element_Type)
      return Set
   with
     Global   => null,
     Annotate => (GNATprove, Higher_Order_Specialization),
     Pre      =>
       (SPARKlib_Full =>
          (for all E1 of S =>
             (for all E2 of S =>
                Equivalent_Elements (Transform_Item (E1), Transform_Item (E2))
                = Equivalent_Elements (E1, E2)))),
     Post     =>
       (SPARKlib_Full =>
          Length (Transform_Distinct'Result) = Length (S)
          and then
            (for all E of S =>
               Contains (Transform_Distinct'Result, Transform_Item (E)))
          and then
            (for all E of Transform_Distinct'Result =>
               (for some F of S =>
                  Equivalent_Elements (E, Transform_Item (F)))));
   --  Return a new set containing elements obtained by applying Transform_Item
   --  to elements of S. Transform_Distinct can only be used with
   --  Transform_Item functions which never collapse elements. Use Transform
   --  otherwise.

end SPARK.Containers.Functional.Sets.Higher_Order;
