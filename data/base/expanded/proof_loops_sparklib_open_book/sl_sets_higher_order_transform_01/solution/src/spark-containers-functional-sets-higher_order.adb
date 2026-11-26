--
--  Copyright (C) 2023-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;

with SPARK.Big_Integers; use SPARK.Big_Integers;

package body SPARK.Containers.Functional.Sets.Higher_Order is

   -----------------------------------
   -- Local subprogram declarations --
   -----------------------------------

   function All_Distinct
     (S              : Set;
      Transform_Item :
        not null access function (I : Element_Type) return Element_Type)
      return Boolean
   is (for all E1 of S =>
         (for all E2 of S =>
            (if Equivalent_Elements (Transform_Item (E1), Transform_Item (E2))
             then Equivalent_Elements (E1, E2))))
   with Ghost => Static;
   --  Check that Transform_Item always returns distinct elements

   --------------------
   -- Transform_Impl --
   --------------------

   function Transform_Impl
     (S              : Set;
      Transform_Item :
        not null access function (E : Element_Type) return Element_Type)
      return Set
   with
     Pre  =>
       (SPARKlib_Full =>
          (for all E1 of S =>
             (for all E2 of S =>
                (if Equivalent_Elements (E1, E2)
                 then
                   Equivalent_Elements
                     (Transform_Item (E1), Transform_Item (E2)))))),
     Post =>
       (Static =>
          Length (Transform_Impl'Result) <= Length (S)
          and then
            (if All_Distinct (S, Transform_Item)
             then Length (Transform_Impl'Result) = Length (S))
          and then
            (for all E of S =>
               Contains (Transform_Impl'Result, Transform_Item (E)))
          and then
            (for all E of Transform_Impl'Result =>
               (for some F of S =>
                  Equivalent_Elements (E, Transform_Item (F)))));

   function Transform_Impl
     (S              : Set;
      Transform_Item :
        not null access function (E : Element_Type) return Element_Type)
      return Set
   is separate;

   ---------------
   -- Transform --
   ---------------

   function Transform
     (S              : Set;
      Transform_Item :
        not null access function (E : Element_Type) return Element_Type)
      return Set
   with
     Refined_Post =>
       (Static =>
          Length (Transform'Result) <= Length (S)
          and then
            (if All_Distinct (S, Transform_Item)
             then Length (Transform'Result) = Length (S))
          and then
            (for all E of S => Contains (Transform'Result, Transform_Item (E)))
          and then
            (for all E of Transform'Result =>
               (for some F of S =>
                  Equivalent_Elements (E, Transform_Item (F)))))
   is
   begin
      return Transform_Impl (S, Transform_Item);
   end Transform;

end SPARK.Containers.Functional.Sets.Higher_Order;
