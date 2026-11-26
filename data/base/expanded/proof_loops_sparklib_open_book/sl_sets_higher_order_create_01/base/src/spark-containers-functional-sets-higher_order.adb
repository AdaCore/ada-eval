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
     (New_Length : Big_Natural;
      New_Item   :
        not null access function (I : Big_Positive) return Element_Type)
      return Boolean
   is (for all I1 in Interval'(1, New_Length) =>
         (for all I2 in Interval'(1, New_Length) =>
            (if Equivalent_Elements (New_Item (I1), New_Item (I2))
             then I1 = I2)))
   with Ghost => Static;
   --  Check that New_Item always returns distinct elements

   -----------------
   -- Create_Impl --
   -----------------

   function Create_Impl
     (New_Length : Big_Natural;
      New_Item   :
        not null access function (I : Big_Positive) return Element_Type)
      return Set
   with
     Post =>
       (Static =>
          Length (Create_Impl'Result) <= New_Length
          and then
            (if All_Distinct (New_Length, New_Item)
             then Length (Create_Impl'Result) = New_Length)
          and then
            (for all I in Interval'(1, New_Length) =>
               Contains (Create_Impl'Result, New_Item (I)))
          and then
            (for all E of Create_Impl'Result =>
               (for some I in Interval'(1, New_Length) =>
                  Equivalent_Elements (E, New_Item (I)))));

   function Create_Impl
     (New_Length : Big_Natural;
      New_Item   :
        not null access function (I : Big_Positive) return Element_Type)
      return Set
   is separate;

   ------------
   -- Create --
   ------------

   function Create
     (New_Length : Big_Natural;
      New_Item   :
        not null access function (I : Big_Positive) return Element_Type)
      return Set
   with
     Refined_Post =>
       (Static =>
          Length (Create'Result) <= New_Length
          and then
            (if All_Distinct (New_Length, New_Item)
             then Length (Create'Result) = New_Length)
          and then
            (for all I in Interval'(1, New_Length) =>
               Contains (Create'Result, New_Item (I)))
          and then
            (for all E of Create'Result =>
               (for some I in Interval'(1, New_Length) =>
                  Equivalent_Elements (E, New_Item (I)))))
   is
   begin
      return Create_Impl (New_Length, New_Item);
   end Create;

end SPARK.Containers.Functional.Sets.Higher_Order;
