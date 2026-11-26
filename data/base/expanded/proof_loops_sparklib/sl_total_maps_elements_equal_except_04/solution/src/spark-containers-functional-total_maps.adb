--
--  Copyright (C) 2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;

package body SPARK.Containers.Functional.Total_Maps is

   ---------------------------
   -- Elements_Equal_Except --
   ---------------------------

   function Elements_Equal_Except
     (Left : Map; Right : Map; X : Key_Type; Y : Key_Type) return Boolean
     is separate;

   -------------------------
   -- Element_Logic_Equal --
   -------------------------

   function Element_Logic_Equal (Left, Right : Element_Type) return Boolean is
   begin
      return Left = Right;
   end Element_Logic_Equal;

   --------------------------
   -- Lemma_Get_Equivalent --
   --------------------------

   procedure Lemma_Get_Equivalent (Container : Map; Key_1, Key_2 : Key_Type)
   is null;

end SPARK.Containers.Functional.Total_Maps;
