--
--  Copyright (C) 2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;
separate (SPARK.Containers.Functional.Total_Maps)
function Elements_Equal_Except
  (Left : Map; Right : Map; X : Key_Type; Y : Key_Type) return Boolean is
begin
   for S in Maps.Iterate (Left.Content) loop
      declare
         K : constant Key_Type := Maps.Choose (S);
      begin
         if not Equivalent_Keys (K, X)
           and then not Equivalent_Keys (K, Y)
           and then not Element_Logic_Equal (Get (Left, K), Get (Right, K))
         then
            return False;
         end if;
      end;
   end loop;

   return True;
end Elements_Equal_Except;
