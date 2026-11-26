--
--  Copyright (C) 2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;
separate (SPARK.Containers.Functional.Total_Maps)
function Equivalent_Maps (Left : Map; Right : Map) return Boolean is
begin
   for S in Maps.Iterate (Right.Content) loop
      declare
         K : constant Key_Type := Maps.Choose (S);
      begin
         if not Maps.Has_Key (Left.Content, K)
           and then not Equivalent_Elements (Get (Left, K), Get (Right, K))
         then
            return False;
         end if;
      end;
   end loop;

   return True;
end Equivalent_Maps;
