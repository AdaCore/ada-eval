--
--  Copyright (C) 2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;
separate (SPARK.Containers.Functional.Total_Maps)
function Equivalent_Maps (Left : Map; Right : Map) return Boolean is
begin
   for S in Maps.Iterate (Left.Content) loop
      pragma
        Loop_Invariant
          (Static =>
             (for all K of Left.Content =>
                Maps.Has_Key (S, K)
                or else
                  Equivalent_Elements (Get (Left, K), Get (Right, K))));
      declare
         K : constant Key_Type := Maps.Choose (S);
      begin
         if not Equivalent_Elements (Get (Left, K), Get (Right, K)) then
            return False;
         end if;
      end;
   end loop;

   return True;
end Equivalent_Maps;
