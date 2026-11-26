--
--  Copyright (C) 2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;
separate (SPARK.Containers.Functional.Total_Maps)
function All_Default (Container : Map) return Boolean is
begin
   for S in Maps.Iterate (Container.Content) loop
      pragma
        Loop_Invariant
          (Static =>
             (for all K of Container.Content =>
                Maps.Has_Key (S, K)
                or else
                  Element_Logic_Equal
                    (Get (Container, K), Copy_Element (Default_Element))));
      declare
         K : constant Key_Type := Maps.Choose (S);
      begin
         if not Element_Logic_Equal
                  (Get (Container, K), Copy_Element (Default_Element))
         then
            return False;
         end if;
      end;
   end loop;
   return True;
end All_Default;
