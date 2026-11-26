--
--  Copyright (C) 2004-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;
separate (Lists)
procedure Clear (Container : in out List) is
   X : Count_Type;

begin
   if Container.Length = 0 then
      return;
   end if;

   while Container.Length > 1 loop
      pragma Loop_Invariant (Static => Structural_Invariant (Container));
      pragma Loop_Variant (Static => (Decreases => Container.Length));

      X := Container.First;
      Unlink_First_Node (Container);
      Container.Length := Container.Length - 1;
      Free (Container, X);
   end loop;

   --  Exactly one active node is left; drop it

   X := Container.First;
   Unlink_Sole_Node (Container);
   Container.Length := 0;
   Free (Container, X);
end Clear;
