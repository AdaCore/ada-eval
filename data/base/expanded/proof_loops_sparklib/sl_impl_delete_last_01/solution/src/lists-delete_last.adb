--
--  Copyright (C) 2004-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;
separate (Lists)
procedure Delete_Last (Container : in out List; Count : Count_Type) is
   X : Count_Type;
begin
   if Count >= Container.Length then
      Clear (Container);
      return;
   end if;

   if Count = 0 then
      return;
   end if;

   --  Here 0 < Count < Length, so at least one node survives every
   --  iteration (Length stays >= 2 at each tail removal).

   for J in 1 .. Count loop
      pragma Loop_Invariant (Static => Structural_Invariant (Container));
      pragma Loop_Invariant (Static => Container.Length >= Count - J + 2);

      X := Container.Last;
      Unlink_Last_Node (Container);
      Container.Length := Container.Length - 1;
      Free (Container, X);
   end loop;
end Delete_Last;
