--
--  Copyright (C) 2004-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;
separate (Lists)
procedure Insert
  (Container : in out List;
   Before    : Cursor;
   New_Item  : Element_Type;
   Position  : out Cursor;
   Count     : Count_Type)
is
   J  : Count_Type;

begin
   if Count = 0 then
      Position := Before;
      return;
   end if;

   Allocate (Container, New_Item, New_Node => J);
   Insert_Internal (Container, Before.Node, New_Node => J);
   Position := (Node => J);

   for Index in 2 .. Count loop
      Allocate (Container, New_Item, New_Node => J);
      Insert_Internal (Container, Before.Node, New_Node => J);
   end loop;
end Insert;
