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
   L0 : constant Count_Type := Container.Length
   with Ghost => Static;

begin
   if Count = 0 then
      Position := Before;
      return;
   end if;

   Allocate (Container, New_Item, New_Node => J);
   Insert_Internal (Container, Before.Node, New_Node => J);
   Position := (Node => J);

   for Index in 2 .. Count loop
      pragma Loop_Invariant (Static => Structural_Invariant (Container));
      pragma Loop_Invariant (Static => Container.Length = L0 + (Index - 1));
      pragma
        Loop_Invariant
          (Static =>
             Container.Length + (Count - Index + 1) <= Container.Capacity);
      pragma
        Loop_Invariant
          (Static =>
             Before.Node = 0
             or else
               Memory_Index_Sets.Contains
                 (Active_Set (Container), Before.Node));

      Allocate (Container, New_Item, New_Node => J);
      Insert_Internal (Container, Before.Node, New_Node => J);
   end loop;
   pragma Assert (Static => Is_Relevant (Container, Before));
end Insert;
