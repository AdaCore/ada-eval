--
--  Copyright (C) 2004-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

separate (Vectors)
procedure Reverse_Elements (Container : in out Vector) is
begin
   if Length (Container) <= 1 then
      return;
   end if;

   declare
      Len  : constant Capacity_Range := Length (Container);
      --  Captured in a constant: a renamed slice's bounds may not depend on
      --  a variable input in SPARK (E0007).
      I, J : Capacity_Range;
      E    : Elements_Array renames Container.Elements (1 .. Len);

   begin
      I := 1;
      J := Len;
      while I < J loop
         pragma Loop_Variant (Static => (Decreases => J - I));
         declare
            EI : constant Element_Type := E (I).V;

         begin
            E (I).V := E (J).V;
            E (J).V := EI;
         end;

         I := I + 1;
         J := J - 1;
      end loop;
   end;
end Reverse_Elements;
