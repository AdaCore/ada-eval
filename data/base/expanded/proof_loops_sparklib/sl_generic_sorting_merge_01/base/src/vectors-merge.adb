--
--  Copyright (C) 2004-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

separate (Vectors)
procedure Merge (Target : in out Vector; Source : in out Vector) is
   I : Count_Type;
   J : Count_Type;

begin
   I := Length (Target);
   declare
      Current_Last : Extended_Index := Target.Last;
      TA           : Elements_Array renames Target.Elements;
      SA           : Elements_Array renames Source.Elements;

   begin
      Insert_Space (TA, Current_Last, Target.Last + 1, Length (Source));
      J := I + Length (Source);
      while Length (Source) /= 0 loop
         pragma Loop_Variant (Static => (Decreases => J));
         if I = 0 then
            TA (1 .. J) := SA (1 .. Length (Source));
            Source.Last := No_Index;
            exit;
         end if;

         if SA (Length (Source)).V < TA (I).V then
            TA (J) := TA (I);
            I := I - 1;

         else
            TA (J) := SA (Length (Source));
            Source.Last := Source.Last - 1;
         end if;

         J := J - 1;
      end loop;

      Target.Last := Current_Last;
   end;
end Merge;
