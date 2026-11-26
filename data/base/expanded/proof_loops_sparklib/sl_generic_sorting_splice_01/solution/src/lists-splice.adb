--
--  Copyright (C) 2004-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;
separate (Lists)
procedure Splice
  (Target : in out List; Before : Cursor; Source : in out List)
is
   SN : Node_Array renames Source.Nodes;

   Orig_Sum : Count_Type
   with Ghost => Static;
   --  The invariant sum Target.Length + Source.Length (captured once the
   --  guards rule out its overflow), which bounds Target below capacity.

begin
   if Source.Length = 0 then
      return;
   end if;

   Orig_Sum := Target.Length + Source.Length;

   --  Move Source's elements into Target one at a time: insert the first
   --  before Before, then unlink and free it in Source. The running sum
   --  Target.Length + Source.Length stays equal to Orig_Sum, so Target
   --  never overflows and each Insert keeps Before a valid cursor.

   while Source.Length > 1 loop
      pragma Loop_Invariant (Static => Structural_Invariant (Target));
      pragma Loop_Invariant (Static => Structural_Invariant (Source));
      pragma Loop_Invariant (Static => Is_Relevant (Target, Before));
      pragma
        Loop_Invariant (Static => Target.Length + Source.Length = Orig_Sum);
      pragma Loop_Invariant (Static => Orig_Sum <= Target.Capacity);
      pragma Loop_Variant (Static => (Decreases => Source.Length));

      Insert (Target, Before, SN (Source.First).Element);
      declare
         X : constant Count_Type := Source.First;
      begin
         Unlink_First_Node (Source);
         Source.Length := Source.Length - 1;
         Free (Source, X);
      end;
   end loop;

   if Source.Length = 1 then

      --  Insert the last element, then drop the sole node in Source

      Insert (Target, Before, SN (Source.First).Element);
      declare
         X : constant Count_Type := Source.First;
      begin
         Unlink_Sole_Node (Source);
         Source.Length := Source.Length - 1;
         Free (Source, X);
      end;
   end if;
end Splice;
